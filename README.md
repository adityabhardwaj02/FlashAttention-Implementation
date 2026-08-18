# FlashAttention: Implementation and Evaluation

Standard attention, Python-tiled FlashAttention, PyTorch's fused built-in
kernel, and a hand-written CUDA FlashAttention forward kernel with
shared-memory tiling and on-register online softmax.

## Layout

```
common.py            Shared config (B=8, NH=16, D=64), input generation, timing helper
part_a.py             Part A - standard attention (torch.matmul + softmax, full N x N)
part_b.py             Part B - FlashAttention, Python for-loops over tiles, online softmax
part_c.py             Part C - PyTorch built-in fused SDPA (performance upper bound)
part_d.py             Part D - driver that JIT-builds and runs the CUDA kernel
cuda/
  flash_attention_kernel.cu   Custom CUDA kernel (Part D)
  setup.py                    Optional ahead-of-time build script
benchmark_all.py      Runs A/B/C(/D), checks cross-implementation consistency, plots
results/               CSV runtime logs per part
plots/                 runtime_vs_n.png
```

## Running

```bash
python3 benchmark_all.py
```

`common.py` auto-selects `cuda`/`float16` when a GPU is present and falls
back to `cpu`/`float32` otherwise.

## Results (NVIDIA Tesla T4, CUDA 12.8, torch 2.10+cu128, fp16)

### Runtime vs. N (ms)

| N | A: Standard | B: Python-tiled | C: Built-in SDPA | D: Custom CUDA |
|---|---|---|---|---|
| 512 | 2.4 | 20.3 | 1.3 | 124 |
| 1024 | 9.1 | 55.3 | 4.8 | 495 |
| 2048 | 31.8 | skipped | 17.0 | 1,974 |
| 4096 | 142.1 | skipped | 39.7 | 7,924 |
| 8192 | OOM | skipped | 156.9 | 31,611 |

Part A OOMs at N=8192 (materializes the full N x N matrix, O(N^2) memory).
Part B is capped at N=1024 for timing since Python-level tiling is slow by
design (the assignment goal is understanding, not speed). Part D is
correct but ~200x slower than Part C at N=8192 -- it launches only 32
scalar threads per block with no tensor cores or vectorized loads, so the
gap is expected and traceable to occupancy, not a bug.

### Numerical consistency (max abs diff, tolerance < 1e-3)

```
N=32   A-B=1.95e-03  A-C=1.95e-03  B-C=9.77e-04  FAIL
N=64   A-B=1.95e-03  A-C=1.95e-03  B-C=9.77e-04  FAIL
N=128  A-B=1.95e-03  A-C=1.95e-03  B-C=9.77e-04  FAIL
N=256  A-B=9.77e-04  A-C=9.77e-04  B-C=4.88e-04  OK
```

### Part D vs. reference (max abs diff)

```
N=32   diff=1.953e-03  FAIL
N=64   diff=1.953e-03  FAIL
N=128  diff=1.953e-03  FAIL
N=257  diff=9.995e-04  OK
N=512  diff=9.766e-04  OK
```

**Why small-N fails the 1e-3 cutoff:** fp16's machine epsilon near
magnitude 1.0 is 2^-10 ~= 9.77e-4. The observed 1.95e-3 is exactly 2x that
value (2 ULP), and the 9.77e-4 passes are exactly 1 ULP. The deviation is
fp16 rounding, not an algorithmic bug -- it happens to straddle the
assignment's 1e-3 threshold at the smallest sequence lengths. Error does
not grow with N, which is the expected signature of rounding noise rather
than compounding error.

Part B's tiling formula reproduces the paper's worked example exactly:
`M=32768 -> Bc=128, Br=64` for `D=64`.

## Part D kernel design notes

- Grid `(B, NH)`, block `(32)` -- one block per `(b, h)` slice, one thread
  per row within a `Br`-row tile (`Br = Bc = 32`).
- Loop order follows the spec: outer over K/V tiles `j`, inner over Q
  tiles `i`. Running max/sum (`m`, `l`) and the output accumulator
  (`O_acc`) live in global memory across outer iterations, since they must
  persist across all `Tc` steps for every one of the `Tr` row-tiles.
- Shared memory holds only the current `(Qi, Kj, Vj, S)` tiles
  (`Br*d + 2*Bc*d + Br*Bc` floats), with two `__syncthreads()` barriers
  per outer iteration.
- Accumulation is done in `float32` even though `Q/K/V/O` are `float16`,
  for numerical stability; the final pass divides by `l` and casts back
  to `half`.
- Memory is O(N) -- `m_buf`/`l_buf` are size `(B,NH,N)`, `O_acc` is
  `(B,NH,N,d)`, and no N x N buffer is ever allocated, unlike Part A.
- Runtime is still O(N^2), same as every other variant -- FlashAttention
  reduces memory, not the number of operations. Confirmed empirically:
  runtime roughly 4x's with each doubling of N (512->8192 above).
