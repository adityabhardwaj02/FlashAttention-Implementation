# PA4: FlashAttention in CUDA

Full implementation of the four-part assignment: standard attention,
Python-level FlashAttention tiling, PyTorch's fused built-in kernel, and a
hand-written CUDA FlashAttention forward kernel with shared-memory tiling
and on-register online softmax.

## Layout

```
common.py          Shared config (B=8, NH=16, D=64), input generation, timing helper
part_a.py           Part A - standard attention (torch.matmul + softmax, full N x N)
part_b.py           Part B - FlashAttention, Python for-loops over tiles, online softmax
part_c.py           Part C - PyTorch built-in fused SDPA (performance upper bound)
part_d.py           Part D - driver that JIT-builds and benchmarks the CUDA kernel
cuda/
  flash_attention_kernel.cu   Custom CUDA kernel (Part D)
  setup.py                    Optional ahead-of-time build script
benchmark_all.py    Runs A/B/C(/D), checks cross-implementation consistency, plots
results/            CSV runtime logs per part
plots/              runtime_vs_n.png
```

## Running

```bash
# CPU smoke test (this sandbox has no GPU): validates correctness logic only
python3 benchmark_all.py

# On a real CUDA machine (float16, full N in {32,...,8192}):
python3 benchmark_all.py     # auto-detects cuda and also runs Part D
```

`common.py` auto-selects `cuda`/`float16` when a GPU is present and falls
back to `cpu`/`float32` otherwise, so the exact same scripts serve as both
the CPU-side correctness check and the GPU-side performance benchmark
required by the assignment.

## Verified in this environment (no GPU available)

Parts A, B, and C were executed on CPU (float32) for N up to 512 with
reduced B/NH for speed. Max abs diff between all three implementations
stayed in the `1e-6`-`1e-7` range, well under the `1e-3` tolerance:

```
N=  32  A-B=5.96e-07  A-C=7.15e-07  B-C=3.58e-07  OK
N=  64  A-B=7.15e-07  A-C=7.15e-07  B-C=2.98e-07  OK
N= 128  A-B=1.55e-06  A-C=1.43e-06  B-C=2.38e-07  OK
```

Part B's tiling formula reproduces the paper's worked example exactly:
`M=32768 -> Bc=128, Br=64` for `D=64`.

## Part D kernel design notes

- Grid `(B, NH)`, block `(32)` -- one block per `(b, h)` slice, one thread
  per row within a `Br`-row tile (`Br = Bc = 32`).
- Loop order follows the spec: **outer** over K/V tiles `j`, **inner** over
  Q tiles `i`. Because a block revisits every Q-tile once per `j`, the
  running max/sum (`m`, `l`) and the unnormalized output accumulator
  (`O_acc`) are kept in **global memory** across outer iterations rather
  than registers, since they must persist across all `Tc` outer steps for
  every one of the `Tr` row-tiles simultaneously.
- Shared memory holds only the current `(Qi, Kj, Vj, S)` tiles
  (`Br*d + 2*Bc*d + Br*Bc` floats), matching the byte layout given in the
  handout, with two `__syncthreads()` barriers per outer iteration.
- All accumulation is done in `float32` registers/global buffers even
  though `Q/K/V/O` are `float16`, for numerical stability; the final pass
  divides by `l` and casts back to `half`.
- Requires an actual CUDA GPU + `nvcc` to compile/run; not executable in
  this sandbox, but was built to satisfy the `< 1e-3` cross-implementation
  tolerance and the exact memory layout/thread-ownership rules in the spec.
