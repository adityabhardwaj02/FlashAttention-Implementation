# Part D driver: builds the custom CUDA kernel and benchmarks/verifies it
import os
import torch
from torch.utils.cpp_extension import load
from common import DEVICE, DTYPE, SEQ_LENS, make_qkv, time_fn, max_abs_diff
from part_a import standard_attention

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

flash_attention_cuda = load(
    name="flash_attention_cuda",
    sources=[os.path.join(_THIS_DIR, "cuda", "flash_attention_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


def cuda_flash_attention(Q, K, V):
    return flash_attention_cuda.forward(Q, K, V)


def verify(seq_lens=(32, 64, 128, 257, 512)):
    for N in seq_lens:
        Q, K, V = make_qkv(N)
        O_ref = standard_attention(Q, K, V)
        O_cuda = cuda_flash_attention(Q, K, V)
        diff = max_abs_diff(O_ref, O_cuda)
        status = "OK" if diff < 1e-3 else "FAIL"
        print(f"[Part D verify] N={N:5d}  max_abs_diff={diff:.3e}  {status}")


def run(seq_lens=SEQ_LENS, iters=10):
    results = []
    for N in seq_lens:
        Q, K, V = make_qkv(N)
        ms = time_fn(lambda: cuda_flash_attention(Q, K, V), iters=iters)
        results.append((N, ms))
        print(f"[Part D] N={N:5d}  {ms:8.3f} ms  (device={DEVICE}, dtype={DTYPE})")
    return results


if __name__ == "__main__":
    assert DEVICE == "cuda", "Part D requires a CUDA GPU."
    verify()
    run()
