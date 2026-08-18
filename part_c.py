# Part C: PyTorch's built-in fused FlashAttention (performance upper bound)
import torch
import torch.nn.functional as F
from common import DEVICE, DTYPE, SEQ_LENS, make_qkv, time_fn


def builtin_flash_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=False)


def run(seq_lens=SEQ_LENS, iters=10):
    results = []
    for N in seq_lens:
        Q, K, V = make_qkv(N)
        ms = time_fn(lambda: builtin_flash_attention(Q, K, V), iters=iters)
        results.append((N, ms))
        print(f"[Part C] N={N:5d}  {ms:8.3f} ms  (device={DEVICE}, dtype={DTYPE})")
    return results


if __name__ == "__main__":
    run()
