# Part A: standard scaled dot-product attention (baseline)
import math
import torch
from common import D, DEVICE, DTYPE, SEQ_LENS, make_qkv, time_fn


def standard_attention(Q, K, V):
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V)


def run(seq_lens=SEQ_LENS, iters=10):
    results = []
    for N in seq_lens:
        Q, K, V = make_qkv(N)
        ms = time_fn(lambda: standard_attention(Q, K, V), iters=iters)
        results.append((N, ms))
        print(f"[Part A] N={N:5d}  {ms:8.3f} ms  (device={DEVICE}, dtype={DTYPE})")
    return results


if __name__ == "__main__":
    run()
