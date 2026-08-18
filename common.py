import time
import torch

B = 8
NH = 16
D = 64
SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def make_qkv(N, batch=B, heads=NH, dim=D, device=DEVICE, dtype=DTYPE, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(batch, heads, N, dim, generator=g).to(device=device, dtype=dtype)
    K = torch.randn(batch, heads, N, dim, generator=g).to(device=device, dtype=dtype)
    V = torch.randn(batch, heads, N, dim, generator=g).to(device=device, dtype=dtype)
    return Q, K, V


def time_fn(fn, iters=10, warmup=2):
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters * 1000.0


def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()
