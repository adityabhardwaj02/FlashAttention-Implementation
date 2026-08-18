# Part B: FlashAttention with Python-level tiling + online softmax
import math
import torch
from common import D, DEVICE, DTYPE, SEQ_LENS, make_qkv, time_fn

M_SRAM = 32 * 1024
Bc = M_SRAM // (4 * D)
Br = min(Bc, D)


def flash_attention_python(Q, K, V):
    Bsz, NH, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Tc = (N + Bc - 1) // Bc
    Tr = (N + Br - 1) // Br

    O = torch.zeros_like(Q, dtype=torch.float32)
    m = torch.full((Bsz, NH, N), float("-inf"), device=Q.device, dtype=torch.float32)
    l = torch.zeros((Bsz, NH, N), device=Q.device, dtype=torch.float32)

    for j in range(Tc):
        c0, c1 = j * Bc, min((j + 1) * Bc, N)
        Kj = K[:, :, c0:c1, :].float()
        Vj = V[:, :, c0:c1, :].float()

        for i in range(Tr):
            r0, r1 = i * Br, min((i + 1) * Br, N)
            Qi = Q[:, :, r0:r1, :].float()

            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale
            m_tilde = Sij.max(dim=-1).values
            P_tilde = torch.exp(Sij - m_tilde.unsqueeze(-1))
            l_tilde = P_tilde.sum(dim=-1)

            m_prev = m[:, :, r0:r1]
            l_prev = l[:, :, r0:r1]
            O_prev = O[:, :, r0:r1, :]

            m_new = torch.maximum(m_prev, m_tilde)
            corr_prev = torch.exp(m_prev - m_new)
            corr_tilde = torch.exp(m_tilde - m_new)

            l_new = corr_prev * l_prev + corr_tilde * l_tilde
            PV = torch.matmul(P_tilde, Vj)
            O_new = O_prev * corr_prev.unsqueeze(-1) + corr_tilde.unsqueeze(-1) * PV

            m[:, :, r0:r1] = m_new
            l[:, :, r0:r1] = l_new
            O[:, :, r0:r1, :] = O_new

    O = O / l.unsqueeze(-1)
    return O.to(Q.dtype)


def run(seq_lens=SEQ_LENS, iters=10, max_N_for_timing=1024):
    results = []
    for N in seq_lens:
        if N > max_N_for_timing:
            continue
        Q, K, V = make_qkv(N)
        ms = time_fn(lambda: flash_attention_python(Q, K, V), iters=iters)
        results.append((N, ms))
        print(f"[Part B] N={N:5d}  {ms:8.3f} ms")
    return results


if __name__ == "__main__":
    run()
