# Runs Parts A-D, checks numerical consistency, plots runtime vs N
import csv
import os
import torch
import matplotlib.pyplot as plt

from common import DEVICE, DTYPE, SEQ_LENS, make_qkv, max_abs_diff
import part_a
import part_b
import part_c

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def check_consistency(seq_lens=(32, 64, 128, 256)):
    for N in seq_lens:
        Q, K, V = make_qkv(N)
        Oa = part_a.standard_attention(Q, K, V)
        Ob = part_b.flash_attention_python(Q, K, V)
        Oc = part_c.builtin_flash_attention(Q, K, V)
        d_ab = max_abs_diff(Oa, Ob)
        d_ac = max_abs_diff(Oa, Oc)
        d_bc = max_abs_diff(Ob, Oc)
        ok = all(d < 1e-3 for d in (d_ab, d_ac, d_bc))
        print(f"N={N:5d}  A-B={d_ab:.2e}  A-C={d_ac:.2e}  B-C={d_bc:.2e}  {'OK' if ok else 'FAIL'}")


def write_csv(name, results):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "ms"])
        w.writerows(results)
    return path


def plot_all(results_a, results_b, results_c, results_d=None):
    plt.figure(figsize=(7, 5))
    for label, res in [("A: Standard (PyTorch)", results_a),
                        ("B: FlashAttn (Python loops)", results_b),
                        ("C: FlashAttn (PyTorch built-in)", results_c),
                        ("D: FlashAttn (custom CUDA)", results_d)]:
        if not res:
            continue
        ns = [r[0] for r in res]
        ms = [r[1] for r in res]
        plt.plot(ns, ms, marker="o", label=label)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length N")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Attention runtime vs. N (device={DEVICE}, dtype={DTYPE})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out_path = os.path.join(PLOTS_DIR, "runtime_vs_n.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    check_consistency()

    results_a = part_a.run()
    write_csv("part_a", results_a)

    results_b = part_b.run(max_N_for_timing=1024)
    write_csv("part_b", results_b)

    results_c = part_c.run()
    write_csv("part_c", results_c)

    results_d = None
    if DEVICE == "cuda":
        import part_d
        part_d.verify()
        results_d = part_d.run()
        write_csv("part_d", results_d)

    plot_all(results_a, results_b, results_c, results_d)
