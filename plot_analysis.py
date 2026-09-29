"""
Render figures from the analysis CSVs in results/.

Kept separate from run_analysis.py so figures can be regenerated (restyled,
relabelled) without re-running any training.

    python plot_analysis.py
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"


def read_csv(name):
    path = f"{OUT}/{name}"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.reader(f))


def plot_ablation():
    rows = read_csv("ablation.csv")
    if not rows:
        return
    labels = [r[0] for r in rows[1:]]
    params = [int(r[1]) for r in rows[1:]]
    nmse = [float(r[2]) for r in rows[1:]]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:gray", "tab:blue", "tab:cyan", "tab:orange", "tab:red"]
    bars = ax.barh(range(len(labels)), nmse, color=colors[:len(labels)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("NMSE (dB) — lower (further left) is better")
    ax.set_title("Component ablation (fair per-method training budget)")
    for i, (b, v, p) in enumerate(zip(bars, nmse, params)):
        ax.text(v / 2, i, f"{v:.2f} dB   ({p/1e3:.0f}K params)",
                va="center", ha="center", fontsize=9, color="white", weight="bold")
    ax.set_xlim(min(nmse) * 1.12, 0)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/ablation.png", dpi=150); plt.close()


def plot_control():
    rows = read_csv("embedding_control.csv")
    if not rows:
        return
    labels = [r[0] for r in rows[1:]]
    vals = [float(r[1]) for r in rows[1:]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["tab:blue" if "correct" in l else
            ("tab:red" if "MISMATCH" in l else "tab:gray") for l in labels]
    ax.bar(range(len(labels)), vals, color=cols)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Control: is the mobility embedding actually used?\n"
                 "(mismatched ≈ correct  ⇒  no genuine personalization)", fontsize=10)
    base = vals[0]
    for i, v in enumerate(vals):                       # label inside each bar
        ax.text(i, v / 2, f"{v:.2f} dB\n({v - base:+.2f})", ha="center",
                va="center", fontsize=9, color="white", weight="bold")
    ax.set_ylim(min(vals) * 1.12, 0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/embedding_control.png", dpi=150); plt.close()


def plot_complexity():
    rows = read_csv("complexity.csv")
    if not rows:
        return
    names = [r[0] for r in rows[1:]]
    payload = [float(r[2]) for r in rows[1:]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["tab:red" if "RC" in n else "tab:blue" if "Hyper" in n else "tab:gray"
            for n in names]
    ax.bar(range(len(names)), payload, color=cols)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
    ax.set_ylabel("Federated payload (MB / round)")
    ax.set_title("Communication cost per federated round (float32 upload)")
    for i, v in enumerate(payload):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/complexity.png", dpi=150); plt.close()


if __name__ == "__main__":
    plot_ablation(); plot_control(); plot_complexity()
    print("figures written to ./results/")
