"""
System-model / architecture figure for the paper (IET Communications style).

Produces paper_figures/fig0_architecture.{png,pdf} -- a two-panel block diagram:

  (a) Federated system: K non-IID vehicles, what is exchanged with the server,
      and the three-stage training procedure.
  (b) Per-vehicle estimator: the mobility embedding drives a hypernetwork that
      generates the conditioning parameters of an unrolled OAMP network.

    python make_architecture.py
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "paper_figures"
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

C_VEH = "#dce9f7"      # vehicles / inputs
C_SRV = "#f7ddd9"      # server
C_HYP = "#e6dcf5"      # hypernetwork
C_EST = "#dff0e2"      # unfolding estimator
C_OUT = "#fdf0d0"      # outputs
EDGE = "#3b3b3b"


def box(ax, x, y, w, h, text, fc, fs=7, bold=False, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={r}",
                                linewidth=0.7, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p, q, text=None, style="-|>", rad=0.0, fs=6, dashed=False,
          tx=0.0, ty=0.0, color=EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7,
                                 linewidth=0.7, color=color,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text,
                ha="center", va="center", fontsize=fs, color=color,
                bbox=dict(fc="white", ec="none", pad=0.6))


# --------------------------------------------------------------------------- #
def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(a)  Federated system", fontsize=8, pad=3)

    # ---- server ----
    box(ax, 0.13, 0.80, 0.74, 0.14,
        "Server\n" r"shared denoiser $W$   ·   hypernetwork $g_\phi$",
        C_SRV, fs=6.6)

    # ---- vehicles ----
    names = ["urban", "highway", "rural", "tunnel"]
    subs = ["10–40", "90–140", "50–90", "30–70"]
    for i, (n, sv) in enumerate(zip(names, subs)):
        x = 0.012 + i * 0.249
        box(ax, x, 0.24, 0.215, 0.175,
            f"Vehicle {i+1}\n{n}\n{sv} km/h", C_VEH, fs=5.8)

    # ---- exchange arrows ----
    arrow(ax, (0.33, 0.795), (0.33, 0.425),
          r"broadcast" "\n" r"$W,\ \phi$", fs=5.9, tx=-0.105)
    arrow(ax, (0.67, 0.425), (0.67, 0.795),
          r"upload" "\n" r"$\Delta W,\ \theta_i^{\star}$", fs=5.9, tx=0.115)

    # ---- privacy note ----
    ax.text(0.5, 0.175, r"each vehicle keeps its own $\{\mathbf{y}_i,\mathbf{s}_i\}$",
            ha="center", va="center", fontsize=5.9, style="italic", color="#555")
    ax.text(0.5, 0.115, "raw channel data never leaves the vehicle",
            ha="center", va="center", fontsize=6.1, style="italic", color="#a03030")

    # ---- training stages ----
    ax.text(0.5, 0.035,
            r"(1) FedAvg shared $W$    (2) fit per-vehicle $\theta_i^{\star}$"
            "\n"
            r"(3) distil $g_\phi:\ \mathbf{s}_i \mapsto \theta_i^{\star}$",
            ha="center", va="center", fontsize=6.1, linespacing=1.4)


def panel_b(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b)  Per-vehicle conditioned unfolding estimator",
                 fontsize=8, pad=3)

    # ---- mobility embedding -> hypernetwork -> theta ----
    box(ax, 0.005, 0.805, 0.335, 0.145,
        "mobility embedding\n"
        r"$\mathbf{s}=[\nu_{\max},\tau_{\rm rms},{\rm SNR},v,K]$", C_VEH, fs=5.9)
    box(ax, 0.395, 0.805, 0.215, 0.145,
        "hypernetwork\n" r"$g_\phi(\mathbf{s})$", C_HYP, fs=6.6)
    box(ax, 0.665, 0.805, 0.33, 0.145,
        r"$\theta=\{\gamma_t,\ \lambda_t,\ {\rm FiLM}_t,$" "\n"
        r"LoRA $A(\mathbf{s}),B(\mathbf{s})\}$", C_OUT, fs=5.9)
    arrow(ax, (0.340, 0.877), (0.395, 0.877))
    arrow(ax, (0.610, 0.877), (0.665, 0.877))

    # conditioning arrow down into the estimator
    arrow(ax, (0.83, 0.802), (0.83, 0.675), dashed=True)
    ax.text(0.845, 0.738, "conditions\nevery layer", fontsize=5.4,
            ha="left", va="center", color="#555")

    # ---- inputs ----
    box(ax, 0.005, 0.475, 0.195, 0.125,
        "pilot obs.\n" r"$\mathbf{y}$", C_VEH, fs=6.2)
    box(ax, 0.005, 0.300, 0.195, 0.125,
        "OTFS operator\n" r"$\mathbf{\Phi}$", C_VEH, fs=5.7)

    # ---- unrolled estimator ----
    box(ax, 0.245, 0.235, 0.735, 0.44, "", C_EST, r=0.02)
    ax.text(0.612, 0.632, r"unrolled OAMP,  $t=1\ldots T$",
            ha="center", va="center", fontsize=6.6, weight="bold")

    box(ax, 0.272, 0.410, 0.315, 0.165,
        "linear step\n"
        r"$\mathbf{r}_t=\hat{\mathbf{h}}+\gamma_t\mathbf{W}_t"
        r"(\mathbf{y}-\mathbf{\Phi}\hat{\mathbf{h}})$", "white", fs=5.5)
    box(ax, 0.638, 0.410, 0.315, 0.165,
        "conditioned denoiser\n"
        r"shrink$(\lambda_t)$ + FiLM + LoRA", "white", fs=5.5)
    arrow(ax, (0.587, 0.492), (0.638, 0.492))

    # recurrence
    arrow(ax, (0.945, 0.405), (0.285, 0.405), rad=-0.26)
    ax.text(0.615, 0.288, r"$\hat{\mathbf{h}}^{(t)}\rightarrow\hat{\mathbf{h}}^{(t+1)}$",
            ha="center", va="center", fontsize=5.7, color="#444",
            bbox=dict(fc=C_EST, ec="none", pad=1.2))

    arrow(ax, (0.200, 0.537), (0.272, 0.515))
    arrow(ax, (0.200, 0.362), (0.272, 0.452))

    # ---- output ----
    box(ax, 0.40, 0.045, 0.42, 0.115,
        r"estimated DD channel  $\hat{\mathbf{h}}$", C_OUT, fs=6.6)
    arrow(ax, (0.61, 0.232), (0.61, 0.162))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.85),
                             gridspec_kw={"width_ratios": [1.0, 1.30]})
    panel_a(axes[0])
    panel_b(axes[1])
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig0_architecture.{ext}", dpi=600)
    plt.close(fig)
    print(f"wrote {OUT}/fig0_architecture.png / .pdf")


if __name__ == "__main__":
    main()
