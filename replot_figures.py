"""
Re-render the paper figures from the CSVs in paper_figures/ -- no retraining.

Keeping all styling here means the figures can be restyled for a specific
journal template (fonts, size, legend placement, greyscale) as many times as
needed without repeating any of the parametric sweeps.

    python replot_figures.py
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper_figures"

# --------------------------------------------------------------------------- #
# IET Communications style: single column ~8.5 cm, serif, compact legend
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.7,
    "legend.framealpha": 0.85,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.handlelength": 1.9,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
FIGSIZE = (3.45, 2.6)
DPI = 600

STYLE = {
    "LS":              dict(c="0.60", m="o", ls=(0, (4, 2))),
    "LMMSE":           dict(c="0.25", m="s", ls=(0, (6, 1.5, 1, 1.5))),
    "CNN":             dict(c="tab:green",  m="^", ls="-"),
    "LAMP":            dict(c="tab:purple", m="P", ls=(0, (3, 1, 1, 1))),
    "OAMP-DUN":        dict(c="tab:orange", m="v", ls=(0, (5, 2))),
    "PF-DUN-Hyper":    dict(c="tab:blue",   m="D", ls="-"),
    "PF-DUN-Hyper-RC": dict(c="tab:red",    m="*", ls="-"),
}
ORDER = ["LS", "LMMSE", "CNN", "LAMP", "OAMP-DUN", "PF-DUN-Hyper", "PF-DUN-Hyper-RC"]


def read(name):
    p = f"{OUT}/{name}.csv"
    if not os.path.exists(p):
        print(f"  (skip {name}: no csv)")
        return None
    with open(p) as f:
        return list(csv.reader(f))


def finish(fig, ax, name, xlabel, ylabel, legend=True, ncol=2, loc="best"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(ncol=ncol, loc=loc)
    fig.set_size_inches(*FIGSIZE)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"  -> {name}")


def series(rows):
    """rows -> {method: [floats]} using the first column as the method name."""
    out = {}
    for r in rows[1:]:
        try:
            out[r[0]] = [float(v) for v in r[1:]]
        except ValueError:
            pass
    return out


def xs_from_header(rows, prefix, suffix=None):
    """Pull the x values out of column names like 'SNR10.0_mean' or 'QL0.6'."""
    xs = []
    for h in rows[0][1:]:
        if not h.startswith(prefix):
            continue
        if suffix is not None and not h.endswith(suffix):
            continue
        t = h[len(prefix):]
        for s in ("_mean", "_std"):
            if t.endswith(s):
                t = t[: -len(s)]
        try:
            xs.append(float(t))
        except ValueError:
            pass
    return xs


def curve(ax, x, y, name, yerr=None):
    st = STYLE[name]
    if yerr is not None and np.any(np.asarray(yerr) > 1e-9):
        ax.errorbar(x, y, yerr=yerr, color=st["c"], marker=st["m"],
                    linestyle=st["ls"], label=name, capsize=1.8, elinewidth=0.7)
    else:
        ax.plot(x, y, color=st["c"], marker=st["m"], linestyle=st["ls"], label=name)


# --------------------------------------------------------------------------- #
def fig1():
    rows = read("fig1_nmse_vs_snr")
    if not rows: return
    snrs = xs_from_header(rows, "SNR", suffix="_mean")   # means then stds
    fig, ax = plt.subplots()
    for r in rows[1:]:
        m = r[0]
        if m not in STYLE: continue
        vals = [float(v) for v in r[1:]]
        mu, sd = vals[:len(snrs)], vals[len(snrs):len(snrs) * 2]
        curve(ax, snrs, mu, m, yerr=sd if len(sd) == len(snrs) else None)
    finish(fig, ax, "fig1_nmse_vs_snr", "SNR (dB)", "NMSE (dB)", ncol=2)


def fig2():
    rows = read("fig2_pilot_overhead")
    if not rows: return
    x = xs_from_header(rows, "QL")
    d = series(rows)
    fig, ax = plt.subplots()
    for m in ORDER:
        if m in d: curve(ax, x, d[m], m)
    finish(fig, ax, "fig2_pilot_overhead", "Pilot overhead $Q/L$", "NMSE (dB)", ncol=2)


def fig3():
    rows = read("fig3_velocity")
    if not rows: return
    x = xs_from_header(rows, "v")
    d = series(rows)
    fig, ax = plt.subplots()
    for m in ORDER:
        if m in d: curve(ax, x, d[m], m)
    finish(fig, ax, "fig3_velocity", "Vehicle velocity (km/h)", "NMSE (dB)", ncol=2)


def fig4():
    rows = read("fig4_convergence")
    if not rows: return
    d = series(rows)
    fig, ax = plt.subplots()
    for m in ORDER:
        if m not in d: continue
        y = d[m]
        x = list(range(0, 2 * len(y), 2))      # evaluations every 2 rounds
        st = STYLE[m]
        ax.plot(x, y, color=st["c"], marker=st["m"], linestyle=st["ls"], label=m)
    finish(fig, ax, "fig4_convergence", "Federated round", "NMSE (dB)", ncol=1)


def fig5():
    rows = read("fig5_depth")
    if not rows: return
    x = xs_from_header(rows, "T")
    d = series(rows)
    fig, ax = plt.subplots()
    for m in ORDER:
        if m in d: curve(ax, x, d[m], m)
    finish(fig, ax, "fig5_depth", "Number of unfolding layers $T$", "NMSE (dB)", ncol=1)


def fig6():
    rows = read("fig6_rank")
    if not rows: return
    labels = [r[0] for r in rows[1:]]
    par = [float(r[1]) / 1e3 for r in rows[1:]]
    val = [float(r[2]) for r in rows[1:]]
    order = np.argsort(par)
    fig, ax = plt.subplots()
    ax.plot([par[i] for i in order], [val[i] for i in order],
            color="tab:red", marker="*", linestyle="-")
    for i in order:
        ax.annotate(labels[i].replace("rank", "r="), (par[i], val[i]),
                    textcoords="offset points", xytext=(3, 4), fontsize=5.5)
    finish(fig, ax, "fig6_rank", "Federated payload (10$^3$ parameters/round)",
           "NMSE (dB)", legend=False)


def fig7():
    rows = read("fig7_clients")
    if not rows: return
    x = xs_from_header(rows, "K")
    d = series(rows)
    fig, ax = plt.subplots()
    for m in ORDER:
        if m in d: curve(ax, x, d[m], m)
    finish(fig, ax, "fig7_clients", "Number of federated vehicles $K$",
           "NMSE (dB)", ncol=1)


def fig8():
    rows = read("fig8_ber")
    if not rows: return
    x = xs_from_header(rows, "SNR")
    d = series(rows)
    nz = [v for m in d for v in d[m] if v > 0]
    floor = min(nz) / 2 if nz else 1e-5
    fig, ax = plt.subplots()
    for m in ORDER:
        if m not in d: continue
        st = STYLE[m]
        ax.semilogy(x, np.maximum(d[m], floor), color=st["c"], marker=st["m"],
                    linestyle=st["ls"], label=m)
    finish(fig, ax, "fig8_ber", "SNR (dB)", "Bit error rate", ncol=2)


if __name__ == "__main__":
    print("re-rendering figures from CSVs:")
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
    print(f"done -> ./{OUT}/")
