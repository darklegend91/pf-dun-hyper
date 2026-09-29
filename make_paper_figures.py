"""
=============================================================================
 make_paper_figures.py  --  Parametric studies -> 8 paper figures
=============================================================================

Generates eight figures formatted for IET Communications (single-column,
~8.5 cm wide, serif, grayscale-distinguishable markers/linestyles, no embedded
titles because IET captions sit below the figure).

  Fig. 1  NMSE vs SNR                     (all estimators, +/- std over seeds)
  Fig. 2  NMSE vs pilot overhead  Q/L     (pilot efficiency)
  Fig. 3  NMSE vs vehicle velocity        (Doppler / mobility robustness)
  Fig. 4  NMSE vs federated round         (convergence & communication rounds)
  Fig. 5  NMSE vs unfolding depth  T      (architectural parametric)
  Fig. 6  Payload vs NMSE, rank sweep     (row-compression Pareto)
  Fig. 7  NMSE vs number of clients       (federated scalability)
  Fig. 8  BER vs SNR                      (end-to-end, after DD equalization)

Every figure writes its raw numbers to paper_figures/*.csv and is saved as soon
as its data is ready, so an interrupted run still leaves completed figures.

All training uses a FAIR per-method budget (patience-based early stopping with
best-checkpoint restore), so no estimator is penalised by a round budget chosen
to suit another.

    python make_paper_figures.py                # all eight
    python make_paper_figures.py --figs 1,2,6   # a subset
"""
from __future__ import annotations
import os, csv, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from otfs_data import (make_federation, make_velocity_federation,
                       build_otfs_operator, SCENARIOS)
from models import (PFDUNHyper, CNNEstimator, LAMPNet,
                    ls_estimate, lmmse_estimate, nmse)
from federated import federated_train, evaluate

OUT = "paper_figures"

# --------------------------------------------------------------------------- #
# IET Communications figure style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "legend.framealpha": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
FIGSIZE = (3.5, 2.75)          # single IET column
DPI = 600

# colour + marker + linestyle, chosen to stay distinguishable in greyscale
STYLE = {
    "LS":               dict(c="0.55", m="o", ls="--"),
    "LMMSE":            dict(c="0.25", m="s", ls="-."),
    "CNN":              dict(c="tab:green",  m="^", ls="-"),
    "LAMP":             dict(c="tab:purple", m="P", ls="-"),
    "OAMP-DUN":         dict(c="tab:orange", m="v", ls="--"),
    "PF-DUN-Hyper":     dict(c="tab:blue",   m="D", ls="-"),
    "PF-DUN-Hyper-RC":  dict(c="tab:red",    m="*", ls="-"),
}
NEURAL = ["CNN", "LAMP", "OAMP-DUN", "PF-DUN-Hyper", "PF-DUN-Hyper-RC"]
ALL = ["LS", "LMMSE"] + NEURAL


def plot_line(ax, x, y, name, yerr=None):
    st = STYLE[name]
    if yerr is not None and np.any(np.asarray(yerr) > 0):
        ax.errorbar(x, y, yerr=yerr, color=st["c"], marker=st["m"],
                    linestyle=st["ls"], label=name, capsize=2)
    else:
        ax.plot(x, y, color=st["c"], marker=st["m"], linestyle=st["ls"], label=name)


def save(fig, ax, name, xlabel, ylabel, legend=True, ncol=1):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if legend:
        ax.legend(ncol=ncol, handlelength=2.4)
    fig.set_size_inches(*FIGSIZE)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"  -> saved {OUT}/{name}.png / .pdf", flush=True)


def save_csv(name, header, rows):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# --------------------------------------------------------------------------- #
# model factory + fair training
# --------------------------------------------------------------------------- #
def build(name, N, M, T, Q, rank=16):
    L = N * M
    if name == "CNN":              return CNNEstimator(N, M)
    if name == "LAMP":             return LAMPNet(L, Q, T=T)
    if name == "OAMP-DUN":         return PFDUNHyper(L, T=T, use_hyper=False)
    if name == "PF-DUN-Hyper":     return PFDUNHyper(L, T=T, use_hyper=True)
    if name == "PF-DUN-Hyper-RC":  return PFDUNHyper(L, T=T, use_hyper=True,
                                                     compress=True, compress_rank=rank)
    raise ValueError(name)


def train_fair(name, clients, a, T=None, rank=16, model=None, seed=None):
    """Train one estimator to convergence; return (model, history)."""
    T = T or a.T
    if model is None:
        model = build(name, a.N, a.M, T, clients[0].Q, rank).to(a.device)
    return federated_train(model, clients, a.max_rounds, a.local_epochs, a.lr,
                           a.batch_size, 1.0, a.device,
                           seed=a.seed if seed is None else seed,
                           patience=a.patience, eval_every=a.eval_every)


def classical(clients, device):
    ls, lm = [], []
    for c in clients:
        Phi, y, h = c.Phi.to(device), c.Y[:, 0].to(device), c.H[:, 0].to(device)
        ls.append(10 * np.log10(nmse(ls_estimate(y, Phi), h).item()))
        lm.append(10 * np.log10(nmse(lmmse_estimate(y, Phi, c.sigma2.to(device)), h).item()))
    return float(np.mean(ls)), float(np.mean(lm))


# =========================================================================== #
# Fig. 1 -- NMSE vs SNR  (also supplies Fig. 4 convergence histories)
# =========================================================================== #
def fig1(a, cache):
    print("\n[Fig 1] NMSE vs SNR", flush=True)
    snrs = [float(s) for s in a.snr_list.split(",")]
    acc = {m: {s: [] for s in snrs} for m in ALL}
    hist_store = {}
    for seed in range(a.seeds_fig1):
        for snr in snrs:
            cl = make_federation(a.clients, a.N, a.M, a.Q, snr, a.samples,
                                 frames=1, seed=seed)
            ls, lm = classical(cl, a.device)
            acc["LS"][snr].append(ls); acc["LMMSE"][snr].append(lm)
            for name in NEURAL:
                mdl, hist = train_fair(name, cl, a, seed=seed)
                val, _ = evaluate(mdl, cl, a.device)
                acc[name][snr].append(val)
                if seed == 0 and abs(snr - a.conv_snr) < 1e-9:
                    hist_store[name] = hist
            print(f"  seed{seed} SNR{snr:5.1f}: " +
                  " ".join(f"{m}={acc[m][snr][-1]:6.2f}" for m in ALL), flush=True)
    cache["conv"] = hist_store

    mu = {m: [float(np.mean(acc[m][s])) for s in snrs] for m in ALL}
    sd = {m: [float(np.std(acc[m][s])) for s in snrs] for m in ALL}
    save_csv("fig1_nmse_vs_snr", ["method"] + [f"SNR{s}_mean" for s in snrs]
             + [f"SNR{s}_std" for s in snrs],
             [[m] + [f"{v:.3f}" for v in mu[m]] + [f"{v:.3f}" for v in sd[m]] for m in ALL])

    fig, ax = plt.subplots()
    for m in ALL:
        plot_line(ax, snrs, mu[m], m, yerr=sd[m])
    save(fig, ax, "fig1_nmse_vs_snr", "SNR (dB)", "NMSE (dB)", ncol=2)
    return mu


# =========================================================================== #
# Fig. 2 -- NMSE vs pilot overhead
# =========================================================================== #
def fig2(a, cache):
    print("\n[Fig 2] NMSE vs pilot overhead", flush=True)
    ratios = [float(r) for r in a.pilot_list.split(",")]
    res = {m: [] for m in ALL}
    L = a.N * a.M
    for pr in ratios:
        Q = max(8, int(pr * L))
        cl = make_federation(a.clients, a.N, a.M, Q, a.snr, a.samples, frames=1, seed=0)
        ls, lm = classical(cl, a.device)
        res["LS"].append(ls); res["LMMSE"].append(lm)
        for name in NEURAL:
            mdl, _ = train_fair(name, cl, a)
            val, _ = evaluate(mdl, cl, a.device)
            res[name].append(val)
        print(f"  Q/L={pr:.2f}: " + " ".join(f"{m}={res[m][-1]:6.2f}" for m in ALL), flush=True)

    save_csv("fig2_pilot_overhead", ["method"] + [f"QL{r}" for r in ratios],
             [[m] + [f"{v:.3f}" for v in res[m]] for m in ALL])
    fig, ax = plt.subplots()
    for m in ALL:
        plot_line(ax, ratios, res[m], m)
    save(fig, ax, "fig2_pilot_overhead", "Pilot overhead $Q/L$", "NMSE (dB)", ncol=2)


# =========================================================================== #
# Fig. 3 -- NMSE vs vehicle velocity
# =========================================================================== #
def fig3(a, cache):
    print("\n[Fig 3] NMSE vs velocity", flush=True)
    train_v = [20, 50, 80, 110, 140]
    test_v = [10, 30, 50, 70, 90, 110, 130, 150]
    tr = make_velocity_federation(train_v, a.N, a.M, a.Q, a.snr, a.samples,
                                  frames=1, seed=0)
    te = make_velocity_federation(test_v, a.N, a.M, a.Q, a.snr,
                                  max(64, a.samples // 2), frames=1, seed=7)
    res = {}
    # classical reference
    res["LS"], res["LMMSE"] = [], []
    for c in te:
        ls, lm = classical([c], a.device)
        res["LS"].append(ls); res["LMMSE"].append(lm)
    for name in NEURAL:
        mdl, _ = train_fair(name, tr, a)
        res[name] = [evaluate(mdl, [c], a.device)[0] for c in te]
        print(f"  {name:18s} " + " ".join(f"{v:.0f}:{d:6.2f}"
                                          for v, d in zip(test_v, res[name])), flush=True)

    save_csv("fig3_velocity", ["method"] + [f"v{v}" for v in test_v],
             [[m] + [f"{v:.3f}" for v in res[m]] for m in ALL])
    fig, ax = plt.subplots()
    for m in ALL:
        plot_line(ax, test_v, res[m], m)
    save(fig, ax, "fig3_velocity", "Vehicle velocity (km/h)", "NMSE (dB)", ncol=2)


# =========================================================================== #
# Fig. 4 -- convergence
# =========================================================================== #
def fig4(a, cache):
    print("\n[Fig 4] convergence", flush=True)
    hist = cache.get("conv")
    if not hist:                                   # Fig 1 not run -> train here
        cl = make_federation(a.clients, a.N, a.M, a.Q, a.conv_snr, a.samples,
                             frames=1, seed=0)
        hist = {}
        for name in NEURAL:
            _, h = train_fair(name, cl, a)
            hist[name] = h
    rows = []
    fig, ax = plt.subplots()
    for m in NEURAL:
        if m not in hist or not hist[m]:
            continue
        r = [p[0] for p in hist[m]]; v = [p[1] for p in hist[m]]
        st = STYLE[m]
        ax.plot(r, v, color=st["c"], marker=st["m"], linestyle=st["ls"], label=m)
        rows.append([m] + [f"{x:.3f}" for x in v])
    save_csv("fig4_convergence", ["method", "nmse_per_eval..."], rows)
    save(fig, ax, "fig4_convergence", "Federated round", "NMSE (dB)")


# =========================================================================== #
# Fig. 5 -- unfolding depth T
# =========================================================================== #
def fig5(a, cache):
    print("\n[Fig 5] NMSE vs unfolding depth T", flush=True)
    Ts = [int(t) for t in a.T_list.split(",")]
    methods = ["OAMP-DUN", "PF-DUN-Hyper", "PF-DUN-Hyper-RC"]
    res = {m: [] for m in methods}
    cl = make_federation(a.clients, a.N, a.M, a.Q, a.snr, a.samples, frames=1, seed=0)
    for T in Ts:
        for m in methods:
            mdl, _ = train_fair(m, cl, a, T=T)
            res[m].append(evaluate(mdl, cl, a.device)[0])
        print(f"  T={T:2d}: " + " ".join(f"{m}={res[m][-1]:6.2f}" for m in methods), flush=True)

    save_csv("fig5_depth", ["method"] + [f"T{t}" for t in Ts],
             [[m] + [f"{v:.3f}" for v in res[m]] for m in methods])
    fig, ax = plt.subplots()
    for m in methods:
        plot_line(ax, Ts, res[m], m)
    save(fig, ax, "fig5_depth", "Number of unfolding layers $T$", "NMSE (dB)")


# =========================================================================== #
# Fig. 6 -- row-compression rank sweep (payload vs NMSE)
# =========================================================================== #
def fig6(a, cache):
    print("\n[Fig 6] compression rank sweep", flush=True)
    ranks = [int(r) for r in a.rank_list.split(",")]
    cl = make_federation(a.clients, a.N, a.M, a.Q, a.snr, a.samples, frames=1, seed=0)
    pts, rows = [], []
    mdl, _ = train_fair("PF-DUN-Hyper", cl, a)
    f0 = evaluate(mdl, cl, a.device)[0]
    p0 = sum(p.numel() for p in mdl.parameters())
    pts.append((p0 / 1e3, f0, "full")); rows.append(["full", p0, f"{f0:.3f}"])
    print(f"  full : {p0:>8,} par  {f0:6.2f} dB", flush=True)
    for r in ranks:
        mdl, _ = train_fair("PF-DUN-Hyper-RC", cl, a, rank=r)
        f = evaluate(mdl, cl, a.device)[0]
        p = sum(q.numel() for q in mdl.parameters())
        pts.append((p / 1e3, f, f"r={r}")); rows.append([f"rank{r}", p, f"{f:.3f}"])
        print(f"  r={r:<3d}: {p:>8,} par  {f:6.2f} dB", flush=True)

    save_csv("fig6_rank", ["variant", "params", "nmse_db"], rows)
    fig, ax = plt.subplots()
    xs = [p for p, _, _ in pts]; ys = [f for _, f, _ in pts]
    ax.plot(xs, ys, color="tab:red", marker="*", linestyle="-")
    for x, y, lb in pts:
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6)
    save(fig, ax, "fig6_rank", "Federated payload (10$^3$ parameters/round)",
         "NMSE (dB)", legend=False)


# =========================================================================== #
# Fig. 7 -- federated scalability (number of clients)
# =========================================================================== #
def fig7(a, cache):
    print("\n[Fig 7] NMSE vs number of clients", flush=True)
    counts = [int(c) for c in a.client_list.split(",")]
    methods = ["CNN", "OAMP-DUN", "PF-DUN-Hyper", "PF-DUN-Hyper-RC"]
    res = {m: [] for m in methods}
    for nc in counts:
        # keep the TOTAL data budget fixed so the axis isolates federation
        per = max(48, (a.clients * a.samples) // nc)
        cl = make_federation(nc, a.N, a.M, a.Q, a.snr, per, frames=1, seed=0)
        for m in methods:
            mdl, _ = train_fair(m, cl, a)
            res[m].append(evaluate(mdl, cl, a.device)[0])
        print(f"  K={nc:2d} (n/client={per}): " +
              " ".join(f"{m}={res[m][-1]:6.2f}" for m in methods), flush=True)

    save_csv("fig7_clients", ["method"] + [f"K{c}" for c in counts],
             [[m] + [f"{v:.3f}" for v in res[m]] for m in methods])
    fig, ax = plt.subplots()
    for m in methods:
        plot_line(ax, counts, res[m], m)
    save(fig, ax, "fig7_clients", "Number of federated vehicles $K$", "NMSE (dB)")


# =========================================================================== #
# Fig. 8 -- end-to-end BER
# =========================================================================== #
def fig8(a, cache):
    print("\n[Fig 8] BER vs SNR", flush=True)
    snrs = [float(s) for s in a.ber_snr_list.split(",")]
    L = a.N * a.M
    res = {m: [] for m in ALL}
    for snr in snrs:
        cl = make_federation(a.clients, a.N, a.M, a.Q, snr, a.samples, frames=1, seed=0)
        trained = {}
        for name in NEURAL:
            trained[name], _ = train_fair(name, cl, a)
            trained[name].eval()

        acc = {m: [] for m in ALL}
        for c in cl:
            Phi, sig = c.Phi.to(a.device), c.sigma2.to(a.device)
            n = min(a.ber_samples, len(c))
            h_t = c.H[:n, 0].to(a.device); y_p = c.Y[:n, 0].to(a.device)
            s = c.S[:n].to(a.device)
            est = {"LS": ls_estimate(y_p, Phi),
                   "LMMSE": lmmse_estimate(y_p, Phi, sig)}
            with torch.no_grad():
                for name in NEURAL:
                    est[name] = trained[name](y_p.unsqueeze(1), Phi, sig, s, frames=1)[:, 0]

            rng = np.random.default_rng(11)
            bits = rng.integers(0, 2, size=(n, L, 2))
            xq = ((1 - 2 * bits[:, :, 0]) + 1j * (1 - 2 * bits[:, :, 1])) / np.sqrt(2)
            xd = torch.from_numpy(xq.astype(np.complex64)).to(a.device)

            for i in range(n):
                Hm = torch.from_numpy(build_otfs_operator(
                    h_t[i].cpu().numpy().reshape(a.N, a.M), a.N, a.M)).to(a.device)
                yd = Hm @ xd[i]
                s2 = (yd.abs() ** 2).mean() / (10 ** (snr / 10))
                yd = yd + torch.sqrt(s2 / 2) * (torch.randn_like(yd.real)
                                                + 1j * torch.randn_like(yd.real))
                for m, he in est.items():
                    He = torch.from_numpy(build_otfs_operator(
                        he[i].cpu().numpy().reshape(a.N, a.M), a.N, a.M)).to(a.device)
                    A = He @ He.conj().t() + s2 * torch.eye(L, dtype=He.dtype, device=a.device)
                    xh = He.conj().t() @ torch.linalg.solve(A, yd)
                    b = np.stack([(xh.real < 0).cpu().numpy().astype(int),
                                  (xh.imag < 0).cpu().numpy().astype(int)], -1)
                    acc[m].append((b != bits[i]).mean())
        for m in ALL:
            res[m].append(float(np.mean(acc[m])))
        print(f"  SNR{snr:5.1f}: " + " ".join(f"{m}={res[m][-1]:.4f}" for m in ALL), flush=True)

    save_csv("fig8_ber", ["method"] + [f"SNR{s}" for s in snrs],
             [[m] + [f"{v:.6f}" for v in res[m]] for m in ALL])
    fig, ax = plt.subplots()
    floor = 1.0 / (a.ber_samples * a.clients * L * 2)
    for m in ALL:
        st = STYLE[m]
        ax.semilogy(snrs, np.maximum(res[m], floor), color=st["c"], marker=st["m"],
                    linestyle=st["ls"], label=m)
    ax.set_yscale("log")
    save(fig, ax, "fig8_ber", "SNR (dB)", "Bit error rate", ncol=2)


# =========================================================================== #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figs", default="1,2,3,4,5,6,7,8")
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--pilot_ratio", type=float, default=0.6)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--conv_snr", type=float, default=10.0)
    p.add_argument("--snr_list", default="0,5,10,15,20")
    p.add_argument("--ber_snr_list", default="0,5,10,15")
    p.add_argument("--pilot_list", default="0.3,0.45,0.6,0.75,0.9")
    p.add_argument("--T_list", default="2,4,6,8,12")
    p.add_argument("--rank_list", default="2,4,8,16,32")
    p.add_argument("--client_list", default="2,4,8,12")
    p.add_argument("--seeds_fig1", type=int, default=2)
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ber_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    a.Q = int(a.pilot_ratio * a.N * a.M)
    torch.manual_seed(a.seed)
    print(f"=== paper figures | OTFS {a.N}x{a.M} (L={a.N*a.M}, Q={a.Q}) | "
          f"K={a.clients} | fair budget max_rounds={a.max_rounds} patience={a.patience} "
          f"| {a.device} ===")

    want = [s.strip() for s in a.figs.split(",")]
    cache = {}
    table = {"1": fig1, "2": fig2, "3": fig3, "4": fig4,
             "5": fig5, "6": fig6, "7": fig7, "8": fig8}
    for k in want:                       # honour the requested ORDER, so cheap
        if k in table:                   # figures can be front-loaded and saved
            t0 = time.time()             # before any long-running one starts
            table[k](a, cache)
            print(f"  [Fig {k} done in {time.time()-t0:.0f}s]", flush=True)
    print(f"\nAll requested figures in ./{OUT}/")


if __name__ == "__main__":
    main()
