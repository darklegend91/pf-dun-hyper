"""
Full experiment suite for publication (multi-seed, true OTFS operator).

Produces (in results/):
  nmse_vs_snr.png     - NMSE (dB) vs SNR, all methods, with std error bars
  convergence.png     - NMSE vs FL round at a fixed SNR (seed 0)
  comm_cost.png       - final NMSE vs federated payload (params) trade-off
  results_table.md    - comparison table (mean +/- std) + per-scenario
  results.csv         - raw numbers (mean and std)

Methods:
  LS, LMMSE                          (classical anchors)
  CNN + FedAvg                       (deep-learning baseline, ChannelNet-style)
  DUN + FedAvg (no personalization)  (ablation)
  PF-DUN-Hyper                       (proposed: + Doppler hypernetwork)
  PF-DUN-Hyper + Temporal            (proposed full hybrid)
"""
from __future__ import annotations
import os, csv, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from otfs_data import make_federation
from models import (PFDUNHyper, CNNEstimator, LAMPNet,
                    ls_estimate, lmmse_estimate, nmse)
from federated import federated_train, evaluate

OUT = "results"
NEURAL_METHODS = ["CNN + FedAvg", "LAMP + FedAvg", "DUN + FedAvg",
                  "PF-DUN-Hyper", "PF-DUN-Hyper-RC"]
ALL_METHODS = ["LS", "LMMSE"] + NEURAL_METHODS


def classical_nmse(clients, device):
    ls_tot, lm_tot = 0.0, 0.0
    for c in clients:
        Phi = c.Phi.to(device)
        y = c.Y[:, 0, :].to(device); h = c.H[:, 0, :].to(device)
        ls_tot += 10 * torch.log10(nmse(ls_estimate(y, Phi), h)).item()
        lm_tot += 10 * torch.log10(nmse(lmmse_estimate(y, Phi, c.sigma2.to(device)), h)).item()
    n = len(clients)
    return ls_tot / n, lm_tot / n


def build(name, N, M, T, Q):
    """Factory: map a method name to a fresh model instance."""
    L = N * M
    if name == "CNN + FedAvg":
        return CNNEstimator(N, M)
    if name == "LAMP + FedAvg":                       # learned-AMP unfolding baseline
        return LAMPNet(L, Q, T=T)
    if name == "DUN + FedAvg":                         # OAMP unfolding, no personalization
        return PFDUNHyper(L, T=T, use_hyper=False, use_temporal=False)
    if name == "PF-DUN-Hyper":                         # proposed: full-rank hypernetwork
        return PFDUNHyper(L, T=T, use_hyper=True, use_temporal=False)
    if name == "PF-DUN-Hyper-RC":                      # proposed + row-compressed payload
        return PFDUNHyper(L, T=T, use_hyper=True, use_temporal=False,
                          compress=True, compress_rank=16)
    raise ValueError(name)


def run_one(name, clients, args, device, log=False):
    model = build(name, args.N, args.M, args.T, clients[0].Q).to(device)
    model, hist = federated_train(
        model, clients, args.rounds, args.local_epochs, args.lr,
        args.batch_size, args.client_frac, device,
        mobility_agg=True, seed=args.seed,
        log_every=(max(1, args.rounds // 5) if log else args.rounds),
        patience=args.patience, eval_every=args.eval_every)
    final, by_sc = evaluate(model, clients, device)
    n_params = sum(p.numel() for p in model.parameters())
    return final, by_sc, hist, n_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--pilot_ratio", type=float, default=0.6)
    p.add_argument("--clients", type=int, default=8)
    p.add_argument("--samples", type=int, default=384)
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--client_frac", type=float, default=1.0)
    p.add_argument("--snr_list", type=str, default="0,5,10,15,20")
    p.add_argument("--conv_snr", type=float, default=10.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--row_compress", type=float, default=1.0,
                   help="fraction of observed rows to keep (<1 = faster, less overhead)")
    p.add_argument("--patience", type=int, default=4,
                   help="FAIR BUDGET: stop a method after this many non-improving "
                        "evals (0 = fixed --rounds for every method, which unfairly "
                        "penalises slow-converging baselines)")
    p.add_argument("--eval_every", type=int, default=2)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = args.N * args.M
    Q = int(args.pilot_ratio * L)
    snr_list = [float(x) for x in args.snr_list.split(",")]
    seeds = list(range(args.seeds))

    print(f"=== Suite | OTFS {args.N}x{args.M} L={L} Q={Q} (true operator) | "
          f"clients={args.clients} rounds={args.rounds} frames={args.frames} "
          f"seeds={args.seeds} | {device} ===")

    # store[method][snr] = list over seeds
    store = {m: {s: [] for s in snr_list} for m in ALL_METHODS}
    params = {}
    conv_hist = {}          # method -> history (seed 0, conv_snr)
    by_scen = {}            # method -> {scenario: [over seeds]}

    def finalize(n_done):
        """Write tables + figures from the seeds completed so far.

        Called after EVERY seed so a long run that is interrupted still leaves
        valid (fewer-seed) results on disk instead of nothing.
        """
        args.seeds_done = n_done
        stats = {m: {s: (float(np.mean(store[m][s])), float(np.std(store[m][s])))
                     for s in snr_list} for m in ALL_METHODS}
        write_table(stats, params, snr_list, by_scen, args)
        plot_snr(stats, snr_list)
        plot_convergence(conv_hist)
        plot_comm_cost(stats, params, args.conv_snr)
        return stats

    for si in seeds:
        args.seed = si
        torch.manual_seed(si)
        print(f"\n################## SEED {si} ##################")
        for snr in snr_list:
            clients = make_federation(args.clients, args.N, args.M, Q, snr,
                                      args.samples, frames=args.frames, seed=si,
                                      row_compress=args.row_compress)
            ls_db, lm_db = classical_nmse(clients, device)
            store["LS"][snr].append(ls_db); store["LMMSE"][snr].append(lm_db)
            print(f"[seed {si} | SNR {snr:4.0f}] LS {ls_db:6.2f} LMMSE {lm_db:6.2f}", flush=True)
            for name in NEURAL_METHODS:
                log = (si == 0 and snr == args.conv_snr)
                final, by_sc, hist, n_params = run_one(name, clients, args, device, log=log)
                store[name][snr].append(final)
                params[name] = n_params
                if si == 0 and snr == args.conv_snr:
                    conv_hist[name] = hist
                    by_scen[name] = {k: [v] for k, v in by_sc.items()}
                elif snr == args.conv_snr and name in by_scen:
                    for k, v in by_sc.items():
                        by_scen[name][k].append(v)
                print(f"    {name:26s} SNR {snr:4.0f}: {final:6.2f} dB", flush=True)

        finalize(si + 1)                      # checkpoint after every seed
        print(f"### checkpoint written after seed {si} "
              f"({si + 1}/{len(seeds)} seeds) ###", flush=True)

    stats = {m: {s: (float(np.mean(store[m][s])), float(np.std(store[m][s])))
                 for s in snr_list} for m in ALL_METHODS}
    write_table(stats, params, snr_list, by_scen, args)
    plot_snr(stats, snr_list)
    plot_convergence(conv_hist)
    plot_comm_cost(stats, params, args.conv_snr)
    print(f"\nAll outputs in ./{OUT}/")


def write_table(stats, params, snr_list, by_scen, args):
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        head = ["method", "params"]
        for s in snr_list:
            head += [f"SNR{s}_mean", f"SNR{s}_std"]
        w.writerow(head)
        for m in ALL_METHODS:
            row = [m, params.get(m, 0)]
            for s in snr_list:
                mu, sd = stats[m][s]; row += [f"{mu:.3f}", f"{sd:.3f}"]
            w.writerow(row)

    lines = [f"# PF-DUN-Hyper — Results (NMSE dB, mean ± std over seeds; lower is better)\n",
             f"OTFS {args.N}×{args.M} (L={args.N*args.M}, **true twisted-conv operator**), "
             f"pilot ratio {args.pilot_ratio}, {args.clients} non-IID clients, "
             f"{args.rounds} FL rounds, {args.frames} frames, "
             f"{getattr(args, 'seeds_done', args.seeds)} seeds.\n",
             "| Method | Params | " + " | ".join(f"{s} dB" for s in snr_list) + " |",
             "|" + "---|" * (len(snr_list) + 2)]
    for m in ALL_METHODS:
        prm = f"{params.get(m,0):,}" if params.get(m, 0) else "—"
        cells = " | ".join(f"{stats[m][s][0]:.2f} ± {stats[m][s][1]:.2f}" for s in snr_list)
        lines.append(f"| {m} | {prm} | {cells} |")
    lines.append(f"\n## Per-scenario NMSE (dB) at {args.conv_snr} dB SNR (mean over seeds)\n")
    if by_scen:
        scs = list(next(iter(by_scen.values())).keys())
        lines.append("| Method | " + " | ".join(scs) + " |")
        lines.append("|" + "---|" * (len(scs) + 1))
        for m in NEURAL_METHODS:
            if m in by_scen:
                lines.append(f"| {m} | " + " | ".join(f"{np.mean(by_scen[m][sc]):.2f}" for sc in scs) + " |")
    with open(f"{OUT}/results_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


STYLE = {
    "LS": ("--", "o", "gray"), "LMMSE": ("--", "s", "black"),
    "CNN + FedAvg": ("-", "^", "tab:green"), "LAMP + FedAvg": ("-", "P", "tab:purple"),
    "DUN + FedAvg": ("-", "v", "tab:orange"),
    "PF-DUN-Hyper": ("-", "D", "tab:blue"), "PF-DUN-Hyper-RC": ("-", "*", "tab:red"),
}


def plot_snr(stats, snr_list):
    plt.figure(figsize=(7, 5))
    for m, (ls, mk, col) in STYLE.items():
        mus = [stats[m][s][0] for s in snr_list]
        sds = [stats[m][s][1] for s in snr_list]
        plt.errorbar(snr_list, mus, yerr=sds, fmt=ls + mk, color=col, label=m,
                     linewidth=2, markersize=7, capsize=3)
    plt.xlabel("SNR (dB)"); plt.ylabel("NMSE (dB)")
    plt.title("Federated OTFS Channel Estimation: NMSE vs SNR")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/nmse_vs_snr.png", dpi=150); plt.close()


def plot_convergence(conv_hist):
    if not conv_hist:
        return
    plt.figure(figsize=(7, 5))
    for m, hist in conv_hist.items():
        plt.plot([r for r, _ in hist], [v for _, v in hist], marker=".",
                 color=STYLE[m][2], label=m, linewidth=2)
    plt.xlabel("Federated round"); plt.ylabel("NMSE (dB)")
    plt.title("Convergence (SNR = 10 dB, seed 0)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/convergence.png", dpi=150); plt.close()


def plot_comm_cost(stats, params, snr):
    plt.figure(figsize=(7, 5))
    for m in NEURAL_METHODS:
        plt.scatter(params[m], stats[m][snr][0], s=120, color=STYLE[m][2], label=m, zorder=3)
        plt.annotate(m, (params[m], stats[m][snr][0]), textcoords="offset points",
                     xytext=(6, 6), fontsize=8)
    plt.xlabel("Federated payload (parameters / round)")
    plt.ylabel(f"NMSE (dB) @ {snr} dB SNR")
    plt.title("Communication cost vs accuracy")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/comm_cost.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
