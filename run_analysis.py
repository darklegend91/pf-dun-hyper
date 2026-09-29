"""
=============================================================================
 run_analysis.py -- Extended evaluation suite
=============================================================================

Beyond the headline NMSE-vs-SNR sweep (run_experiments.py), these studies probe
*why* the method works and where it breaks. Each writes a table + figure into
results/.

  --suite ber          End-to-end BER vs SNR after DD-domain LMMSE equalization.
                       (NMSE is a proxy; BER is what actually matters to a link.)
  --suite pilot        NMSE vs pilot overhead Q/L -- pilot efficiency.
  --suite mobility     NMSE vs vehicle velocity, plus ZERO-SHOT generalization
                       to velocities never seen during training. This is the
                       decisive test of the Doppler hypernetwork.
  --suite ablation     Component ablation: hypernetwork / mobility-aggregation /
                       temporal core / compression, each on and off.
  --suite rank         Row-compression rank sweep: params vs NMSE Pareto front.
  --suite robust       Sensitivity to a NOISY mobility embedding (what if the
                       on-vehicle Doppler estimate is wrong?).
  --suite complexity   Parameter count, payload, and inference latency.
  --suite all          Everything above.

All training uses the FAIR budget (patience-based early stopping) so no method
is penalised by a round budget that suits another.
"""
from __future__ import annotations
import os, csv, time, copy, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from otfs_data import (make_federation, make_velocity_federation,
                       build_otfs_operator, make_pilot)
from models import (PFDUNHyper, CNNEstimator, LAMPNet,
                    ls_estimate, lmmse_estimate, nmse)
from federated import federated_train, evaluate

OUT = "results"
STYLE = {
    "LS": ("--", "o", "gray"), "LMMSE": ("--", "s", "black"),
    "CNN + FedAvg": ("-", "^", "tab:green"), "LAMP + FedAvg": ("-", "P", "tab:purple"),
    "DUN + FedAvg": ("-", "v", "tab:orange"),
    "PF-DUN-Hyper": ("-", "D", "tab:blue"), "PF-DUN-Hyper-RC": ("-", "*", "tab:red"),
}
NEURAL = ["CNN + FedAvg", "LAMP + FedAvg", "DUN + FedAvg",
          "PF-DUN-Hyper", "PF-DUN-Hyper-RC"]


def build(name, N, M, T, Q):
    L = N * M
    if name == "CNN + FedAvg":     return CNNEstimator(N, M)
    if name == "LAMP + FedAvg":    return LAMPNet(L, Q, T=T)
    if name == "DUN + FedAvg":     return PFDUNHyper(L, T=T, use_hyper=False)
    if name == "PF-DUN-Hyper":     return PFDUNHyper(L, T=T, use_hyper=True)
    if name == "PF-DUN-Hyper-RC":  return PFDUNHyper(L, T=T, use_hyper=True,
                                                     compress=True, compress_rank=16)
    raise ValueError(name)


def train_fair(name, clients, a, device, model=None):
    """Train one method to convergence (fair budget) and return it."""
    if model is None:
        model = build(name, a.N, a.M, a.T, clients[0].Q).to(device)
    model, hist = federated_train(model, clients, a.max_rounds, a.local_epochs, a.lr,
                                  a.batch_size, 1.0, device, seed=a.seed,
                                  patience=a.patience, eval_every=a.eval_every)
    return model, hist


def save_csv(path, header, rows):
    with open(f"{OUT}/{path}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# =========================================================================== #
#  SUITE 1: End-to-end BER after DD-domain equalization
# =========================================================================== #
def dd_channel_matrix(h_vec, N, M):
    """
    OTFS twisted convolution is symmetric: y = Phi(x)h = Phi(h)x.
    So the effective channel matrix acting on the DATA symbols is built from h
    exactly like the pilot operator is built from x.
    """
    return build_otfs_operator(h_vec.reshape(N, M), N, M)


@torch.no_grad()
def suite_ber(a, device):
    """QPSK data through the true channel, equalized with each method's estimate."""
    print("\n===== SUITE: end-to-end BER =====")
    snrs = [float(s) for s in a.snr_list.split(",")]
    L, Q = a.N * a.M, int(a.pilot_ratio * a.N * a.M)
    results = {m: [] for m in ["LS", "LMMSE"] + NEURAL}

    for snr in snrs:
        clients = make_federation(a.clients, a.N, a.M, Q, snr, a.samples,
                                  frames=1, seed=a.seed)
        trained = {}
        for name in NEURAL:
            with torch.enable_grad():
                trained[name], _ = train_fair(name, clients, a, device)
            trained[name].eval()

        ber_acc = {m: [] for m in results}
        for c in clients:
            Phi = c.Phi.to(device)
            sigma2 = c.sigma2.to(device)
            n_test = min(a.ber_samples, len(c))
            h_true = c.H[:n_test, 0, :].to(device)
            y_pilot = c.Y[:n_test, 0, :].to(device)
            s = c.S[:n_test].to(device)

            # --- channel estimates from each method ---
            est = {"LS": ls_estimate(y_pilot, Phi),
                   "LMMSE": lmmse_estimate(y_pilot, Phi, sigma2)}
            for name in NEURAL:
                est[name] = trained[name](y_pilot.unsqueeze(1), Phi, sigma2, s,
                                          frames=1)[:, 0, :]

            # --- transmit QPSK data through the TRUE channel, equalize ---
            rng = np.random.default_rng(7 + n_test)
            bits = rng.integers(0, 2, size=(n_test, L, 2))
            xq = ((1 - 2 * bits[:, :, 0]) + 1j * (1 - 2 * bits[:, :, 1])) / np.sqrt(2)
            x_data = torch.from_numpy(xq.astype(np.complex64)).to(device)

            for i in range(n_test):
                Hm = torch.from_numpy(dd_channel_matrix(
                    h_true[i].cpu().numpy(), a.N, a.M)).to(device)
                y_d = Hm @ x_data[i]
                sp = (y_d.abs() ** 2).mean()
                s2 = sp / (10 ** (snr / 10))
                y_d = y_d + torch.sqrt(s2 / 2) * (torch.randn_like(y_d.real)
                                                  + 1j * torch.randn_like(y_d.real))
                for m, h_e in est.items():
                    He = torch.from_numpy(dd_channel_matrix(
                        h_e[i].cpu().numpy(), a.N, a.M)).to(device)
                    # LMMSE equalizer with the ESTIMATED channel
                    A = He @ He.conj().t() + s2 * torch.eye(L, dtype=He.dtype, device=device)
                    x_hat = He.conj().t() @ torch.linalg.solve(A, y_d)
                    b = torch.stack([(x_hat.real < 0).long(), (x_hat.imag < 0).long()], -1)
                    ber_acc[m].append((b.cpu().numpy() != bits[i]).mean())

        for m in results:
            results[m].append(float(np.mean(ber_acc[m])))
        print(f"[BER @ {snr:4.0f} dB] " +
              "  ".join(f"{m.split(' ')[0]}:{results[m][-1]:.4f}" for m in results), flush=True)

    save_csv("ber.csv", ["method"] + [f"SNR{s}" for s in snrs],
             [[m] + [f"{v:.5f}" for v in results[m]] for m in results])
    plt.figure(figsize=(7, 5))
    for m, (ls, mk, col) in STYLE.items():
        plt.semilogy(snrs, np.maximum(results[m], 1e-5), ls + mk, color=col,
                     label=m, linewidth=2, markersize=7)
    plt.xlabel("SNR (dB)"); plt.ylabel("BER"); plt.grid(True, which="both", alpha=0.3)
    plt.title("End-to-end BER after DD-domain LMMSE equalization")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/ber_vs_snr.png", dpi=150); plt.close()
    return results


# =========================================================================== #
#  SUITE 2: pilot overhead
# =========================================================================== #
def suite_pilot(a, device):
    print("\n===== SUITE: pilot overhead =====")
    ratios = [float(x) for x in a.pilot_list.split(",")]
    L = a.N * a.M
    res = {m: [] for m in ["LS", "LMMSE"] + NEURAL}
    for pr in ratios:
        Q = max(8, int(pr * L))
        clients = make_federation(a.clients, a.N, a.M, Q, a.snr, a.samples,
                                  frames=1, seed=a.seed)
        ls_db = np.mean([10 * np.log10(nmse(ls_estimate(c.Y[:, 0].to(device), c.Phi.to(device)),
                                            c.H[:, 0].to(device)).item()) for c in clients])
        lm_db = np.mean([10 * np.log10(nmse(lmmse_estimate(c.Y[:, 0].to(device), c.Phi.to(device),
                                                           c.sigma2.to(device)),
                                            c.H[:, 0].to(device)).item()) for c in clients])
        res["LS"].append(ls_db); res["LMMSE"].append(lm_db)
        for name in NEURAL:
            model, _ = train_fair(name, clients, a, device)
            final, _ = evaluate(model, clients, device)
            res[name].append(final)
        print(f"[pilot {pr:.2f}] " + "  ".join(f"{m.split(' ')[0]}:{res[m][-1]:6.2f}"
                                              for m in res), flush=True)

    save_csv("pilot_overhead.csv", ["method"] + [f"ratio{r}" for r in ratios],
             [[m] + [f"{v:.3f}" for v in res[m]] for m in res])
    plt.figure(figsize=(7, 5))
    for m, (ls, mk, col) in STYLE.items():
        plt.plot(ratios, res[m], ls + mk, color=col, label=m, linewidth=2, markersize=7)
    plt.xlabel("Pilot overhead  Q/L"); plt.ylabel("NMSE (dB)")
    plt.title(f"NMSE vs pilot overhead (SNR = {a.snr} dB)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/pilot_overhead.png", dpi=150); plt.close()
    return res


# =========================================================================== #
#  SUITE 3: mobility sweep + ZERO-SHOT generalization to unseen velocities
# =========================================================================== #
def suite_mobility(a, device):
    print("\n===== SUITE: mobility / velocity generalization =====")
    L, Q = a.N * a.M, int(a.pilot_ratio * a.N * a.M)
    train_v = [20, 40, 60, 100, 120]                 # velocities SEEN in training
    test_v = [10, 30, 50, 70, 80, 90, 110, 130, 140]  # includes UNSEEN + extrapolation

    tr_clients = make_velocity_federation(train_v, a.N, a.M, Q, a.snr, a.samples,
                                          frames=1, seed=a.seed)
    te_clients = make_velocity_federation(test_v, a.N, a.M, Q, a.snr,
                                          max(64, a.samples // 3), frames=1, seed=a.seed + 50)
    res = {}
    for name in NEURAL:
        model, _ = train_fair(name, tr_clients, a, device)
        per_v = []
        for c in te_clients:
            avg, _ = evaluate(model, [c], device)
            per_v.append(avg)
        res[name] = per_v
        print(f"[{name:22s}] " + " ".join(f"{v:.0f}km/h:{d:6.2f}"
                                          for v, d in zip(test_v, per_v)), flush=True)

    save_csv("mobility_generalization.csv", ["method"] + [f"v{v}" for v in test_v],
             [[m] + [f"{v:.3f}" for v in res[m]] for m in res])
    plt.figure(figsize=(7.5, 5))
    for m in NEURAL:
        ls, mk, col = STYLE[m]
        plt.plot(test_v, res[m], ls + mk, color=col, label=m, linewidth=2, markersize=7)
    for v in train_v:
        plt.axvline(v, color="gray", alpha=0.25, linestyle=":")
    plt.xlabel("Vehicle velocity (km/h)   —  dotted lines = velocities seen in training")
    plt.ylabel("NMSE (dB)")
    plt.title("Generalization across mobility (test velocities unseen in training)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/mobility_generalization.png", dpi=150); plt.close()
    return res, test_v


# =========================================================================== #
#  SUITE 4: component ablation
# =========================================================================== #
def suite_ablation(a, device):
    print("\n===== SUITE: ablation =====")
    L, Q = a.N * a.M, int(a.pilot_ratio * a.N * a.M)
    clients = make_federation(a.clients, a.N, a.M, Q, a.snr, a.samples,
                              frames=2, seed=a.seed)
    variants = {
        "OAMP unfolding only (no hyper)": dict(use_hyper=False, mob=True, temporal=False, comp=False),
        "+ Doppler hypernetwork":          dict(use_hyper=True,  mob=True, temporal=False, comp=False),
        "+ hypernet, plain FedAvg agg":    dict(use_hyper=True,  mob=False, temporal=False, comp=False),
        "+ hypernet + temporal core":      dict(use_hyper=True,  mob=True, temporal=True,  comp=False),
        "+ hypernet + row compression":    dict(use_hyper=True,  mob=True, temporal=False, comp=True),
    }
    rows = []
    for label, cfg in variants.items():
        model = PFDUNHyper(L, T=a.T, use_hyper=cfg["use_hyper"],
                           use_temporal=cfg["temporal"], compress=cfg["comp"],
                           compress_rank=16).to(device)
        model, _ = federated_train(model, clients, a.max_rounds, a.local_epochs, a.lr,
                                   a.batch_size, 1.0, device, seed=a.seed,
                                   mobility_agg=cfg["mob"], patience=a.patience,
                                   eval_every=a.eval_every)
        final, _ = evaluate(model, clients, device)
        n_par = sum(p.numel() for p in model.parameters())
        rows.append([label, n_par, f"{final:.3f}"])
        print(f"[{label:34s}] {n_par:>9,}  {final:6.2f} dB", flush=True)
    save_csv("ablation.csv", ["variant", "params", "nmse_db"], rows)
    return rows


# =========================================================================== #
#  SUITE 5: compression rank sweep (Pareto)
# =========================================================================== #
def suite_rank(a, device):
    print("\n===== SUITE: compression rank sweep =====")
    L, Q = a.N * a.M, int(a.pilot_ratio * a.N * a.M)
    clients = make_federation(a.clients, a.N, a.M, Q, a.snr, a.samples,
                              frames=1, seed=a.seed)
    ranks = [int(r) for r in a.rank_list.split(",")]
    rows, pts = [], []
    # uncompressed reference
    m0 = PFDUNHyper(L, T=a.T, use_hyper=True).to(device)
    m0, _ = train_fair("PF-DUN-Hyper", clients, a, device, model=m0)
    f0, _ = evaluate(m0, clients, device)
    p0 = sum(p.numel() for p in m0.parameters())
    rows.append(["full (no compression)", p0, f"{f0:.3f}"]); pts.append((p0, f0, "full"))
    print(f"[full rank] {p0:,} params  {f0:.2f} dB", flush=True)
    for r in ranks:
        m = PFDUNHyper(L, T=a.T, use_hyper=True, compress=True, compress_rank=r).to(device)
        m, _ = train_fair("PF-DUN-Hyper-RC", clients, a, device, model=m)
        f, _ = evaluate(m, clients, device)
        p = sum(q.numel() for q in m.parameters())
        rows.append([f"rank {r}", p, f"{f:.3f}"]); pts.append((p, f, f"r={r}"))
        print(f"[rank {r:3d}] {p:,} params  {f:.2f} dB", flush=True)
    save_csv("rank_sweep.csv", ["variant", "params", "nmse_db"], rows)

    plt.figure(figsize=(7, 5))
    xs = [p for p, _, _ in pts]; ys = [f for _, f, _ in pts]
    plt.plot(xs, ys, "-o", color="tab:red", linewidth=2)
    for x, y, lb in pts:
        plt.annotate(lb, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.xlabel("Federated payload (parameters / round)"); plt.ylabel("NMSE (dB)")
    plt.title("Row-compression Pareto front: payload vs accuracy")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/rank_sweep.png", dpi=150); plt.close()
    return rows


# =========================================================================== #
#  SUITE 6: robustness to a noisy mobility embedding
# =========================================================================== #
def suite_robust(a, device):
    print("\n===== SUITE: robustness to noisy mobility embedding =====")
    L, Q = a.N * a.M, int(a.pilot_ratio * a.N * a.M)
    clients = make_federation(a.clients, a.N, a.M, Q, a.snr, a.samples,
                              frames=1, seed=a.seed)
    model, _ = train_fair("PF-DUN-Hyper", clients, a, device)

    def eval_with(transform, cross=False):
        """
        Evaluate the trained model with a transformed mobility embedding.

        cross=True feeds client i the embedding of a DIFFERENT client (i+1),
        i.e. a genuine scenario mismatch (tunnel vehicle given highway stats).
        Shuffling *within* a client is not a valid control, because all samples
        of one client share a scenario and therefore near-identical embeddings.
        """
        tot = 0.0
        n = len(clients)
        with torch.no_grad():
            for i, c in enumerate(clients):
                Phi, sig = c.Phi.to(device), c.sigma2.to(device)
                h, y, s = c.H.to(device), c.Y.to(device), c.S.to(device)
                if cross:
                    other = clients[(i + 1) % n].S.to(device)
                    reps = (s.shape[0] + other.shape[0] - 1) // other.shape[0]
                    s_use = other.repeat(reps, 1)[:s.shape[0]]
                else:
                    s_use = transform(s)
                hh = model(y, Phi, sig, s_use, frames=c.frames)
                tot += 10 * torch.log10(nmse(hh.reshape(-1, c.L), h.reshape(-1, c.L))).item()
        return tot / len(clients)

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    vals = []
    for eps in noise_levels:
        vals.append(eval_with(lambda s, e=eps: s + e * torch.randn_like(s) * s.abs().mean()))
        print(f"[embedding noise {eps:.2f}] {vals[-1]:6.2f} dB", flush=True)

    # ---- CONTROL EXPERIMENTS: is the embedding actually being USED? ----
    # If shuffling / zeroing s does NOT hurt, then the hypernetwork's gain comes
    # from extra capacity rather than genuine mobility personalization. This is
    # the control a reviewer will demand.
    ctrl = {
        "correct s": vals[0],
        "MISMATCHED s (other client)": eval_with(None, cross=True),   # the real control
        "shuffled s (within client)": eval_with(lambda s: s[torch.randperm(s.shape[0])]),
        "zeroed s": eval_with(lambda s: torch.zeros_like(s)),
        "random s": eval_with(lambda s: torch.rand_like(s)),
    }
    print("\n  -- control: is the mobility embedding used? --")
    for k, v in ctrl.items():
        print(f"  [{k:26s}] {v:6.2f} dB", flush=True)

    save_csv("robustness.csv", ["noise_std"] + [str(e) for e in noise_levels],
             [["PF-DUN-Hyper"] + [f"{v:.3f}" for v in vals]])
    save_csv("embedding_control.csv", ["condition", "nmse_db"],
             [[k, f"{v:.3f}"] for k, v in ctrl.items()])
    plt.figure(figsize=(7, 5))
    plt.plot(noise_levels, vals, "-D", color="tab:blue", linewidth=2, markersize=7)
    plt.xlabel("Relative noise on mobility embedding s"); plt.ylabel("NMSE (dB)")
    plt.title("Robustness to imperfect Doppler/velocity estimates")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/robustness.png", dpi=150); plt.close()
    return noise_levels, vals


# =========================================================================== #
#  SUITE 8: ORACLE per-scenario models -- does personalization have headroom?
# =========================================================================== #
def suite_oracle(a, device):
    """
    The prerequisite test for ANY personalization claim.

    Train a SEPARATE model on each scenario (an oracle that knows the scenario
    and never has to compromise), and compare against one shared model trained
    on all scenarios jointly.

      * oracle clearly better than shared  -> heterogeneity is real, a
        personalization mechanism has something to gain; it is worth fixing.
      * oracle ~= shared                   -> one global estimator is already
        near-optimal for every scenario. There is NOTHING to personalize in
        this setup, and no mechanism can help. The premise must change.
    """
    from otfs_data import SCENARIOS, make_federation as mk
    print("\n===== SUITE: oracle per-scenario vs shared =====")
    L, Q = a.N * a.M, int(a.pilot_ratio * a.N * a.M)
    scen = list(SCENARIOS.keys())

    # --- shared model: trained on all scenarios jointly ---
    all_clients = mk(len(scen), a.N, a.M, Q, a.snr, a.samples, frames=1, seed=a.seed)
    shared, _ = train_fair("PF-DUN-Hyper", all_clients, a, device)

    rows = []
    for i, sc in enumerate(scen):
        one = [all_clients[i]]                       # this scenario's client only
        shared_db, _ = evaluate(shared, one, device)
        oracle_model, _ = train_fair("PF-DUN-Hyper", one, a, device)   # oracle
        oracle_db, _ = evaluate(oracle_model, one, device)
        gap = shared_db - oracle_db                  # >0 means oracle is better
        rows.append([sc, f"{shared_db:.3f}", f"{oracle_db:.3f}", f"{gap:.3f}"])
        print(f"[{sc:12s}] shared {shared_db:7.2f}   oracle {oracle_db:7.2f}   "
              f"headroom {gap:+.2f} dB", flush=True)

    gaps = [float(r[3]) for r in rows]
    mean_gap = float(np.mean(gaps))
    print(f"\n  mean personalization headroom: {mean_gap:+.2f} dB")
    print("  => " + ("heterogeneity is REAL; personalization can help"
                     if mean_gap > 0.5 else
                     "NO headroom: one global model already suffices for every "
                     "scenario, so no personalization mechanism can help here"))
    save_csv("oracle_headroom.csv",
             ["scenario", "shared_nmse_db", "oracle_nmse_db", "headroom_db"], rows)

    plt.figure(figsize=(7.5, 4.5))
    x = np.arange(len(scen)); w = 0.38
    plt.bar(x - w / 2, [float(r[1]) for r in rows], w, label="shared model", color="tab:blue")
    plt.bar(x + w / 2, [float(r[2]) for r in rows], w, label="oracle (per-scenario)",
            color="tab:red")
    plt.xticks(x, scen); plt.ylabel("NMSE (dB)")
    plt.title(f"Personalization headroom: mean {mean_gap:+.2f} dB\n"
              "(bars equal ⇒ nothing to personalize)", fontsize=10)
    plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/oracle_headroom.png", dpi=150); plt.close()
    return rows, mean_gap


# =========================================================================== #
#  SUITE 7: complexity / latency
# =========================================================================== #
def suite_complexity(a, device):
    print("\n===== SUITE: complexity =====")
    L, Q = a.N * a.M, int(a.pilot_ratio * a.N * a.M)
    clients = make_federation(2, a.N, a.M, Q, a.snr, 64, frames=1, seed=a.seed)
    c = clients[0]
    Phi, sig = c.Phi.to(device), c.sigma2.to(device)
    y, s = c.Y[:32].to(device), c.S[:32].to(device)
    rows = []
    for name in NEURAL:
        m = build(name, a.N, a.M, a.T, Q).to(device).eval()
        n_par = sum(p.numel() for p in m.parameters())
        payload_mb = n_par * 4 / 1e6                       # float32 upload per round
        with torch.no_grad():
            m(y, Phi, sig, s, frames=1)                    # warm-up
            t0 = time.time()
            for _ in range(5):
                m(y, Phi, sig, s, frames=1)
            dt = (time.time() - t0) / 5 / y.shape[0] * 1e3  # ms per estimate
        rows.append([name, n_par, f"{payload_mb:.2f}", f"{dt:.3f}"])
        print(f"[{name:22s}] {n_par:>9,} params  {payload_mb:5.2f} MB/round  {dt:.3f} ms/est",
              flush=True)
    save_csv("complexity.csv", ["method", "params", "payload_MB_per_round", "ms_per_estimate"], rows)
    return rows


# =========================================================================== #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="all",
                   choices=["all", "ber", "pilot", "mobility", "ablation",
                            "rank", "robust", "complexity", "oracle"])
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--pilot_ratio", type=float, default=0.6)
    p.add_argument("--clients", type=int, default=6)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--snr_list", default="0,10,20")
    p.add_argument("--pilot_list", default="0.3,0.45,0.6,0.75,0.9")
    p.add_argument("--rank_list", default="2,4,8,16,32")
    p.add_argument("--max_rounds", type=int, default=40)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ber_samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    print(f"=== Analysis suite '{a.suite}' | OTFS {a.N}x{a.M} | fair budget: "
          f"max_rounds={a.max_rounds} patience={a.patience} | {device} ===")

    run = a.suite
    if run in ("all", "complexity"): suite_complexity(a, device)
    if run in ("all", "oracle"):     suite_oracle(a, device)
    if run in ("all", "ablation"):   suite_ablation(a, device)
    if run in ("all", "rank"):       suite_rank(a, device)
    if run in ("all", "robust"):     suite_robust(a, device)
    if run in ("all", "mobility"):   suite_mobility(a, device)
    if run in ("all", "pilot"):      suite_pilot(a, device)
    if run in ("all", "ber"):        suite_ber(a, device)
    print(f"\nOutputs written to ./{OUT}/")


if __name__ == "__main__":
    main()
