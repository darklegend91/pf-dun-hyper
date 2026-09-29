"""
Entry point: federated OTFS channel estimation with PF-DUN-Hyper.

Examples
--------
# Full hybrid (hypernetwork + temporal core), federated
python train.py --use_hyper --use_temporal --rounds 40

# Ablation: no personalization (FedAvg on plain unfolding)
python train.py --no_hyper --rounds 40

# Quick smoke test
python train.py --rounds 3 --samples 128 --clients 4
"""
from __future__ import annotations
import argparse
import torch

from otfs_data import make_federation, SCENARIOS
from models import PFDUNHyper, ls_estimate, lmmse_estimate, nmse
from federated import federated_train, evaluate


def baselines(clients, device):
    """Classical LS / LMMSE anchors (first frame only)."""
    ls_tot, lm_tot = 0.0, 0.0
    for c in clients:
        Phi = c.Phi.to(device)
        y = c.Y[:, 0, :].to(device)
        h = c.H[:, 0, :].to(device)
        h_ls = ls_estimate(y, Phi)
        h_lm = lmmse_estimate(y, Phi, c.sigma2.to(device))
        ls_tot += 10 * torch.log10(nmse(h_ls, h)).item()
        lm_tot += 10 * torch.log10(nmse(h_lm, h)).item()
    n = len(clients)
    return ls_tot / n, lm_tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8, help="Doppler bins")
    p.add_argument("--M", type=int, default=8, help="delay bins")
    p.add_argument("--pilot_ratio", type=float, default=0.6, help="Q/L")
    p.add_argument("--snr_db", type=float, default=10.0)
    p.add_argument("--clients", type=int, default=8)
    p.add_argument("--samples", type=int, default=512, help="per client")
    p.add_argument("--frames", type=int, default=1, help=">1 enables temporal seq")
    p.add_argument("--T", type=int, default=8, help="unfolding layers")
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--client_frac", type=float, default=1.0)
    p.add_argument("--use_hyper", dest="use_hyper", action="store_true", default=True)
    p.add_argument("--no_hyper", dest="use_hyper", action="store_false")
    p.add_argument("--use_temporal", action="store_true", default=False)
    p.add_argument("--no_mobility_agg", dest="mobility_agg", action="store_false", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    L = args.N * args.M
    Q = int(args.pilot_ratio * L)
    if args.frames > 1:
        args.use_temporal = True

    print(f"=== PF-DUN-Hyper | OTFS {args.N}x{args.M} (L={L}), Q={Q} pilots, "
          f"SNR={args.snr_db}dB ===")
    print(f"clients={args.clients}  hyper={args.use_hyper}  temporal={args.use_temporal}  "
          f"mobility_agg={args.mobility_agg}  device={device}")

    clients = make_federation(args.clients, args.N, args.M, Q, args.snr_db,
                              args.samples, frames=args.frames, seed=args.seed)

    ls_db, lm_db = baselines(clients, device)
    print(f"\n[baselines]  LS: {ls_db:6.2f} dB   LMMSE: {lm_db:6.2f} dB\n")

    model = PFDUNHyper(L, T=args.T, emb_dim=5,
                       use_hyper=args.use_hyper,
                       use_temporal=args.use_temporal).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable params: {n_params:,}  "
          f"(federated payload per round)\n")

    model, history = federated_train(
        model, clients, args.rounds, args.local_epochs, args.lr,
        args.batch_size, args.client_frac, device,
        mobility_agg=args.mobility_agg, seed=args.seed)

    final, by_sc = evaluate(model, clients, device)
    print(f"\n=== FINAL ===")
    print(f"LS {ls_db:.2f} | LMMSE {lm_db:.2f} | PF-DUN-Hyper {final:.2f} dB")
    for k, v in by_sc.items():
        print(f"   {k:12s}: {v:6.2f} dB")


if __name__ == "__main__":
    main()
