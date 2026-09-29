"""
Does the mobility conditioning actually personalise?

For each model variant we report:
  shared      : NMSE with the CORRECT mobility embedding
  mismatched  : NMSE when each client is given ANOTHER client's embedding
  gap         : mismatched - shared.  This is the real test. If the gap is ~0
                the embedding is decorative; a large gap means the estimator
                genuinely specialises to the vehicle's mobility state.
  oracle      : NMSE of a separate model trained per scenario (the ceiling)
  captured    : fraction of the oracle headroom the shared model recovers

    python test_personalization.py --variant strong
"""
from __future__ import annotations
import argparse, copy
import numpy as np
import torch

from otfs_data import make_federation, SCENARIOS
from models import PFDUNHyper, nmse
from federated import federated_train, federated_train_phn, evaluate


def train(clients, a, phn=False, **kw):
    m = PFDUNHyper(a.N * a.M, T=a.T, hidden_denoiser=a.hidden,
                   use_hyper=True, **kw).to(a.device)
    if phn:
        m, _ = federated_train_phn(m, clients, a.max_rounds, 1, a.lr, a.batch_size,
                                   a.device, seed=a.seed, patience=a.patience,
                                   eval_every=a.eval_every)
    else:
        m, _ = federated_train(m, clients, a.max_rounds, 1, a.lr, a.batch_size, 1.0,
                               a.device, seed=a.seed, patience=a.patience,
                               eval_every=a.eval_every)
    return m


@torch.no_grad()
def eval_mismatched(model, clients, device):
    """Give client i the embedding of client i+1 (a genuine scenario mismatch)."""
    tot, n = 0.0, len(clients)
    for i, c in enumerate(clients):
        Phi, sig = c.Phi.to(device), c.sigma2.to(device)
        h, y = c.H.to(device), c.Y.to(device)
        other = clients[(i + 1) % n].S.to(device)
        s_use = other.repeat((c.S.shape[0] + other.shape[0] - 1) // other.shape[0], 1)[:c.S.shape[0]]
        hh = model(y, Phi, sig, s_use, frames=c.frames)
        tot += 10 * torch.log10(nmse(hh.reshape(-1, c.L), h.reshape(-1, c.L))).item()
    return tot / n


def film_oracle(model, clients, a, steps=60):
    """
    Upper bound on what FiLM-style conditioning can achieve.

    Freeze the shared denoiser and optimise ONLY the per-client conditioning
    parameters theta_i directly on each client's data (no hypernetwork, no
    generalisation requirement -- a cheat). If this recovers most of the oracle
    headroom, the conditioning mechanism is expressive enough and the problem is
    learning the mapping s -> theta. If it recovers little, FiLM modulation of a
    shared denoiser is fundamentally too weak and no hypernetwork can fix it.
    """
    tot = 0.0
    for c in clients:
        local = copy.deepcopy(model).to(a.device)
        for p in local.parameters():
            p.requires_grad_(False)
        s_mean = c.S.mean(0).to(a.device)
        theta = {k: v.clone().requires_grad_(True)
                 for k, v in local.gen_cond(s_mean).items()}
        opt = torch.optim.Adam(list(theta.values()), lr=1e-2)
        Phi, sig = c.Phi.to(a.device), c.sigma2.to(a.device)
        n = len(c)
        for _ in range(steps):
            idx = torch.randperm(n)[:a.batch_size]
            h, y, _ = c.batch(idx)
            h, y = h.to(a.device), y.to(a.device)
            hh = local(y, Phi, sig, None, frames=c.frames, cond=theta)
            loss = nmse(hh.reshape(-1, c.L), h.reshape(-1, c.L))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            h, y = c.H.to(a.device), c.Y.to(a.device)
            hh = local(y, Phi, sig, None, frames=c.frames, cond=theta)
            tot += 10 * torch.log10(nmse(hh.reshape(-1, c.L), h.reshape(-1, c.L))).item()
    return tot / len(clients)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--pilot_ratio", type=float, default=0.6)
    p.add_argument("--samples", type=int, default=160)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--max_rounds", type=int, default=25)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cond_rank", type=int, default=4)
    p.add_argument("--variants", default="weak,strong")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    Q = int(a.pilot_ratio * a.N * a.M)
    scen = list(SCENARIOS.keys())

    clients = make_federation(len(scen), a.N, a.M, Q, a.snr, a.samples,
                              frames=1, seed=a.seed)

    # ---- oracle ceiling: one model per scenario ----
    print("training per-scenario oracles (ceiling) ...", flush=True)
    oracle = []
    for c in clients:
        m = train([c], a, strong_cond=False)
        oracle.append(evaluate(m, [c], a.device)[0])
    oracle_mean = float(np.mean(oracle))
    print(f"  oracle mean = {oracle_mean:.2f} dB  "
          f"({', '.join(f'{s}:{v:.2f}' for s, v in zip(scen, oracle))})\n")

    cfg = {
        "weak":       dict(phn=False, strong_cond=False),
        "strong":     dict(phn=False, strong_cond=True, cond_rank=a.cond_rank),
        "phn":        dict(phn=True,  strong_cond=False),
        "phn+strong": dict(phn=True,  strong_cond=True, cond_rank=a.cond_rank),
    }
    # reference point: FedAvg, no personalisation benefit expected
    ref = evaluate(train(clients, a, phn=False, strong_cond=False), clients, a.device)[0]
    head = ref - oracle_mean
    print(f"reference (FedAvg, weak cond) = {ref:.2f} dB;  headroom to oracle "
          f"= {head:.2f} dB\n")

    ref_model = train(clients, a, phn=False, strong_cond=False)
    fo = film_oracle(ref_model, clients, a)
    print(f"ceiling  FiLM-only  (cheating per-client fit) = {fo:6.2f} dB "
          f"-> recovers {(ref-fo)/head*100:5.1f}% of headroom")
    strong_model = train(clients, a, phn=False, strong_cond=True,
                         cond_rank=a.cond_rank)
    ref_s = evaluate(strong_model, clients, a.device)[0]
    fos = film_oracle(strong_model, clients, a)
    print(f"ceiling  FiLM+LoRA (cheating per-client fit) = {fos:6.2f} dB "
          f"-> recovers {(ref_s-fos)/head*100:5.1f}% of headroom "
          f"(its own base {ref_s:.2f} dB)\n")

    # ---- DISTILLATION ROUTE: train normally -> fit per-client theta -> distil ----
    print("distillation route (FedAvg + per-client adapters + hypernet distil):")
    dm = train(clients, a, phn=False, strong_cond=True, cond_rank=a.cond_rank)
    base_db = evaluate(dm, clients, a.device)[0]
    tg = fit_local_thetas(dm, clients, a)
    dm, rl = distil_hypernet(dm, tg, a)
    d_shared = evaluate(dm, clients, a.device)[0]
    d_mism = eval_mismatched(dm, clients, a.device)
    print(f"  base(FedAvg,strong) {base_db:6.2f} dB | after distil {d_shared:6.2f} dB "
          f"| mismatched {d_mism:6.2f} dB | gap {d_mism-d_shared:+.2f} dB "
          f"| captured {(ref-d_shared)/head*100:5.1f}% | regr-loss {rl:.2e}\n")

    print(f"{'variant':12s} {'params':>9s} {'shared':>8s} {'mismatch':>9s} "
          f"{'gap':>7s} {'captured':>9s}")
    print("-" * 60)
    for name in [v.strip() for v in a.variants.split(",")]:
        m = train(clients, a, **cfg[name])
        shared = evaluate(m, clients, a.device)[0]
        mism = eval_mismatched(m, clients, a.device)
        npar = sum(q.numel() for q in m.parameters())
        cap = (ref - shared) / head * 100 if abs(head) > 1e-6 else float("nan")
        print(f"{name:12s} {npar:>9,} {shared:8.2f} {mism:9.2f} {mism-shared:+7.2f} "
              f"{cap:8.1f}%", flush=True)




def fit_local_thetas(model, clients, a, steps=120):
    """Per-client conditioning parameters, fitted with the shared denoiser frozen."""
    out = []
    for c in clients:
        local = copy.deepcopy(model).to(a.device)
        for p in local.parameters():
            p.requires_grad_(False)
        s_mean = c.S.mean(0).to(a.device)
        theta = {k: v.clone().requires_grad_(True)
                 for k, v in local.gen_cond(s_mean).items()}
        opt = torch.optim.Adam(list(theta.values()), lr=1e-2)
        Phi, sig = c.Phi.to(a.device), c.sigma2.to(a.device)
        n = len(c)
        for _ in range(steps):
            idx = torch.randperm(n)[:a.batch_size]
            h, y, _ = c.batch(idx)
            h, y = h.to(a.device), y.to(a.device)
            hh = local(y, Phi, sig, None, frames=c.frames, cond=theta)
            loss = nmse(hh.reshape(-1, c.L), h.reshape(-1, c.L))
            opt.zero_grad(); loss.backward(); opt.step()
        out.append((c.S.clone(), {k: v.detach() for k, v in theta.items()}))
    return out


def distil_hypernet(model, targets, a, steps=3000, lr=1e-3):
    """Regress the hypernetwork onto the locally-fitted per-client parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.hyper.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.hyper.parameters(), lr=lr)
    for i in range(steps):
        opt.zero_grad()
        loss = 0.0
        for s_all, tgt in targets:
            # Regress over the client's whole embedding distribution, not just
            # its mean: inference uses PER-SAMPLE s, so the mapping must be
            # correct across that spread, not only at one point.
            idx = torch.randperm(s_all.shape[0])[:16]
            for s_one in s_all[idx].to(a.device):
                pred = model.cond_from_hyper(s_one)
                for k in tgt:
                    loss = loss + torch.nn.functional.mse_loss(pred[k], tgt[k])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.hyper.parameters(), 5.0)
        opt.step()
    return model, float(loss)


if __name__ == "__main__":
    main()
