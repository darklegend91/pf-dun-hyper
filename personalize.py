"""
=============================================================================
 personalize.py -- the three-stage training route that makes the mobility
                   conditioning actually personalise
=============================================================================

Why this exists
---------------
Plain FedAvg on a hypernetwork does NOT learn a mobility-dependent estimator:

  * Each client trains its own COPY of the hypernetwork, and within one client
    the embedding s is nearly constant -- so local training carries no signal
    about how theta should vary WITH s.
  * Averaging those copies then blends away whatever each client found.

Measured consequence (see results.md): swapping one vehicle's embedding for
another's changed NMSE by 0.04 dB, i.e. no personalisation at all.

Two things were needed:

  1. EXPRESSIVENESS. FiLM only rescales activations. Fitting theta per client
     directly (a cheat, an upper bound) recovered just ~19-30% of the oracle
     headroom. Adding mobility-conditioned LOW-RANK WEIGHT ADAPTATION
     (W2 + B(s)A(s), see models.FiLMDenoiser) raised that ceiling to ~84-102%.

  2. LEARNING. Rather than averaging hypernetwork copies, train it explicitly
     to PREDICT each vehicle's locally-optimised parameters:

        stage 1  FedAvg the shared denoiser W               (federated)
        stage 2  freeze W, fit theta_i* on each client      (local)
        stage 3  distil g_phi onto {(s_i, theta_i*)}        (server)

     Stage 3 regresses over each client's WHOLE embedding distribution, not
     just its mean: inference uses per-sample s (velocity is drawn per sample),
     so the mapping must be right across that spread.

Result: mismatch gap 0.12 dB -> 3.28 dB, NMSE -16.7 -> -19.9 dB.
"""
from __future__ import annotations
import copy
import torch

from models import nmse
from federated import federated_train


# --------------------------------------------------------------------------- #
def _hyper_batch(model, s_batch):
    """Hypernetwork outputs for a BATCH of embeddings, keyed like a theta dict."""
    hp = model.hyper(s_batch)
    out = {k: hp[k] for k in ("gamma", "lam", "scale", "shift")}
    if model.strong_cond:
        out["scale2"], out["shift2"] = hp["scale2"], hp["shift2"]
        out["lora_A"], out["lora_B"] = hp["lora"][0], hp["lora"][1]
    return out


def fit_local_thetas(model, clients, device, steps=120, lr=1e-2, batch_size=64):
    """
    Stage 2. With the shared denoiser FROZEN, optimise each client's own
    conditioning parameters theta_i on its own data.
    Returns [(s_all_i, theta_i*)] -- the regression targets for stage 3.
    """
    targets = []
    for c in clients:
        local = copy.deepcopy(model).to(device)
        for p in local.parameters():
            p.requires_grad_(False)
        s_mean = c.S.mean(0).to(device)
        theta = {k: v.clone().requires_grad_(True)
                 for k, v in local.gen_cond(s_mean).items()}
        opt = torch.optim.Adam(list(theta.values()), lr=lr)
        Phi, sig = c.Phi.to(device), c.sigma2.to(device)
        n = len(c)
        for _ in range(steps):
            idx = torch.randperm(n)[:batch_size]
            h, y, _ = c.batch(idx)
            h, y = h.to(device), y.to(device)
            hh = local(y, Phi, sig, None, frames=c.frames, cond=theta)
            loss = nmse(hh.reshape(-1, c.L), h.reshape(-1, c.L))
            opt.zero_grad(); loss.backward(); opt.step()
        targets.append((c.S.clone().to(device),
                        {k: v.detach() for k, v in theta.items()}))
    return targets


def distil_hypernet(model, targets, device, steps=900, lr=1e-3, s_batch=32):
    """
    Stage 3. Regress the hypernetwork onto the per-client targets.

    Vectorised over the embedding batch: the hypernetwork is evaluated once on
    a (s_batch, d) tensor per client per step rather than once per embedding.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.hyper.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.hyper.parameters(), lr=lr)
    last = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for s_all, tgt in targets:
            idx = torch.randperm(s_all.shape[0], device=s_all.device)[:s_batch]
            pred = _hyper_batch(model, s_all[idx])
            for k, t in tgt.items():
                loss = loss + torch.nn.functional.mse_loss(
                    pred[k], t.unsqueeze(0).expand_as(pred[k]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.hyper.parameters(), 5.0)
        opt.step()
        last = float(loss)
    for p in model.parameters():
        p.requires_grad_(True)
    return model, last


def train_personalized(model, clients, rounds, local_epochs, lr, batch_size,
                       device, seed=0, patience=4, eval_every=2,
                       theta_steps=120, distil_steps=900):
    """Full three-stage route. Drop-in replacement for federated_train()."""
    model, hist = federated_train(model, clients, rounds, local_epochs, lr,
                                  batch_size, 1.0, device, seed=seed,
                                  patience=patience, eval_every=eval_every)
    targets = fit_local_thetas(model, clients, device, steps=theta_steps,
                               batch_size=batch_size)
    model, _ = distil_hypernet(model, targets, device, steps=distil_steps)
    return model, hist


@torch.no_grad()
def mismatch_gap(model, clients, device):
    """Diagnostic: NMSE penalty when each client is fed another client's s."""
    def run(cross):
        tot, n = 0.0, len(clients)
        for i, c in enumerate(clients):
            Phi, sig = c.Phi.to(device), c.sigma2.to(device)
            h, y = c.H.to(device), c.Y.to(device)
            if cross:
                o = clients[(i + 1) % n].S.to(device)
                reps = (c.S.shape[0] + o.shape[0] - 1) // o.shape[0]
                s = o.repeat(reps, 1)[: c.S.shape[0]]
            else:
                s = c.S.to(device)
            hh = model(y, Phi, sig, s, frames=c.frames)
            tot += 10 * torch.log10(nmse(hh.reshape(-1, c.L),
                                         h.reshape(-1, c.L))).item()
        return tot / n
    a, b = run(False), run(True)
    return a, b, b - a
