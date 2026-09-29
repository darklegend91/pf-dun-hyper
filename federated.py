"""
Federated training with mobility-weighted, staleness-robust aggregation.

Only the shared parameters Theta = {hypernetwork, denoiser, temporal core} are
communicated. Personalized estimator weights are generated locally by the
hypernetwork from each vehicle's mobility embedding and never leave the client.
"""
from __future__ import annotations
import copy
import torch
from models import nmse


def local_train(model, client, epochs, lr, batch_size, device):
    """One client's local update. Returns updated state_dict and mean loss."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Phi = client.Phi.to(device)
    sigma2 = client.sigma2.to(device)
    n = len(client)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            h, y, s = client.batch(idx)
            h, y, s = h.to(device), y.to(device), s.to(device)
            h_hat = model(y, Phi, sigma2, s, frames=client.frames)
            loss = nmse(h_hat.reshape(-1, client.L), h.reshape(-1, client.L))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            last = loss.item()
    return copy.deepcopy(model.state_dict()), last


def mobility_weight(client):
    """Coverage weight rho(s): up-weight high-Doppler (rarer/harder) clients."""
    doppler = client.S[:, 0].mean().item()
    return 1.0 + doppler          # simple, monotic in Doppler spread


def aggregate(global_state, client_states, weights):
    """Weighted average of client state dicts."""
    total = sum(weights)
    new_state = copy.deepcopy(global_state)
    for k in new_state:
        if not torch.is_floating_point(new_state[k]) and not torch.is_complex(new_state[k]):
            continue
        acc = torch.zeros_like(new_state[k], dtype=torch.float32)
        for st, w in zip(client_states, weights):
            acc += (w / total) * st[k].float()
        new_state[k] = acc.to(new_state[k].dtype)
    return new_state


@torch.no_grad()
def evaluate(model, clients, device):
    """Mean NMSE (dB) across clients on their local data."""
    model = model.to(device).eval()
    per_client = {}
    total = 0.0
    for c in clients:
        Phi = c.Phi.to(device)
        sigma2 = c.sigma2.to(device)
        h, y, s = c.H.to(device), c.Y.to(device), c.S.to(device)
        h_hat = model(y, Phi, sigma2, s, frames=c.frames)
        val = nmse(h_hat.reshape(-1, c.L), h.reshape(-1, c.L))
        db = 10 * torch.log10(val).item()
        per_client.setdefault(c.scenario, []).append(db)
        total += db
    model.train()
    avg = total / len(clients)
    by_sc = {k: sum(v) / len(v) for k, v in per_client.items()}
    return avg, by_sc


def federated_train_phn(global_model, clients, rounds, local_epochs, lr,
                        batch_size, device, seed=0, patience=0, min_delta=0.05,
                        eval_every=2, hyper_lr=1e-2, hyper_steps=40, verbose=False):
    """
    pFedHN-style federated training (Shamsian et al., 2021) for the
    mobility-conditioned hypernetwork.

    WHY THIS EXISTS
    ---------------
    Plain FedAvg averages every client's *copy* of the hypernetwork. Within a
    single client the mobility embedding s is essentially constant, so local
    training carries no signal about how theta should VARY with s, and the
    averaging step blends away whatever specialisation each client found. The
    result is a hypernetwork whose output barely depends on s.

    Instead, each round:
      1. the server generates theta_i = g_phi(s_i) for each client;
      2. the client treats theta_i as free local parameters and optimises them
         (together with the shared denoiser) on its own data  -> theta_i*;
      3. the shared denoiser is FedAvg-ed as usual, while the hypernetwork is
         trained by REGRESSION onto the targets {(s_i, theta_i*)}.

    Step 3 is what teaches the mapping: the hypernetwork now receives, per
    round, a supervised example of the right parameters for each distinct
    mobility state.
    """
    g = torch.Generator().manual_seed(seed)
    history = []
    best, stale, best_state = float("inf"), 0, None
    hyper_opt = torch.optim.Adam(global_model.hyper.parameters(), lr=hyper_lr)

    # parameters shared by every client (everything except the hypernetwork)
    shared_names = [n for n, _ in global_model.named_parameters()
                    if not n.startswith("hyper.")]

    for r in range(rounds):
        targets, shared_states, weights = [], [], []
        for c in clients:
            local = copy.deepcopy(global_model).to(device)
            s_mean = c.S.mean(0).to(device)
            theta = {k: v.clone().requires_grad_(True)
                     for k, v in local.gen_cond(s_mean).items()}
            shared_params = [p for n, p in local.named_parameters()
                             if not n.startswith("hyper.")]
            opt = torch.optim.Adam(list(theta.values()) + shared_params, lr=lr)

            Phi, sig = c.Phi.to(device), c.sigma2.to(device)
            n = len(c)
            for _ in range(local_epochs):
                perm = torch.randperm(n)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    h, y, _ = c.batch(idx)
                    h, y = h.to(device), y.to(device)
                    hh = local(y, Phi, sig, None, frames=c.frames, cond=theta)
                    loss = nmse(hh.reshape(-1, c.L), h.reshape(-1, c.L))
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(theta.values()) + shared_params, 5.0)
                    opt.step()

            targets.append((s_mean, {k: v.detach() for k, v in theta.items()}))
            shared_states.append({n: p.detach().clone()
                                  for n, p in local.named_parameters()
                                  if not n.startswith("hyper.")})
            weights.append(len(c))

        # --- FedAvg the shared (non-hypernetwork) parameters ---
        tot = sum(weights)
        gsd = global_model.state_dict()
        for nme in shared_names:
            acc = torch.zeros_like(gsd[nme], dtype=torch.float32)
            for st, w in zip(shared_states, weights):
                acc += (w / tot) * st[nme].float()
            gsd[nme] = acc.to(gsd[nme].dtype)
        global_model.load_state_dict(gsd)

        # --- train the hypernetwork to PREDICT the per-client optima ---
        for _ in range(hyper_steps):
            hyper_opt.zero_grad()
            loss = 0.0
            for s_mean, tgt in targets:
                pred = global_model.cond_from_hyper(s_mean)
                for k in tgt:
                    loss = loss + torch.nn.functional.mse_loss(pred[k], tgt[k])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(global_model.hyper.parameters(), 5.0)
            hyper_opt.step()

        if patience > 0 and (r % eval_every == 0 or r == rounds - 1):
            avg, _ = evaluate(global_model, clients, device)
            history.append((r, avg))
            if verbose:
                print(f"  [phn round {r:3d}] NMSE {avg:6.2f} dB")
            if avg < best - min_delta:
                best, stale = avg, 0
                best_state = copy.deepcopy(global_model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    if best_state is not None:
                        global_model.load_state_dict(best_state)
                    return global_model, history

    if patience > 0 and best_state is not None:
        global_model.load_state_dict(best_state)
    return global_model, history


def federated_train(global_model, clients, rounds, local_epochs, lr,
                    batch_size, client_frac, device, staleness_beta=1.0,
                    mobility_agg=True, seed=0, log_every=1,
                    patience=0, min_delta=0.05, eval_every=2):
    """
    Main FL loop. Returns trained global model and history.

    FAIR TRAINING BUDGET (patience > 0):
      Every method is trained until *it* stops improving, rather than for a
      fixed number of rounds. Without this, slow-converging baselines are
      unfairly penalised by a budget chosen to suit the fastest method --
      which inflates the reported gap. With patience>0 we stop a method when
      its NMSE has not improved by `min_delta` dB for `patience` evaluations,
      and report the round at which it stopped.
    """
    g = torch.Generator().manual_seed(seed)
    history = []
    n_clients = len(clients)
    n_select = max(1, int(client_frac * n_clients))
    best, stale, best_state = float("inf"), 0, None

    for r in range(rounds):
        sel = torch.randperm(n_clients, generator=g)[:n_select].tolist()
        client_states, weights = [], []
        for ci in sel:
            local = copy.deepcopy(global_model)
            state, loss = local_train(local, clients[ci], local_epochs, lr,
                                      batch_size, device)
            client_states.append(state)
            w = len(clients[ci])
            if mobility_agg:
                w *= mobility_weight(clients[ci])
            weights.append(w)

        new_state = aggregate(global_model.state_dict(), client_states, weights)
        global_model.load_state_dict(new_state)

        # --- early stopping on a plateau (fair, per-method training budget) ---
        if patience > 0 and (r % eval_every == 0 or r == rounds - 1):
            avg, _ = evaluate(global_model, clients, device)
            history.append((r, avg))
            if avg < best - min_delta:                 # meaningful improvement
                best, stale = avg, 0
                best_state = copy.deepcopy(global_model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    if best_state is not None:         # restore best checkpoint
                        global_model.load_state_dict(best_state)
                    print(f"[early stop @ round {r}] best NMSE {best:.2f} dB")
                    return global_model, history

        if patience == 0 and (r % log_every == 0 or r == rounds - 1):
            avg, by_sc = evaluate(global_model, clients, device)
            history.append((r, avg))
            sc_str = "  ".join(f"{k}:{v:5.1f}" for k, v in by_sc.items())
            print(f"[round {r:3d}] NMSE {avg:6.2f} dB | {sc_str}")

    if patience > 0 and best_state is not None:
        global_model.load_state_dict(best_state)
    return global_model, history
