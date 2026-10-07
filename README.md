# PF-DUN-Hyper — System Model, Architecture and Full Process

**Personalized Federated Deep-Unfolding with a Doppler-aware Hypernetwork for
vehicular OTFS channel estimation.**

This README describes the complete system as it is implemented in this repository:
the physical system model, the estimator architecture, the federated training
process, the experiment pipeline, the recorded results, and where the current
evidence does and does not support each claim. Everything here is taken from the
code and the recorded logs and CSVs. Where the code and the paper figures disagree,
that is stated.

---

## Contents

1. [Problem and context](#1-problem-and-context)
2. [System model](#2-system-model)
3. [Estimator architecture](#3-estimator-architecture)
4. [Federated training process](#4-federated-training-process)
5. [Input → processing → output](#5-input--processing--output)
6. [Baselines](#6-baselines)
7. [Metrics](#7-metrics)
8. [Experiment pipeline and parametric studies](#8-experiment-pipeline-and-parametric-studies)
9. [Recorded results](#9-recorded-results)
10. [Claim–evidence alignment and known caveats](#10-claimevidence-alignment-and-known-caveats)
11. [Code map](#11-code-map)
12. [Reproduction](#12-reproduction)

---

## 1. Problem and context

Vehicles must estimate their wireless channel continuously in order to decode data.
In the vehicular setting, three difficulties occur together:

| Difficulty | What it means | How this system responds |
|---|---|---|
| **High mobility** | Large Doppler, so the channel changes within a frame | OTFS delay–Doppler (DD) representation, where the channel is sparse and quasi-static |
| **Non-IID clients** | A tunnel channel differs statistically from a highway channel | Hypernetwork that generates estimator parameters from a mobility embedding |
| **Privacy and bandwidth** | Raw CSI leaks location and is expensive to upload | Federated learning; low-rank compression of the transmitted model |

**Waveform.** OTFS places symbols on an `N × M` delay–Doppler grid. A doubly-selective
multipath channel becomes a small number of non-zero DD taps, so estimation becomes a
**sparse linear inverse problem**.

**Estimator family.** The backbone is **deep unfolding**: `T` iterations of an
OAMP-style algorithm are unrolled into `T` network layers. On top of this, a
**hypernetwork** maps each vehicle's mobility state to the per-layer parameters.

**Where it sits in the literature.** See [RELATED_WORK.md](RELATED_WORK.md), which covers
deep-unfolding OTFS estimation, hypernetwork personalized FL (pFedHN, HyperFedNet) and
vehicular FL.

---

## 2. System model

### 2.1 Network model

```
                        ┌──────────────────────────────┐
                        │            SERVER            │
                        │ shared denoiser W,  g_φ      │
                        └──────┬────────────────▲──────┘
                 broadcast     │                │   upload
                 model         │                │   local update
          ┌────────────┬───────┴────┬───────────┴┬────────────┐
          ▼            ▼            ▼            ▼            ▼
      Vehicle 1    Vehicle 2    Vehicle 3    Vehicle 4   …  Vehicle K
      urban_slow   highway      rural        tunnel       (round-robin
      10–40 km/h   90–140 km/h  50–90 km/h   30–70 km/h    over regimes)
      {y, s, h}    {y, s, h}    {y, s, h}    {y, s, h}
      local only   local only   local only   local only
```

- `K` vehicles (clients). Client `i` is assigned scenario `i mod 4`
  (`make_federation`, [otfs_data.py:221](otfs_data.py:221)).
- Each client holds `n_i` training samples `(h, y, s)`. They are generated locally and
  used only for local training.
- All clients share one protocol-level pilot and one OTFS operator `Φ`.

### 2.2 OTFS frame and pilot

| Symbol | Meaning | Default |
|---|---|---|
| `N` | Doppler bins | 16 (paper figures), 8 (`train.py`) |
| `M` | delay bins | 16 (paper figures), 8 (`train.py`) |
| `L = N·M` | DD grid size / channel length | 256 |
| `x ∈ ℂ^{N×M}` | DD pilot, QPSK, fixed seed 12345 | `x[k,l] = e^{j(π/4 + q·π/2)} / √(NM)`, `q ∈ {0,1,2,3}` |

The pilot is normalized so every column of `Φ` has unit norm
([otfs_data.py:48](otfs_data.py:48)).

### 2.3 OTFS input–output relation (true twisted-convolution operator)

The received DD samples are the 2-D circular convolution of the pilot with the DD
channel:

```
y[k,l] = Σ_{k',l'} x[(k−k') mod N, (l−l') mod M] · h[k',l']  +  n[k,l]
```

In vector form, `y = Φ h + n`, where column `(k',l')` of `Φ ∈ ℂ^{L×L}` is the pilot
circularly shifted by `(k',l')`:

```
Φ[:, k'·M + l'] = vec( roll(x, (k', l')) )
```

([otfs_data.py:57](otfs_data.py:57)). This is the structured OTFS operator, not a
random sensing matrix.

### 2.4 Pilot overhead and data-side row compression

- **Reduced pilot overhead.** Only `Q = ⌊ρ·L⌋` of the `L` DD bins are observed. The
  index set `Ω` is a fixed random subset (seed 12346), so `Φ_Ω = Φ[Ω, :] ∈ ℂ^{Q×L}`.
  The default is `ρ = Q/L = 0.6`, which gives `Q = 153` for a 16×16 grid.
- **Optional row compression** (`--row_compress c < 1`). A fixed matrix
  `S ∈ ℂ^{Q_c×Q}` with orthonormal rows (SVD of a complex Gaussian, seed 12347) gives
  `Φ_c = S Φ_Ω` with `Q_c = round(c·Q)`. It has no trainable parameters, and it shrinks
  the `Q × Q` eigendecomposition used inside OAMP
  ([otfs_data.py:121](otfs_data.py:121)).

### 2.5 Channel model — sparse DD channel with physical heterogeneity

For each sample, `sample_dd_channel` ([otfs_data.py:77](otfs_data.py:77)) draws:

| Quantity | Model |
|---|---|
| Velocity | `v ~ U(v_min, v_max)` (per scenario) |
| Number of paths | `P ~ U{P_min … P_max}` |
| Max Doppler bin | `k_max = (v / 140 km/h) · (N/2 − 1)` (normalized mapping, `V_REF = 140`) |
| Delays | `ℓ_p = min(⌊Exp(d)⌋, M−1)`, with `d = max(1, ds_frac·M)` |
| Powers | `p_p ∝ exp(−ℓ_p / d)`, normalized so `Σ p_p = 1` (exponential PDP) |
| Doppler per path | `κ_p = round(k_max · cos θ_p) mod N`, `θ_p ~ U(0, 2π)` (Clarke/Jakes) |
| Gains | NLoS: `g_p ~ CN(0, p_p)`. If `K > 0`, tap 0 is Rician: `√(K/(K+1)·p_0) + CN(0, p_0/(K+1))` |
| Channel | `h[κ_p·M + ℓ_p] += g_p` (on-grid, integer delay and Doppler) |

**Scenarios (non-IID by physics)** — [otfs_data.py:39](otfs_data.py:39):

| Scenario | Velocity (km/h) | Paths | RMS delay (`ds_frac·M`) | Rician K |
|---|---|---|---|---|
| `urban_slow` | 10–40 | 6–10 | 0.35·M | 0 (Rayleigh) |
| `highway` | 90–140 | 3–5 | 0.20·M | 6 |
| `rural` | 50–90 | 2–4 | 0.15·M | 8 |
| `tunnel` | 30–70 | 8–14 | 0.45·M | 0 (Rayleigh) |

`make_velocity_federation` ([otfs_data.py:192](otfs_data.py:192)) builds one client per
*exact* velocity, with all other parameters held at the `highway` scenario. The velocity
study (Fig. 3) uses it.

### 2.6 Temporal evolution (multi-frame mode, `frames > 1`)

Across frames, each channel follows an AR(1) model on its fixed support:

```
h_f = 0.98 · h_{f−1} + 0.15 · w_f ⊙ 1{|h_{f−1}| > 0},     w_f[l] = a + jb,  a,b ~ N(0,1)
```

([otfs_data.py:162](otfs_data.py:162)).

### 2.7 Noise and SNR

For each realization and frame:

```
σ² = mean(|Φ h|²) / 10^{SNR/10},      n ~ CN(0, σ² I_Q)
```

The client stores one `σ²` value, which is that of its **last** generated realization.
The estimator uses this value for all of the client's samples
([otfs_data.py:183](otfs_data.py:183)).

### 2.8 Mobility embedding `s ∈ ℝ⁵`

```
s = [ k_max / N,          # normalized Doppler spread
      max_p ℓ_p / M,      # normalized delay spread
      SNR_dB / 30,        # SNR (populated in ClientDataset, otfs_data.py:182)
      v / 140,            # normalized velocity
      K / 10 ]            # Rician factor
```

Each sample has its own embedding. Velocity and delay spread vary per sample; K is
constant within a scenario; SNR is constant across *all* clients in a run.

### 2.9 Problem formulation

Given `(y, Φ, σ², s)` at a vehicle, estimate the sparse DD channel `h`:

```
ĥ = f_Θ( y, Φ, σ² ; s )          minimize   E[ ‖ĥ − h‖² / ‖h‖² ]   (NMSE)
```

The minimization is subject to the federated constraint: raw `(h, y)` never leaves the
vehicle, and only model parameters are exchanged.

---

## 3. Estimator architecture

### 3.1 End-to-end block diagram

```
   mobility embedding s ∈ ℝ⁵ ─────────────────────────────┐
                                                           ▼
                                   ┌─────────────────────────────────────────┐
                                   │ HYPERNETWORK g_φ(s)                     │
                                   │ trunk: 5 → 128 → 128 (GELU)             │
                                   │ heads (optionally rank-r compressed):   │
                                   │  γ_t = 2·sigmoid(·)   step size   (T)   │
                                   │  λ_t = softplus(·)    threshold   (T)   │
                                   │  α_t, β_t   FiLM scale/shift (T×256)    │
                                   │  [strong] α'_t, β'_t, LoRA A(s), B(s)   │
                                   └────────────────────┬────────────────────┘
                                                        │ θ(s) conditions every layer
   y ∈ ℂ^Q ──┐                                          ▼
   Φ ∈ ℂ^{Q×L}├──► ┌──────────── UNROLLED OAMP, t = 1 … T ─────────────────┐
   σ²        ─┘    │ ĥ⁽⁰⁾ = 0  (+ GRU temporal bias for frame f > 0)       │
                   │                                                       │
                   │ ┌── linear step (LMMSE-type, cached eigh) ──────────┐ │
                   │ │ r_t = ĥ + γ_t·v_t·Φᴴ(v_t ΦΦᴴ + σ²I)⁻¹(y − Φĥ)     │ │
                   │ └──────────────────────────┬────────────────────────┘ │
                   │                            ▼                          │
                   │ ┌── conditioned denoiser (weights shared over t) ───┐ │
                   │ │ soft-threshold(r_t; λ_t·τ_t) → MLP 2L→256→256→2L  │ │
                   │ │ FiLM(α_t, β_t) [+ LoRA B(s)A(s)] [+ FiLM₂]        │ │
                   │ │ ĥ⁽ᵗ⁾ = x + g · MLP(x)      (learnable gate g)     │ │
                   │ └──────────────────────────┬────────────────────────┘ │
                   │               ĥ⁽ᵗ⁾ → next layer                        │
                   └───────────────────────────┬───────────────────────────┘
                                               ▼
                                  ĥ ∈ ℂ^L   estimated DD channel
```

### 3.2 Hypernetwork `g_φ` — `DopplerHyperNet` ([models.py:113](models.py:113))

| Output (per sample) | Shape | Activation | Role |
|---|---|---|---|
| `γ_t` | `T` | `2·sigmoid` → (0, 2) | OAMP step size per layer |
| `λ_t` | `T` | `softplus` → > 0 | shrinkage threshold multiplier |
| `α_t` (scale), `β_t` (shift) | `T × 256` | linear | FiLM after denoiser layer 1 |
| *strong only:* `α'_t`, `β'_t` | `T × 256` | linear | FiLM after denoiser layer 2 |
| *strong only:* `A(s)`, `B(s)` | `r × 256`, `256 × r` (`r = cond_rank = 4`) | linear, head init ≈ 0 | low-rank weight adaptation of denoiser layer 2 (shared across `t`) |

**Model-side row compression** (`compress=True`, the "-RC" variants). Each generating
head `Linear(128 → D)` is replaced by `Linear(128 → r, no bias) → Linear(r → D)` with
`r = compress_rank` (default 16). The FiLM heads dominate the parameter count, so this
roughly halves the model.

### 3.3 Unrolled OAMP layer — linear step ([models.py:238](models.py:238))

```
v_t   = max(mean|ĥ⁽ᵗ⁻¹⁾|², 1e−4) + 1e−3          (scalar per batch, not differentiated)
A_t   = v_t · ΦΦᴴ + σ² I_Q
r_t   = ĥ⁽ᵗ⁻¹⁾ + γ_t · v_t · Φᴴ A_t⁻¹ (y − Φ ĥ⁽ᵗ⁻¹⁾)
τ_t   = mean_l |r_t − ĥ⁽ᵗ⁻¹⁾| + 1e−4              (per sample)
```

**Runtime optimization.** `A_t` shares its eigenvectors with `ΦΦᴴ = U diag(e) Uᴴ`. One
`eigh` per forward pass therefore serves all `T` layers:
`A_t⁻¹ = U diag(1/(v_t e + σ²)) Uᴴ`. This replaces a `Q × Q` solve in every layer.

### 3.4 Conditioned denoiser — `FiLMDenoiser` ([models.py:58](models.py:58))

One denoiser module is **shared by all `T` layers**. Per-layer behaviour comes only from
`γ_t`, `λ_t` and the FiLM/LoRA conditioning.

```
η      = r ⊙ max(|r| − λ_t τ_t, 0) / |r|           complex soft threshold (sparsity prior)
x      = [Re η ; Im η] ∈ ℝ^{2L}
u₁     = GELU(W₁ x + b₁) ⊙ (1 + α_t) + β_t          FiLM #1
u₂     = W₂ u₁ + b₂  [+ B(s) A(s) u₁]               LoRA-style adaptation (strong)
u₂     = GELU(u₂)  [⊙ (1 + α'_t) + β'_t]            FiLM #2 (strong)
ĥ⁽ᵗ⁾   = complex( x + g · (W₃ u₂ + b₃) )            residual, learnable gate g
```

The gate `g` is learnable, initialized to 0.1 (weak) or 0.4 (strong). In the earlier
runs recorded under `results/`, it was a fixed 0.1.

### 3.5 Temporal core (optional) — `TemporalCore` ([models.py:184](models.py:184))

For frames `f > 0`:

```
z_f   = GRUCell( Linear_{2L→128}([Re;Im] ĥ_{f−1}), z_{f−1} )
ĥ_f⁽⁰⁾ = complex( Linear_{128→2L}(z_f) )           replaces the zero initialization
```

`ĥ_{f−1}` is detached, so no gradient flows back through earlier frames.

### 3.6 Model configurations and size (16×16 grid, `T = 8`, float32)

| Configuration | Constructor | Parameters | Payload / round |
|---|---|---:|---:|
| OAMP-DUN (no hypernetwork) | `PFDUNHyper(use_hyper=False)` | 328,721 | 1.31 MB |
| **PF-DUN-Hyper** (weak, FiLM only) | `PFDUNHyper(use_hyper=True)` | 876,433 | 3.51 MB |
| **PF-DUN-Hyper-RC** (rank 16) | `… compress=True, compress_rank=16` | 421,777 | 1.69 MB |
| PF-DUN-Hyper + temporal | `… use_temporal=True` | 1,107,217 | 4.43 MB |
| PF-DUN-Hyper strong (FiLM×2 + LoRA r=4) | `… strong_cond=True` | 1,669,009 | 6.68 MB |
| PF-DUN-Hyper strong + RC16 | `… strong_cond=True, compress=True` | 532,369 | 2.13 MB |

Breakdown of the weak model: denoiser 328,705 + hypernetwork 547,728. The CSVs under
`results/` and `paper_figures/` show one parameter fewer (for example 876,432), because
they were recorded before the residual gate became learnable.

### 3.7 Tensor shape trace (one forward pass)

| Stage | Tensor | Shape |
|---|---|---|
| input | `y_seq` | `(B, F, Q)` complex |
| input | `Φ` | `(Q, L)` complex |
| input | `s` | `(B, 5)` float |
| precompute | `U, e = eigh(ΦΦᴴ)` | `(Q, Q)`, `(Q,)` |
| hypernetwork | `γ, λ` / `α, β` | `(B, T)` / `(B, T, 256)` |
| per layer | `r_t`, `ĥ⁽ᵗ⁾` | `(B, L)` complex |
| denoiser internal | `x`, `u₁`, `u₂` | `(B, 2L)`, `(B, 256)`, `(B, 256)` |
| output | `ĥ_seq` | `(B, F, L)` complex |

---

## 4. Federated training process

The repository contains **three trainers**. Section 10 explains why it matters which
trainer produced which result.

### 4.1 Trainer A — FedAvg with mobility weighting — `federated_train` ([federated.py:188](federated.py:188))

This trainer produced **all eight paper figures (Figs. 1–8)** and `train.py` /
`run_experiments.py`.

```
for round r = 1 … R:
    S_r ← random ⌊client_frac·K⌋ clients
    for each client i ∈ S_r:                                   (on-vehicle)
        θ_i ← copy of global model (denoiser + hypernetwork [+ GRU])
        Adam(lr), local_epochs over minibatches:
            loss = NMSE( f_θi(y, Φ, σ², s), h )    over all frames
            clip grad-norm 5.0
        upload full state_dict θ_i
    w_i = n_i · (1 + mean_samples s[0])      if mobility_agg   (weight ∈ [1, ~1.44])
    θ ← Σ w_i θ_i / Σ w_i                    (every float tensor in the state_dict)
    if patience > 0: evaluate every `eval_every` rounds; stop after `patience`
                     evaluations without ≥ 0.05 dB gain; restore best checkpoint
```

The early-stopping rule gives a **fair per-method budget**: each method trains until it
plateaus, so no baseline is cut off by a round count chosen to suit another method.

### 4.2 Trainer B — pFedHN-style — `federated_train_phn` ([federated.py:79](federated.py:79))

```
for round r:
    for every client i:
        θ_i ← g_φ(s̄_i)  (s̄_i = mean embedding), treated as free local parameters
        optimize θ_i and the shared (non-hypernetwork) params on local data
        upload shared params, θ_i*, s̄_i
    FedAvg the shared params (weights n_i)
    server: train g_φ for 40 steps by regression   Σ_i ‖g_φ(s̄_i) − θ_i*‖²
```

### 4.3 Trainer C — three-stage personalization — `train_personalized` ([personalize.py:118](personalize.py:118))

This is the route shown in the architecture figure `paper_figures/fig0_architecture`.

```
Stage 1 (federated)  Trainer A → shared denoiser W and an initial g_φ
Stage 2 (local)      freeze everything; per client, optimize θ_i (init g_φ(s̄_i))
                     with Adam lr 1e−2 for 120 steps on local data  → θ_i*
Stage 3 (server)     freeze all except g_φ; regress g_φ(s) onto θ_i* over each
                     client's per-sample embeddings {s}, 900 steps, Adam lr 1e−3
```

Stage 3 needs both `θ_i*` and each client's embedding set `{s}` at the server.

### 4.4 What crosses the network

| Trainer | Downlink | Uplink per client |
|---|---|---|
| A | full model | full model (all parameters, for example 3.51 MB weak / 1.69 MB RC) |
| B | full model | shared parameters + `θ_i*` + `s̄_i` |
| C | full model (stage 1) | stage 1 as A; then `θ_i*` + per-sample `{s}` once |

In all three trainers, raw `h` and `y` never leave the vehicle.

---

## 5. Input → processing → output

| | Content |
|---|---|
| **Input** | Received pilot observations `y ∈ ℂ^Q` (per frame) · shared OTFS operator `Φ ∈ ℂ^{Q×L}` · noise variance `σ²` · mobility embedding `s ∈ ℝ⁵` |
| **Processing** | (1) Hypernetwork `g_φ(s)` → per-layer `γ_t, λ_t`, FiLM (+ LoRA). (2) One `eigh(ΦΦᴴ)`. (3) `T` unrolled OAMP layers, each a linear LMMSE-type correction followed by a conditioned sparse denoiser. (4) Optional GRU bias across frames. (5) Training via federated rounds (§4). |
| **Output** | Estimated DD channel `ĥ ∈ ℂ^L` per frame. Reported as NMSE (dB); federated payload (params/bytes per round); latency per estimate; and, after DD-domain LMMSE equalization, uncoded BER |

---

## 6. Baselines

| Method | Type | Implementation |
|---|---|---|
| **LS** | classical | `ĥ = Φ⁺ y` (pseudo-inverse; minimum-norm because `Q < L`) |
| **LMMSE** | classical | `ĥ = Φᴴ(ΦΦᴴ + σ²I)⁻¹ y` (white prior, `ch_var = 1`) |
| **CNN + FedAvg** | black-box DL (ChannelNet-style) | LS estimate as a `2×N×M` image → 4 conv layers (64 ch) → residual on LS |
| **LAMP + FedAvg** | learned unfolding (Borgerding & Schniter 2017) | learned `B ∈ ℂ^{L×Q}` tied over layers, `γ_t = sigmoid`, learned thresholds |
| **OAMP-DUN + FedAvg** | unfolding, no hypernetwork | same backbone as proposed; shared `γ_t, λ_t`; FiLM fixed to zero |

All neural baselines are trained with Trainer A, using the same fair budget.

---

## 7. Metrics

- **NMSE (dB)** = `10 log₁₀( mean_b ‖ĥ_b − h_b‖² / ‖h_b‖² )`, averaged over clients
  ([models.py:422](models.py:422), [federated.py:59](federated.py:59)).
- **BER (Fig. 8).** QPSK data on all `L` DD bins goes through the true channel matrix
  (the twisted-convolution operator built from `h`) plus noise. Each estimator's `ĥ`
  builds `Ĥ`, followed by DD-domain LMMSE equalization `x̂ = Ĥᴴ(ĤĤᴴ + σ²I)⁻¹ y` and
  hard decisions. The BER is uncoded, from 10 samples per client.
- **Federated payload** = parameter count × 4 bytes per round.
- **Latency** = milliseconds per channel estimate (CPU).

---

## 8. Experiment pipeline and parametric studies

### 8.1 Scripts → outputs

| Script | Produces |
|---|---|
| `make_paper_figures.py` | `paper_figures/fig1…fig8.{png,pdf,csv}`, the eight parametric studies |
| `make_architecture.py` | `paper_figures/fig0_architecture.{png,pdf}`, the system-model diagram |
| `replot_figures.py` | restyles paper figures from their CSVs, without retraining |
| `run_analysis.py` | `results/{ablation,embedding_control,robustness,complexity,oracle_headroom}.{csv,png}` |
| `plot_analysis.py` | re-plots the `results/` figures from CSVs |
| `test_personalization.py` | oracle ceiling, FiLM vs FiLM+LoRA ceilings, distillation route, mismatch gap |
| `diag_conditioning.py` | conditioning-collapse diagnostics |
| `run_experiments.py` | multi-seed SNR sweep → `results/results_table.md`, `nmse_vs_snr.png`, `convergence.png`, `comm_cost.png` |
| `train.py` | a single configuration with ablation flags |

### 8.2 Parametric studies (inputs varied → outputs predicted)

Common setup for `make_paper_figures.py`: 16×16 grid (`L = 256`); `Q/L = 0.6` (`Q = 153`);
`K = 4` clients; SNR 10 dB; `T = 8`; `max_rounds = 20`; `patience = 3`;
`eval_every = 2`; `local_epochs = 1`; batch 64; `lr = 1e−3`. The recorded run used
**160 samples per client** (the script default is 128).

| Fig. | Input varied | Values | Output | Methods |
|---|---|---|---|---|
| 1 | SNR | 0, 5, 10, 15, 20 dB (2 seeds) | NMSE ± std | all 7 |
| 2 | Pilot overhead `Q/L` | 0.30, 0.45, 0.60, 0.75, 0.90 | NMSE | all 7 |
| 3 | Vehicle velocity | train {20, 50, 80, 110, 140}; test {10, 30, …, 150} km/h | NMSE | all 7 |
| 4 | FL round | from Fig. 1, seed 0, 10 dB | NMSE vs round | 5 neural |
| 5 | Unfolding depth `T` | 2, 4, 6, 8, 12 | NMSE | OAMP-DUN, PF-DUN-Hyper, -RC |
| 6 | Compression rank `r` | full, 2, 4, 8, 16, 32 | NMSE vs payload | PF-DUN-Hyper(-RC) |
| 7 | Number of vehicles `K` | 2, 4, 8, 12 (total data fixed) | NMSE | CNN, OAMP-DUN, PF-DUN-Hyper, -RC |
| 8 | SNR | 0, 5, 10, 15 dB | BER | all 7 |

Other input knobs available through the CLI: `--N/--M`, `--samples`, `--client_frac`,
`--local_epochs`, `--frames`, `--row_compress`, `--lr`, `--batch_size`, `--seed(s)`.

---

## 9. Recorded results

### 9.1 NMSE vs SNR — Fig. 1 (dB, mean over 2 seeds, 16×16, K = 4)

| Method | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---:|---:|---:|---:|---:|
| LS | 1.67 | −1.32 | −2.96 | −3.63 | −3.87 |
| LMMSE | 1.61 | −1.33 | −2.96 | −3.63 | −3.87 |
| CNN + FedAvg | −1.83 | −3.29 | −3.52 | −4.00 | −4.95 |
| LAMP + FedAvg | −5.40 | −6.76 | −7.31 | −7.66 | −7.92 |
| OAMP-DUN + FedAvg | −0.27 | −2.88 | −5.48 | −7.04 | −7.71 |
| **PF-DUN-Hyper** | **−8.25** | **−12.25** | **−16.39** | **−21.04** | **−25.75** |
| **PF-DUN-Hyper-RC** | **−8.26** | **−12.48** | **−16.79** | **−21.23** | **−25.70** |

### 9.2 Other parametric studies (paper figures)

| Fig. | Headline |
|---|---|
| 2 — pilot overhead | PF-DUN-Hyper at `Q/L = 0.3` gives −13.21 dB. The best baseline at `Q/L = 0.9` (LAMP) gives −10.35 dB |
| 3 — velocity | PF-DUN-Hyper stays between −16.8 and −18.1 dB over 10–150 km/h (RC: −17.4 to −18.9). LAMP falls from −9.9 to −3.0 dB |
| 4 — convergence | PF-DUN-Hyper reaches about −15.5 dB within 4 rounds |
| 5 — depth | Saturates at `T ≈ 4–6` (−16.7 dB) |
| 6 — rank | Rank 16: 421,776 params, −16.75 dB, versus full 876,432 params, −16.54 dB (52 % smaller). Rank 2: 360,848 params, −16.45 dB |
| 7 — vehicles | PF-DUN-Hyper goes from −18.75 dB (K=2) to −14.96 dB (K=12); the CNN goes from −7.57 to −2.62 dB |
| 8 — BER | At 15 dB: 1.56·10⁻³ (proposed), 4.73·10⁻² (LAMP), 2.51·10⁻¹ (LMMSE) |

### 9.3 Component ablation and controls (`results/`, 16×16, K = 4, 10 dB, single seed)

| Variant | Params | NMSE (dB) |
|---|---:|---:|
| OAMP unfolding only | 328,720 | −4.75 |
| + hypernetwork (FiLM denoiser) | 876,432 | −15.93 |
| + hypernetwork, plain FedAvg weights | 876,432 | −15.90 |
| + hypernetwork + temporal core | 1,107,216 | −15.92 |
| + hypernetwork + row compression | 421,776 | −16.11 |

| Embedding control | NMSE (dB) |
|---|---:|
| correct `s` | −15.75 |
| another client's `s` | −15.71 |
| shuffled `s` | −15.73 |
| zeroed `s` | −15.21 |
| random `s` | −15.28 |

| Oracle (one model per scenario) | Shared | Oracle | Headroom |
|---|---:|---:|---:|
| urban_slow | −16.08 | −17.75 | 1.68 dB |
| highway | −18.23 | −20.59 | 2.36 dB |
| rural | −19.25 | −20.82 | 1.57 dB |
| tunnel | −14.51 | −17.24 | 2.73 dB |

Adding 50 % relative noise to `s` changes NMSE by 0.06 dB.

### 9.4 Making the personalization work (`test_personalization.py`, log in `distil.log`)

| Setting | NMSE (dB) | Mismatch gap |
|---|---:|---:|
| Per-scenario oracle (mean) | −19.11 | — |
| Trainer A, weak conditioning (reference) | −16.74 | +0.12 dB |
| Ceiling: FiLM-only, θ fitted per client | −17.34 (25 % of headroom) | — |
| Ceiling: FiLM + LoRA, θ fitted per client | −18.92 (86 % of headroom) | — |
| **Trainer C (three-stage), strong conditioning** | **−19.95** | **+3.28 dB** |

With the three-stage route and LoRA conditioning, giving a vehicle another vehicle's
embedding costs 3.28 dB, compared with 0.04–0.12 dB under Trainer A. In this setting,
the mobility conditioning does change the estimate.

---

## 10. Claim–evidence alignment and known caveats

These points need to be resolved or disclosed before submission.

1. **The architecture figure and the result figures describe different models.**
   `fig0_architecture` shows strong conditioning (FiLM + LoRA) trained with the
   three-stage route (Trainer C). Figs. 1–8 were produced by `make_paper_figures.py`,
   which builds the **weak** FiLM-only model (`PFDUNHyper(use_hyper=True)`) and trains it
   with **Trainer A**. One of the two has to change:
   - regenerate Figs. 1–8 with `strong_cond=True` + `personalize.train_personalized`, or
   - redraw Fig. 0 as FiLM-only + FedAvg, and drop the personalization framing.
2. **Personalization is supported only under Trainer C.** Under Trainer A (the model
   behind Figs. 1–8), swapping embeddings costs 0.04 dB. The Figs. 1–8 gain therefore
   comes from the unfolding + FiLM residual denoiser, not from Doppler personalization.
   Fig. 3 should be described as robustness across velocity, not as adaptation.
3. **Mobility-weighted aggregation (0.03 dB) and the temporal core (0.00 dB)** have no
   measured effect. `staleness_beta` is accepted by `federated_train` but never used, so
   "staleness-robust aggregation" is not implemented.
4. **Evaluation uses training data.** `evaluate()` scores each client on the same
   samples it trained on, and early stopping selects on that score. Fig. 3 is the
   exception: it uses a separate test federation (seed 7) that includes unseen
   velocities. A held-out test split is needed for every other figure. It also needs to
   show that the Trainer C result (−19.95 dB, better than the oracle) is not overfitting.
5. **Privacy statement in Fig. 0.** The figure says each vehicle keeps `s_i`. Trainers B
   and C send `s̄_i` or the per-sample `{s}` (which includes velocity) and `θ_i*` to the
   server. Raw CSI stays local; mobility metadata does not.
6. **Embedding redundancy.** `s[0]` (Doppler) and `s[3]` (velocity) are exactly
   proportional. `s[2]` (SNR) is identical for every client in a run. Only velocity,
   delay spread and K distinguish vehicles.
7. **Physical abstraction.** Doppler is a normalized linear map (`V_REF = 140 km/h →
   N/2 − 1` bins). The constants `FC = 5.9 GHz` and `C` are defined but not used, and
   no subcarrier spacing or symbol time is modelled. For reference, 140 km/h at 5.9 GHz
   is about 765 Hz of Doppler. Taps are on-grid (no fractional delay or Doppler).
   Channels are synthetic, not ray-traced or 3GPP TDL/CDL.
8. **LMMSE baseline strength.** It uses a white prior with per-tap variance 1, while the
   true channel has total power 1 spread over a few of `L = 256` taps. As a result, it
   collapses onto LS (the two agree to within 0.07 dB). A statistics-aware or genie
   LMMSE would be a stronger reference.
9. **Statistics.** Fig. 1 uses 2 seeds; the other figures and the ablations use one seed.
   `σ²` per client is taken from its last realization. The OAMP variance `v_t` is a
   detached, batch-global scalar.
10. **Baselines are re-implementations**, not the authors' released code.

---

## 11. Code map

| File | Role |
|---|---|
| [otfs_data.py](otfs_data.py) | Pilot, true OTFS operator, row compressor, DD channel synthesis, scenario and velocity federations |
| [models.py](models.py) | `PFDUNHyper` (with `DopplerHyperNet`, `FiLMDenoiser`, `TemporalCore`), `CNNEstimator`, `LAMPNet`, LS/LMMSE, NMSE |
| [federated.py](federated.py) | `local_train`, `mobility_weight`, `aggregate`, `evaluate`, Trainer A `federated_train`, Trainer B `federated_train_phn` |
| [personalize.py](personalize.py) | Trainer C: `fit_local_thetas`, `distil_hypernet`, `train_personalized`, `mismatch_gap` |
| [test_personalization.py](test_personalization.py) | Oracle and ceiling study for the personalization mechanism |
| [make_paper_figures.py](make_paper_figures.py) | Figs. 1–8 (parametric studies) |
| [make_architecture.py](make_architecture.py) | Fig. 0 (system model diagram) |
| [run_analysis.py](run_analysis.py) / [plot_analysis.py](plot_analysis.py) | Ablation, embedding control, robustness, complexity, oracle headroom |
| [run_experiments.py](run_experiments.py) / [train.py](train.py) | SNR sweep suite / single-configuration runs |
| [replot_figures.py](replot_figures.py) | Restyle paper figures from CSV |
| [diag_conditioning.py](diag_conditioning.py) | Conditioning-collapse diagnostics |
| [results.md](results.md) | Detailed results write-up and verdict |
| [paper_figures/CAPTIONS.md](paper_figures/CAPTIONS.md) | Figure captions (IET Communications format) |
| [GUIDE.md](GUIDE.md) / [RELATED_WORK.md](RELATED_WORK.md) | Plain-language guide / related-work section |

---

## 12. Reproduction

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

All eight paper figures (Figs. 1–8, Trainer A, weak model):

```bash
.venv/bin/python make_paper_figures.py --samples 160
```

A subset of figures:

```bash
.venv/bin/python make_paper_figures.py --figs 6,5,3
```

The architecture figure (Fig. 0):

```bash
.venv/bin/python make_architecture.py
```

The personalization study (oracle, ceilings, three-stage route):

```bash
.venv/bin/python test_personalization.py --variants weak,strong
```

A single configuration:

```bash
.venv/bin/python train.py --N 16 --M 16 --use_hyper --rounds 30
```
