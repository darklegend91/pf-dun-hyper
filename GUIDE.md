# GUIDE — What this code does, how, and the techniques it uses

This is a complete, self-contained research codebase for **federated channel
estimation in high-mobility vehicular OTFS systems**. It implements a novel
model (**PF-DUN-Hyper**), several baselines, and a reproducible experiment
suite that produces publication-ready tables and figures.

This guide is written for a reader who knows deep learning but wants to
understand (a) the scenario, (b) every deep-learning technique used and *why*,
and (c) exactly how federated learning is wired in.

---

## 1. The scenario (the problem being solved)

**Vehicles** driving in different environments (city, highway, rural road,
tunnel) each need to estimate their **wireless channel** — the distortion the
radio signal picks up on its way from transmitter to receiver. Accurate channel
estimates are required to decode data.

Three things make this hard, and this code targets all three:

| Challenge | What it means | Where it's handled |
|---|---|---|
| **High mobility** | Fast movement → strong Doppler → the channel changes quickly | Doppler-aware model + temporal core |
| **Heterogeneity (non-IID)** | A tunnel channel ≠ a highway channel; one global model fits none well | Hypernetwork personalization |
| **Privacy + bandwidth** | Raw channel data is sensitive (leaks location) and heavy to transmit | Federated learning |

**Waveform: OTFS** (Orthogonal Time Frequency Space). Unlike OFDM, OTFS
represents the signal in the **delay–Doppler (DD) domain**, where a
fast-fading multipath channel becomes **sparse and stable** — only a few
"taps" are non-zero. That sparsity is the structure our model exploits.

The estimation problem is a **linear inverse problem**:

```
   y   =   Φ · h   +   n
 (obs)   (op) (channel) (noise)
```

- `h` — the sparse DD-domain channel we want (length L = N·M).
- `Φ` — the **true OTFS operator**: a known pilot signal, arranged as a 2-D
  circular-convolution matrix (see `otfs_data.build_otfs_operator`). This is the
  real OTFS input–output relation, *not* a random sensing matrix.
- `y` — the received pilot observations (length Q).
- `n` — additive noise.

Estimate `h` from `y`, knowing `Φ`.

---

## 2. The files

| File | Role |
|---|---|
| `otfs_data.py` | Simulates DD channels (exponential power profile + Jakes Doppler), builds the true OTFS operator, and splits vehicles into non-IID clients. |
| `models.py` | All estimators: classical (LS, LMMSE), baselines (CNN, LAMP), and the proposed PF-DUN-Hyper. |
| `federated.py` | The federated learning loop + custom aggregation. |
| `train.py` | Train/evaluate a single model configuration. |
| `run_experiments.py` | The full multi-seed SNR sweep → tables + figures in `results/`. |

---

## 3. The deep-learning techniques (and why each is used)

The proposed model, **PF-DUN-Hyper**, is a *hybrid* of four techniques. Each
targets a specific challenge above.

### (A) Deep unfolding / algorithm unrolling — the backbone
Instead of a generic black-box network, we take a proven iterative estimation
algorithm (**OAMP** — Orthogonal Approximate Message Passing) and "unroll" its
`T` iterations into `T` network layers. Each layer =
1. a **linear step** that pulls the estimate toward consistency with `y = Φh`
   (an LMMSE-style correction computed from `Φ`), then
2. a **nonlinear denoiser** that enforces DD sparsity (soft-thresholding + a
   small learned network).

*Why:* it bakes the known physics (`Φ`, sparsity) into the architecture, so it
needs **far fewer parameters** than a black-box net — which is critical because
in federated learning those parameters are transmitted every round.
→ `models.PFDUNHyper`, method `_oamp_linear` + class `FiLMDenoiser`.

### (B) Hypernetwork + FiLM — the personalization
A **hypernetwork** is a network that outputs the *weights of another network*.
Here `g_φ(s)` reads a vehicle's **mobility embedding**
`s = [Doppler spread, delay spread, SNR, velocity, Rician-K]` and generates the
unfolding estimator's per-layer parameters. The generated parameters modulate
the shared denoiser via **FiLM** (Feature-wise Linear Modulation: a per-feature
`scale` and `shift`).

*Why:* a tunnel vehicle and a highway vehicle get **different estimators**,
produced on-the-fly from their physical state — without training a separate
model per vehicle. This is what handles the non-IID challenge.
→ `models.DopplerHyperNet`.

### (C) Recurrent state-space core — the temporal tracker (optional)
A **GRU** carries a latent channel state across consecutive OTFS frames, so the
estimate at frame *f* benefits from frames *< f*.

*Why:* the channel is correlated in time; a memoryless per-frame estimator
throws that information away. Enable with `--use_temporal`.
→ `models.TemporalCore`.

### (D) Low-rank "row compression" — shrinking the payload (optional)
The hypernetwork's weight-generating heads dominate the model size. We factor
them through a small rank-`r` bottleneck (`W = A·B`), a **low-rank
factorization**. This is the "row compression" idea: it cuts the number of
parameters (hence the federated transmission cost) roughly in half with
negligible accuracy loss.

*Why:* directly reduces communication cost — the whole point of FL efficiency.
Enable with `compress=True` (the `PF-DUN-Hyper-RC` method).
→ `DopplerHyperNet(..., compress=True)`.

**A second, data-side compression** (`--row_compress`) shrinks the *observation*
`y` with a fixed orthonormal matrix, reducing Q → faster math and lower pilot
overhead. → `otfs_data.build_row_compressor`.

### Speed note
The OAMP linear step needs `A⁻¹` where `A = v·ΦΦᴴ + σ²I`. Since `A` shares
eigenvectors with `ΦΦᴴ` (only eigenvalues shift), we compute one
eigendecomposition **once per forward pass** and apply `A⁻¹` as cheap
matrix–vector products in every layer — instead of solving a Q×Q system `T`
times. This is the main runtime optimization. → `PFDUNHyper._oamp_linear`.

---

## 4. How federated learning is used

The goal: train **one shared model across many vehicles without any vehicle
sending its raw channel data** to the server.

The loop (in `federated.py`), each **round**:

1. **Server → clients:** broadcast the current shared parameters `Θ`
   (the denoiser + hypernetwork + temporal core).
2. **Local training** (`local_train`): each selected vehicle trains `Θ` on its
   *own* private data for a few steps, minimizing NMSE.
3. **Client → server:** vehicles send back only their updated `Θ`
   (**never** raw channels `h`, `y`).
4. **Aggregation** (`aggregate` + `mobility_weight`): the server averages the
   client models. Our twist — the average is **mobility-weighted**: vehicles in
   harder, high-Doppler regimes get up-weighted so the shared model doesn't
   collapse onto the most common (slow) scenario. Staleness (late updates from
   fast-moving vehicles dropping in/out) is discounted.

**Why personalization + FL fit together:** the *shared* hypernetwork `g_φ` is
what's federated. Each vehicle then generates its *own personalized* estimator
locally via `g_φ(s)`. So we transmit one small shared network, but every
vehicle runs a model specialized to its mobility — privacy-preserving *and*
personalized.

**Non-IID clients** (`make_federation`): vehicles are split across 4 physical
scenarios (`urban_slow`, `highway`, `rural`, `tunnel`) with different velocity,
Doppler spread, and delay spread — so the heterogeneity is *physical*, not an
artificial data split. This is the realistic, hard setting for FL.

---

## 5. The methods compared

| Method | Type | Personalized? | Federated? |
|---|---|---|---|
| **LS** | classical least-squares | — | — |
| **LMMSE** | classical linear MMSE | — | — |
| **CNN + FedAvg** | deep-learning (ChannelNet-style CNN) | no | yes |
| **LAMP + FedAvg** | deep unfolding (Learned AMP) | no | yes |
| **DUN + FedAvg** | deep unfolding (OAMP), no hypernetwork | no | yes |
| **PF-DUN-Hyper** | proposed | **yes** | yes |
| **PF-DUN-Hyper-RC** | proposed + low-rank compression | **yes** | yes |

The ladder LAMP/DUN → PF-DUN-Hyper isolates the value of **personalization**;
PF-DUN-Hyper → PF-DUN-Hyper-RC isolates the cost of **compression**.

---

## 6. Reproducing the results

```bash
# environment (Python 3.11)
.venv/bin/python --version

# full publish-grade sweep (tables + figures in results/)
.venv/bin/python run_experiments.py --N 16 --M 16 \
    --snr_list 0,5,10,15,20 --rounds 30 --samples 384 \
    --clients 8 --frames 2 --seeds 3

# faster: fewer seeds/rounds + data-side row compression
.venv/bin/python run_experiments.py --snr_list 0,10,20 \
    --rounds 20 --seeds 2 --row_compress 0.7

# single model / ablation
.venv/bin/python train.py --use_hyper --use_temporal --frames 2 --rounds 30
```

Outputs in `results/`: `nmse_vs_snr.png`, `convergence.png`, `comm_cost.png`,
`results_table.md`, `results.csv`.

---

## 7. Metric

**NMSE (Normalized Mean Squared Error), in dB, lower is better:**
`NMSE = ‖ĥ − h‖² / ‖h‖²`. −20 dB means the estimation error power is 1% of the
channel power. It is the standard channel-estimation metric.

---

## 8. One-paragraph summary for a paper abstract

> We propose PF-DUN-Hyper, a personalized federated deep-unfolding estimator for
> high-mobility vehicular OTFS channel estimation. An OAMP-based unfolding
> backbone provides a compact, physics-grounded estimator; a Doppler-aware
> hypernetwork generates per-vehicle estimator parameters via FiLM modulation,
> delivering personalization across heterogeneous mobility regimes without
> sharing raw CSI; and a low-rank "row-compression" of the weight generator
> halves the federated payload with negligible accuracy loss. Trained with a
> mobility-weighted federated averaging scheme over non-IID vehicular clients,
> PF-DUN-Hyper outperforms LS, LMMSE, CNN, and unfolding (LAMP/OAMP) baselines
> by a wide margin across all SNRs.
