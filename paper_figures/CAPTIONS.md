# Figure captions — IET Communications format

**Common simulation setup.** OTFS delay–Doppler grid `N = M = 16` (`L = 256`); true
twisted-convolution pilot operator built from a fixed QPSK delay–Doppler pilot; pilot
overhead `Q/L = 0.6` unless swept; exponential power-delay profile with Clarke/Jakes
Doppler at `f_c = 5.9 GHz`; `K = 4` non-IID vehicular clients drawn from four mobility
regimes (urban, highway, rural, tunnel); federated averaging with a fair per-method
training budget (patience-based early stopping, best checkpoint restored), so no estimator
is penalised by a round budget chosen to suit another.

Figures are supplied at 600 dpi (`.png`) and as vector `.pdf`, sized for a single IET
column (≈ 8.5 cm). Raw numbers accompany each figure as `.csv`. All figures can be
restyled without retraining via `python replot_figures.py`.

---

**Fig. 0 (system model)** Architecture of the proposed PF-DUN-Hyper estimator.
(a) Federated system: `K` non-IID vehicles drawn from distinct mobility regimes keep their
own pilot observations and mobility embeddings; only the shared denoiser `W` and the
hypernetwork `g_phi` are exchanged with the server. Training proceeds in three stages:
federated averaging of the shared denoiser, per-vehicle fitting of the conditioning
parameters, and distillation of the hypernetwork onto the mapping `s_i -> theta_i*`.
(b) Per-vehicle estimator: the mobility embedding drives the hypernetwork, whose output
conditions every layer of an unrolled OAMP network through per-layer step sizes, shrinkage
thresholds, FiLM modulation and low-rank (LoRA) adaptation of the denoiser weights.
-> `fig0_architecture.pdf`

**Fig. 1** Normalised mean square error versus signal-to-noise ratio for the proposed
PF-DUN-Hyper estimator and its row-compressed variant PF-DUN-Hyper-RC, compared with
classical (LS, LMMSE), CNN and deep-unfolding (LAMP, OAMP-DUN) baselines. Error bars
denote one standard deviation over two independent channel realisations. *(LS and LMMSE
coincide to within 0.07 dB and overlap on the plot.)*
→ `fig1_nmse_vs_snr.pdf` — proposed reaches **−25.7 dB at 20 dB SNR** versus −7.9 dB
(LAMP), −7.7 dB (OAMP-DUN) and −4.9 dB (CNN): a **≈17.8 dB** margin.

**Fig. 2** Normalised mean square error versus pilot overhead `Q/L` at 10 dB SNR.
→ `fig2_pilot_overhead.pdf` — the proposed estimator at `Q/L = 0.3` (**−13.2 dB**)
outperforms every baseline at `Q/L = 0.9` (best −10.3 dB), i.e. it attains better accuracy
with **one third of the pilot resources**.

**Fig. 3** Normalised mean square error versus vehicle velocity at 10 dB SNR. Estimators
are trained on a fixed set of velocities and evaluated across the full mobility range,
including velocities absent from training.
→ `fig3_velocity.pdf` — the proposed estimator is essentially flat
(**−17.4 to −18.9 dB** over 10–150 km/h), whereas LAMP degrades from −9.9 dB to −3.0 dB as
velocity increases. Describe this as **robustness across the mobility range** (see note).

**Fig. 4** Convergence of federated training at 10 dB SNR: normalised mean square error
against communication round. Curves terminate where the fair-budget early-stopping
criterion is met.
→ `fig4_convergence.pdf` — the proposed estimator reaches −15.5 dB within **4 rounds**,
while the baselines have not passed −5 dB at that point.

**Fig. 5** Effect of unfolding depth: normalised mean square error versus the number of
unrolled OAMP layers `T` at 10 dB SNR.
→ `fig5_depth.pdf` — accuracy saturates at **T ≈ 4–6** (−16.7 dB); further layers give no
gain, which bounds the per-estimate complexity.

**Fig. 6** Row-compression trade-off: normalised mean square error against federated
payload per communication round, as the low-rank factorisation rank of the hypernetwork's
weight-generating heads is varied.
→ `fig6_rank.pdf` — rank 16 reduces the payload by **52 %** (876 k → 422 k parameters)
while *improving* NMSE by 0.21 dB; rank 2 gives a **59 %** reduction for a 0.09 dB penalty.

**Fig. 7** Federated scalability: normalised mean square error versus the number of
participating vehicles `K` at 10 dB SNR, with the total training data held constant so the
abscissa isolates the effect of federation (per-client data falls as `K` grows).
→ `fig7_clients.pdf` — the proposed estimator degrades gracefully (−19.0 dB at `K = 2` to
−15.2 dB at `K = 12`), whereas the CNN collapses from −7.6 dB to −2.6 dB.

**Fig. 8** End-to-end uncoded bit error rate versus signal-to-noise ratio for QPSK
transmission, after delay–Doppler-domain LMMSE equalisation using each estimator's channel
estimate. *(BER resolution ≈ 5 × 10⁻⁵ at the sample count used.)*
→ `fig8_ber.pdf` — at 15 dB the proposed estimator attains **1.6 × 10⁻³** against
4.7 × 10⁻² (LAMP) and 2.5 × 10⁻¹ (LMMSE), confirming the NMSE gain translates into link
performance.

---

## ⚠️ Note on claim–figure alignment

These figures accurately report what the code measures. Before writing the accompanying
text, note which claims they do and do not support (see `results.md` §7.2 and §7.5).

**Supported by these figures**
- Large, consistent NMSE and BER gains over classical, CNN and unfolding baselines
  (Figs. 1, 8).
- Superior pilot efficiency (Fig. 2) and graceful federated scaling (Fig. 7).
- Faster convergence, i.e. fewer communication rounds (Fig. 4).
- Row compression cuts the federated payload by ≈52 % at no accuracy cost (Fig. 6).
- Stable accuracy across the full velocity range (Fig. 3).

**NOT established by these figures**
- That the gain arises from *Doppler-based personalisation*. A control experiment
  (`results.md` §7.2) shows that substituting one vehicle's mobility embedding for
  another's changes NMSE by only **0.04 dB**, while an oracle per-scenario study (§7.5)
  shows **≈2.08 dB** of personalisation headroom exists but is not captured. The measured
  driver of the gain is the FiLM-modulated residual denoiser, not mobility conditioning.

**Consequence.** Fig. 3 should be written as *robustness across the mobility range*, not as
evidence of per-vehicle adaptation, and the contribution should be framed around the
unfolding + FiLM denoiser and row compression. Claiming personalisation as the mechanism
without addressing §7.2 would not survive review.
