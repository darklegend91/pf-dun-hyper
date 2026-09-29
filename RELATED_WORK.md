# Related Work

Our method, **PF-DUN-Hyper**, sits at the intersection of three active research
threads: (i) deep-unfolding channel estimation for OTFS/delay-Doppler systems,
(ii) hypernetwork-based personalized federated learning, and (iii) federated
learning for vehicular (V2X) wireless. We review each and then position our
contribution against the gap they leave open.

## Deep unfolding for OTFS channel estimation

Model-based deep learning, in which a proven iterative estimator is *unrolled*
into a fixed-depth trainable network, has become the mainstream approach to
sparse channel estimation because it inherits the sample-efficiency and
interpretability of the underlying algorithm while adapting its parameters from
data [1]. Learned AMP (LAMP) and its orthogonal variant (OAMP) are the canonical
backbones, turning ISTA/AMP iterations into layers of a linear de-correlation
step followed by a learned shrinkage denoiser [1]. For OTFS specifically, the
delay-Doppler representation renders the vehicular multipath channel sparse and
quasi-stationary [2], and recent work exploits this with sparse-prior-guided
and learned-denoising unfolding estimators [3]. These methods, however, learn a
*single* global estimator and are trained and evaluated centrally; they neither
personalize across heterogeneous mobility regimes nor address the privacy and
bandwidth constraints of distributed vehicular training.

## Hypernetwork-based personalized federated learning

Personalized federated learning (pFL) addresses statistical heterogeneity by
tailoring a model to each client rather than forcing a single global consensus.
A prominent line of work uses a *hypernetwork* — a network that generates the
weights of another network — to produce client-specific parameters from a compact
client embedding, so that only the shared hypernetwork is communicated while the
personalized weights are generated locally and never transmitted [4]. This yields
substantial communication savings; decentralized hypernetwork-aggregation schemes
report reductions of up to ~88% in transmitted parameters per round [5]. In these
frameworks, however, the conditioning embedding is *learned* or abstract (a
client id or a data-driven latent), with no physical grounding, and the target
task is generic classification rather than a physics-constrained signal-recovery
problem.

## Federated learning for vehicular wireless

Federated and meta-learning have been applied to channel estimation to keep raw
CSI on-device: FedMetaCE personalizes per-base-station estimators via federated
meta-learning [6], and robust-FL demonstrations use channel estimation as a test
case for heterogeneity- and noise-robust aggregation [7]. In the vehicular
setting, most FL research treats mobility as a *systems* concern — client
selection, staleness, and convergence under churn [8, 9] — optimizing *who*
participates and *when*, rather than feeding the vehicle's physical mobility
state into the estimator itself.

## Positioning and contribution

Across these threads, each ingredient we use exists in isolation, but their
combination for high-mobility OTFS estimation is, to our knowledge, unaddressed.
Deep-unfolding OTFS estimators [1–3] are global and centralized; hypernetwork
pFL [4, 5] personalizes on abstract embeddings for generic tasks; and vehicular
FL [6–9] uses mobility only for scheduling. **PF-DUN-Hyper is the first to
condition a hypernetwork on a *physical* mobility embedding — Doppler spread,
delay spread, SNR, velocity, and Rician K-factor — to generate the per-layer
parameters of an OAMP-unfolding OTFS estimator, trained under a
mobility-weighted federated averaging scheme over non-IID vehicular clients.**
Concretely, our contributions are:

1. A **physics-conditioned personalization** mechanism: a Doppler-aware
   hypernetwork with FiLM modulation maps each vehicle's measurable mobility
   state to its own unfolding estimator, so heterogeneity is handled by physics
   rather than by a per-vehicle model or an abstract latent.
2. A **mobility-weighted, staleness-robust aggregation** rule that up-weights
   rare high-Doppler clients so the shared model does not collapse onto the
   dominant low-mobility regime.
3. A **low-rank "row compression"** of the weight-generating heads that roughly
   halves the federated payload with negligible accuracy loss, complementing the
   communication savings reported for prior hypernetwork pFL [5].
4. A reproducible ablation ladder (LS/LMMSE → CNN → LAMP → OAMP-DUN+FedAvg →
   PF-DUN-Hyper) that isolates the marginal value of personalization and
   compression on non-IID vehicular OTFS channels.

---

### References

[1] *Comprehensive Review of Deep Unfolding Techniques for Next-Generation
Wireless Communication Systems*, arXiv:2502.05952, 2025.
https://arxiv.org/pdf/2502.05952

[2] *New Delay-Doppler Communication Paradigm in the 6G Era: A Survey of OTFS*,
arXiv:2211.12955. https://arxiv.org/pdf/2211.12955

[3] *Multi-Snapshot Deep Denoising for Channel Estimation in OTFS-Modulated
Systems*, arXiv:2605.29777. https://arxiv.org/html/2605.29777
(see also: sparse-prior-guided DL for OTFS CE, IEEE TVT 2024; learned-denoising
sparse adaptive CE for OTFS, IEEE WCL 2024.)

[4] *HyperFedNet: Communication-Efficient Personalized Federated Learning via
Hypernetwork*, arXiv:2402.18445, 2024. https://arxiv.org/abs/2402.18445

[5] *Hypernetwork Aggregation for Decentralized Personalized Federated Learning*,
IJCAI 2025. https://www.ijcai.org/proceedings/2025/161

[6] *Federated Meta-Learning for Personalized Channel Estimation in Massive MIMO
Systems (FedMetaCE)*, IJMLCN.
https://iaeme.com/Home/article_id/IJMLCN_01_01_001

[7] *Robust Federated Learning for Wireless Networks: A Demonstration with
Channel Estimation*, arXiv:2404.03088. https://arxiv.org/pdf/2404.03088

[8] *Mobility-Aware Decentralized Federated Learning with Joint Optimization of
Local Iteration and Leader Selection for Vehicular Networks*, arXiv:2503.06443.
https://arxiv.org/pdf/2503.06443

[9] *Vehicle Selection for C-V2X Mode 4 Based Federated Edge Learning Systems*,
arXiv:2401.07224. https://arxiv.org/pdf/2401.07224
</content>
</invoke>
