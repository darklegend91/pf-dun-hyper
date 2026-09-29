"""
=============================================================================
 models.py  --  All estimator networks and baselines
=============================================================================

This file contains every model compared in the paper:

  Classical baselines (no learning):
     ls_estimate     -- Least Squares (pseudo-inverse)
     lmmse_estimate  -- Linear MMSE

  Deep-learning baselines:
     CNNEstimator    -- ChannelNet-style CNN (LS init + convolutional refine)
     LAMPNet         -- Learned AMP unfolding (Borgerding & Schniter, 2017)

  Proposed model:
     PFDUNHyper      -- Personalized Federated Deep-Unfolding with a
                        Doppler-aware HYPERnetwork. Four techniques combined:
                          (A) deep unfolding / algorithm unrolling  (OAMP)
                          (B) hypernetwork + FiLM  (personalization)
                          (C) GRU state-space core (temporal, optional)
                          (D) low-rank "row compression" of the weight
                              generator (optional, shrinks federated payload)

Deep-learning techniques used, in one place:
  * Algorithm unrolling      -> OAMP / LAMP turned into a fixed-depth network
  * Hypernetwork             -> one network generates another network's weights
  * FiLM conditioning        -> feature-wise (scale, shift) modulation
  * Recurrent state-space    -> GRU cell carrying channel state across frames
  * Low-rank factorization   -> row compression to reduce parameters/payload

Complex numbers: linear algebra is done with torch complex tensors; the learned
denoiser works on a real/imag *stacked* vector (size 2L) because standard NN
layers are real-valued. c2r / r2c convert between the two.
"""
from __future__ import annotations
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Complex <-> real helpers
# --------------------------------------------------------------------------- #
def c2r(x: torch.Tensor) -> torch.Tensor:
    """(..., L) complex  ->  (..., 2L) real  = [real ; imag]."""
    return torch.cat([x.real, x.imag], dim=-1)


def r2c(x: torch.Tensor) -> torch.Tensor:
    """(..., 2L) real  ->  (..., L) complex."""
    L = x.shape[-1] // 2
    return torch.complex(x[..., :L], x[..., L:])


# =========================================================================== #
#  (A) DEEP UNFOLDING denoiser + (B) FiLM conditioning
# =========================================================================== #
class FiLMDenoiser(nn.Module):
    """
    The nonlinear step of one unfolding layer.

    Two parts:
      1. a model-based complex soft-threshold (shrinkage), which enforces the
         DD-domain sparsity prior -- the physics we know a-priori;
      2. a learned residual MLP whose hidden features are modulated by FiLM
         parameters (scale, shift) coming from the hypernetwork. FiLM is how
         the *same* shared denoiser is specialised per vehicle.
    """
    def __init__(self, L, hidden=256, strong=False):
        super().__init__()
        self.fc1 = nn.Linear(2 * L, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2 * L)
        self.act = nn.GELU()
        self.hidden = hidden
        self.strong = strong
        # Residual gate. In the weak variant this was a hard-coded 0.1, which
        # capped the conditioned branch at 10% of the output and was one reason
        # the mobility embedding barely mattered. Now learnable.
        self.res_gate = nn.Parameter(torch.tensor(0.4 if strong else 0.1))

    def forward(self, r_complex, tau, film_scale, film_shift, lam,
                film_scale2=None, film_shift2=None, lora=None):
        # 1) sparsity-promoting complex soft threshold: shrink small taps to 0
        mag = r_complex.abs()
        shrink = r_complex * (torch.clamp(mag - lam * tau, min=0.0) / (mag + 1e-8))

        # 2) learned residual with per-client conditioning
        x = c2r(shrink)
        h = self.act(self.fc1(x))
        h = h * (1 + film_scale) + film_shift            # FiLM #1
        h2 = self.fc2(h)

        # 2b) mobility-conditioned LOW-RANK WEIGHT ADAPTATION (LoRA-style).
        # FiLM can only rescale/shift activations; this lets the embedding
        # actually modify the denoiser's linear map:
        #     h2 <- W2 h + B(s) A(s) h
        if lora is not None:
            A, Bm = lora                                  # (B,r,hidden), (B,hidden,r)
            z = torch.einsum("bh,brh->br", h, A)
            h2 = h2 + torch.einsum("br,bhr->bh", z, Bm)

        h = self.act(h2)
        if film_scale2 is not None:                       # FiLM #2 (deeper conditioning)
            h = h * (1 + film_scale2) + film_shift2
        out = x + self.res_gate * self.fc3(h)
        return r2c(out)


# =========================================================================== #
#  (B) HYPERNETWORK  +  (D) ROW COMPRESSION (low-rank weight generation)
# =========================================================================== #
class DopplerHyperNet(nn.Module):
    """
    g_phi(s) -> parameters of the unfolding estimator.

    Input  s : per-vehicle mobility embedding
               [Doppler spread, delay spread, SNR, velocity, Rician-K].
    Output   : for every one of the T unfolding layers ->
               gamma (step size), lambda (shrink threshold),
               FiLM (scale, shift) for the denoiser.

    Row compression (compress=True):
      The FiLM heads must output T*hidden numbers -- that dominates the model
      size. We factor each head through a small rank-`compress_rank` bottleneck
      (W = A @ B, with A: hyp_hidden x r, B: r x T*hidden). This is a low-rank
      "row compression" of the weight generator and cuts the federated payload
      several-fold with negligible accuracy loss.
    """
    def __init__(self, emb_dim, T, hidden_denoiser, hyp_hidden=128,
                 compress=False, compress_rank=16, strong=False, cond_rank=4):
        super().__init__()
        self.T, self.hidden_denoiser = T, hidden_denoiser
        self.strong, self.cond_rank = strong, cond_rank
        self.trunk = nn.Sequential(
            nn.Linear(emb_dim, hyp_hidden), nn.GELU(),
            nn.Linear(hyp_hidden, hyp_hidden), nn.GELU(),
        )
        self.head_gamma = nn.Linear(hyp_hidden, T)
        self.head_lam = nn.Linear(hyp_hidden, T)
        out_dim = T * hidden_denoiser

        def head(dim):
            """A (optionally low-rank compressed) generating head."""
            if compress:
                return nn.Sequential(nn.Linear(hyp_hidden, compress_rank, bias=False),
                                     nn.Linear(compress_rank, dim))
            return nn.Linear(hyp_hidden, dim)

        self.head_scale = head(out_dim)
        self.head_shift = head(out_dim)
        if strong:
            self.head_scale2 = head(out_dim)
            self.head_shift2 = head(out_dim)
            # LoRA factors shared across the T unfolding layers (the denoiser
            # weights are shared across layers too, so this matches).
            self.head_lora = head(2 * cond_rank * hidden_denoiser)
            last = self.head_lora[-1] if compress else self.head_lora
            nn.init.zeros_(last.bias)                    # start as identity:
            nn.init.normal_(last.weight, std=1e-3)       # no adaptation at init

    def forward(self, s):
        B = s.shape[0]
        z = self.trunk(s)
        out = {
            "gamma": torch.sigmoid(self.head_gamma(z)) * 2.0,          # (B,T)
            "lam": torch.nn.functional.softplus(self.head_lam(z)),     # (B,T)
            "scale": self.head_scale(z).view(B, self.T, self.hidden_denoiser),
            "shift": self.head_shift(z).view(B, self.T, self.hidden_denoiser),
        }
        if self.strong:
            out["scale2"] = self.head_scale2(z).view(B, self.T, self.hidden_denoiser)
            out["shift2"] = self.head_shift2(z).view(B, self.T, self.hidden_denoiser)
            r, hd = self.cond_rank, self.hidden_denoiser
            lo = self.head_lora(z)
            out["lora"] = (lo[:, : r * hd].view(B, r, hd),
                           lo[:, r * hd:].view(B, hd, r))
        return out


# =========================================================================== #
#  (C) TEMPORAL state-space core (GRU) across OTFS frames
# =========================================================================== #
class TemporalCore(nn.Module):
    """Carries a latent channel state z across frames to exploit time-correlation."""
    def __init__(self, L, state=128):
        super().__init__()
        self.enc = nn.Linear(2 * L, state)
        self.gru = nn.GRUCell(state, state)
        self.dec = nn.Linear(state, 2 * L)
        self.state = state

    def init_state(self, B, device):
        return torch.zeros(B, self.state, device=device)

    def step(self, h_prev_complex, z):
        z = self.gru(self.enc(c2r(h_prev_complex)), z)
        return r2c(self.dec(z)), z                       # (bias, new_state)


# =========================================================================== #
#  PROPOSED: PF-DUN-Hyper
# =========================================================================== #
class PFDUNHyper(nn.Module):
    """
    Unrolled OAMP estimator whose per-layer parameters come from the Doppler
    hypernetwork, optionally with a temporal core and low-rank compression.

    forward(y_seq, Phi, sigma2, s, frames):
        y_seq : (B, F, Q) complex   received pilot observations
        Phi   : (Q, L)   complex    TRUE OTFS operator (shared)
        s     : (B, d)   float      mobility embeddings
        -> h_hat_seq (B, F, L) complex   estimated DD channels
    """
    def __init__(self, L, T=8, hidden_denoiser=256, emb_dim=5,
                 use_hyper=True, use_temporal=False,
                 compress=False, compress_rank=16,
                 strong_cond=False, cond_rank=4):
        super().__init__()
        self.L, self.T = L, T
        self.use_hyper, self.use_temporal = use_hyper, use_temporal
        self.strong_cond = strong_cond
        self.denoiser = FiLMDenoiser(L, hidden_denoiser, strong=strong_cond)

        if use_hyper:
            self.hyper = DopplerHyperNet(emb_dim, T, hidden_denoiser,
                                         compress=compress, compress_rank=compress_rank,
                                         strong=strong_cond, cond_rank=cond_rank)
        else:                                            # ablation: shared params, no FiLM
            self.gamma = nn.Parameter(torch.ones(T))
            self.lam = nn.Parameter(torch.ones(T) * 0.1)
            self.register_buffer("zero_film", torch.zeros(T, hidden_denoiser))

        if use_temporal:
            self.temporal = TemporalCore(L)

    # ---- one OAMP linear (de-correlated) step, using a cached eigendecomp ----
    def _oamp_linear(self, h_hat, y, PhiH, eigvals, eigvecs, sigma2, v_hat, gamma):
        """
        OAMP linear estimate:  r = h_hat + gamma * v * Phi^H A^{-1} (y - Phi h_hat),
        with A = v * (Phi Phi^H) + sigma2 I.

        SPEED-UP: A shares the same eigenvectors U as Phi Phi^H (only eigenvalues
        shift by v,sigma2), so we pre-compute U, e = eigh(Phi Phi^H) ONCE per
        forward and apply A^{-1} = U diag(1/(v e + sigma2)) U^H as cheap matmuls,
        instead of solving a Q x Q system in every one of the T layers.
        """
        resid = (y - h_hat @ PhiH.conj()).t()            # (Q,B) ; Phi.t() == PhiH.conj()
        Uh_r = eigvecs.conj().t() @ resid                # (Q,B)
        Ainv_r = eigvecs @ (Uh_r / (v_hat * eigvals + sigma2).unsqueeze(1))
        W_resid = (v_hat * (PhiH @ Ainv_r)).t()          # (B,L)
        return h_hat + gamma * W_resid

    @torch.no_grad()
    def gen_cond(self, s_mean):
        """Per-client conditioning parameters theta_i = g_phi(s_i), detached.

        Shapes drop the batch axis: gamma/lam (T,), scale/shift (T,hidden).
        Used by the pFedHN-style trainer, which optimises these locally and then
        regresses the hypernetwork onto them.
        """
        return {k: v.detach().clone() for k, v in self.cond_from_hyper(s_mean).items()}

    def cond_from_hyper(self, s_mean):
        """Same as gen_cond but differentiable w.r.t. the hypernetwork."""
        hp = self.hyper(s_mean.unsqueeze(0))
        out = {k: hp[k][0] for k in ("gamma", "lam", "scale", "shift")}
        if self.strong_cond:
            out["scale2"] = hp["scale2"][0]
            out["shift2"] = hp["shift2"][0]
            out["lora_A"] = hp["lora"][0][0]
            out["lora_B"] = hp["lora"][1][0]
        return out

    def forward(self, y_seq, Phi, sigma2, s, frames=1, cond=None):
        B, device = y_seq.shape[0], y_seq.device

        # --- pre-compute operator-dependent quantities ONCE (Phi is fixed) ---
        PhiH = Phi.conj().t()                             # (L,Q)
        PPh = Phi @ PhiH                                  # (Q,Q) Hermitian PSD
        eigvals, eigvecs = torch.linalg.eigh(PPh)        # e:(Q,) real, U:(Q,Q)
        eigvals = eigvals.clamp(min=0)

        # --- per-vehicle parameters (from hypernetwork or shared) ---
        if cond is not None:
            # Externally supplied per-client parameters (pFedHN local training):
            # one set per client, broadcast over the batch.
            gamma_all = cond["gamma"].unsqueeze(0).expand(B, -1)
            lam_all = cond["lam"].unsqueeze(0).expand(B, -1)
            scale_all = cond["scale"].unsqueeze(0).expand(B, -1, -1)
            shift_all = cond["shift"].unsqueeze(0).expand(B, -1, -1)
            hp = {}
            if "scale2" in cond:
                hp["scale2"] = cond["scale2"].unsqueeze(0).expand(B, -1, -1)
                hp["shift2"] = cond["shift2"].unsqueeze(0).expand(B, -1, -1)
            if "lora_A" in cond:
                hp["lora"] = (cond["lora_A"].unsqueeze(0).expand(B, -1, -1),
                              cond["lora_B"].unsqueeze(0).expand(B, -1, -1))
        elif self.use_hyper:
            hp = self.hyper(s)
            gamma_all, lam_all = hp["gamma"], hp["lam"]
            scale_all, shift_all = hp["scale"], hp["shift"]
        else:
            hp = {}
            gamma_all = self.gamma.expand(B, self.T)
            lam_all = self.lam.expand(B, self.T)
            scale_all = self.zero_film.unsqueeze(0).expand(B, self.T, -1)
            shift_all = scale_all

        outs = []
        z = self.temporal.init_state(B, device) if self.use_temporal else None
        h_prev = None
        for f in range(frames):
            y = y_seq[:, f, :]
            h_hat = torch.zeros(B, self.L, dtype=y.dtype, device=device)
            if self.use_temporal and h_prev is not None:      # (C) temporal bias
                bias, z = self.temporal.step(h_prev, z)
                h_hat = h_hat + bias
            for t in range(self.T):                            # (A) T unfolding layers
                v_hat = (h_hat.abs() ** 2).mean().clamp(min=1e-4).item() + 1e-3
                r = self._oamp_linear(h_hat, y, PhiH, eigvals, eigvecs,
                                      sigma2, v_hat, gamma_all[:, t:t + 1])
                tau = (r - h_hat).abs().mean(dim=-1, keepdim=True) + 1e-4
                h_hat = self.denoiser(
                    r, tau, scale_all[:, t, :], shift_all[:, t, :], lam_all[:, t:t + 1],
                    film_scale2=hp["scale2"][:, t, :] if "scale2" in hp else None,
                    film_shift2=hp["shift2"][:, t, :] if "shift2" in hp else None,
                    lora=hp.get("lora"))                        # (B) conditioned denoise
            outs.append(h_hat)
            h_prev = h_hat.detach() if self.use_temporal else None
        return torch.stack(outs, dim=1)


# =========================================================================== #
#  BASELINE 1: CNN (ChannelNet-style)
# =========================================================================== #
class CNNEstimator(nn.Module):
    """LS coarse estimate reshaped to an N x M 'image', refined by a CNN."""
    def __init__(self, N, M, channels=64, **kw):
        super().__init__()
        self.N, self.M, self.L = N, M, N * M
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 2, 3, padding=1),
        )

    def forward(self, y_seq, Phi, sigma2, s, frames=1):
        B = y_seq.shape[0]
        pinv = torch.linalg.pinv(Phi)
        outs = []
        for f in range(frames):
            h_ls = y_seq[:, f, :] @ pinv.t()
            img = torch.stack([h_ls.real, h_ls.imag], dim=1).view(B, 2, self.N, self.M)
            ref = self.net(img).view(B, 2, self.L)
            outs.append(torch.complex(ref[:, 0], ref[:, 1]) + h_ls)   # residual on LS
        return torch.stack(outs, dim=1)


# =========================================================================== #
#  BASELINE 2: LAMP (Learned AMP unfolding) -- the classic unfolding technique
# =========================================================================== #
class LAMPNet(nn.Module):
    """
    Learned AMP (Borgerding & Schniter 2017), the standard deep-unfolding
    baseline for sparse channel estimation (and the family used by the LAMP /
    row-compression paper). Each layer:

        r = h_hat + B (y - Phi h_hat)          # B is a LEARNED de-correlation matrix
        h_hat = soft_threshold(r, lambda_t)    # learned per-layer threshold

    Difference vs our OAMP backbone: LAMP *learns* the linear matrix B (tied
    across layers here), whereas OAMP *computes* the optimal LMMSE matrix from
    Phi. LAMP has no personalization and no temporal core.
    """
    def __init__(self, L, Q, T=8, **kw):
        super().__init__()
        self.L, self.Q, self.T = L, Q, T
        # learned complex de-correlation matrix B (L x Q), stored as real/imag.
        # Scale ~1/sqrt(Q) keeps the linear step well-conditioned at init.
        scale = 0.1 / (Q ** 0.5)
        self.B_re = nn.Parameter(torch.randn(L, Q) * scale)
        self.B_im = nn.Parameter(torch.randn(L, Q) * scale)
        self.lam = nn.Parameter(torch.ones(T) * 0.1)
        self.step = nn.Parameter(torch.zeros(T))         # per-layer damping (sigmoid)

    def forward(self, y_seq, Phi, sigma2, s, frames=1):
        B_mat = torch.complex(self.B_re, self.B_im).to(Phi.dtype)   # (L,Q)
        outs = []
        for f in range(frames):
            y = y_seq[:, f, :]
            h_hat = torch.zeros(y.shape[0], self.L, dtype=y.dtype, device=y.device)
            for t in range(self.T):
                resid = y - h_hat @ Phi.t()                 # (B,Q)
                gamma = torch.sigmoid(self.step[t])         # damping in (0,1) for stability
                r = h_hat + gamma * (resid @ B_mat.t())     # (B,L)
                mag = r.abs()
                tau = mag.mean(dim=-1, keepdim=True) + 1e-4  # adaptive threshold
                lam = torch.nn.functional.softplus(self.lam[t])
                h_hat = r * (torch.clamp(mag - lam * tau, min=0.0) / (mag + 1e-8))
            outs.append(h_hat)
        return torch.stack(outs, dim=1)


# =========================================================================== #
#  Classical baselines and metrics
# =========================================================================== #
def ls_estimate(y, Phi):
    """Least squares via pseudo-inverse."""
    return y @ torch.linalg.pinv(Phi).t()


def lmmse_estimate(y, Phi, sigma2, ch_var=1.0):
    """LMMSE with a white channel prior."""
    Q = Phi.shape[0]
    A = ch_var * (Phi @ Phi.conj().t()) + sigma2 * torch.eye(Q, dtype=Phi.dtype, device=Phi.device)
    sol = torch.linalg.solve(A, y.t())
    return (ch_var * (Phi.conj().t() @ sol)).t()


def nmse(h_hat, h):
    """Normalised MSE = ||h_hat - h||^2 / ||h||^2, averaged over the batch."""
    num = (h_hat - h).abs().pow(2).sum(dim=-1)
    den = h.abs().pow(2).sum(dim=-1).clamp(min=1e-9)
    return (num / den).mean()


def nmse_db(h_hat, h):
    return 10 * torch.log10(nmse(h_hat, h))
