"""
OTFS delay-Doppler channel model and non-IID federated client generation.

Physical model (documented assumptions for the paper)
-----------------------------------------------------
* The channel is sparse in the delay-Doppler (DD) domain:
      h(tau, nu) = sum_i h_i delta(tau - tau_i) delta(nu - nu_i)
  with taps placed on the DD grid (on-grid / integer delay-Doppler assumption,
  standard in OTFS estimation literature, e.g. Raviteja et al. 2018/2019).
* Power-delay profile: exponential PDP with a per-scenario RMS delay spread.
* Doppler: each tap's Doppler follows Clarke/Jakes,  nu_i = nu_max cos(theta_i),
  theta_i ~ U(0, 2pi), nu_max set by the vehicle velocity. This produces the
  classic U-shaped Doppler spectrum per scenario.
* TRUE OTFS input-output relation. With a known DD-domain pilot symbol x, the
  received DD samples are the 2D *twisted (circular) convolution* of x with the
  DD channel:
        y[k,l] = sum_{k',l'} x[(k-k') mod N, (l-l') mod M] h[k',l']  + n
  i.e.  y = Phi h + n,  where Phi is the block-circulant OTFS operator built
  from the pilot (NOT a random sensing matrix). Reduced pilot overhead is
  modelled by observing only Q of the L DD bins (Phi has Q rows).

The N=M=16 grid is a downscaled but representative OTFS frame; the operator,
PDP and Doppler statistics are the physically meaningful parts and scale
directly to larger frames.
"""
from __future__ import annotations
import numpy as np
import torch

FC = 5.9e9          # carrier frequency (Hz), C-V2X band
C = 3e8             # speed of light
V_REF = 140.0       # km/h reference mapping velocity -> normalised Doppler grid
_PILOT_SEED = 12345 # fixed pilot is a protocol constant, shared by all vehicles


# --------------------------------------------------------------------------- #
# Mobility scenarios -> statistical heterogeneity (non-IID by physics)
# --------------------------------------------------------------------------- #
SCENARIOS = {
    #                velocity km/h   #taps      rms delay (frac of M)  Rician K
    "urban_slow": dict(v_range=(10, 40),  n_paths=(6, 10), ds_frac=0.35, k_factor=0.0),
    "highway":    dict(v_range=(90, 140), n_paths=(3, 5),  ds_frac=0.20, k_factor=6.0),
    "rural":      dict(v_range=(50, 90),  n_paths=(2, 4),  ds_frac=0.15, k_factor=8.0),
    "tunnel":     dict(v_range=(30, 70),  n_paths=(8, 14), ds_frac=0.45, k_factor=0.0),
}


def make_pilot(N: int, M: int) -> np.ndarray:
    """Fixed unit-energy QPSK DD-domain pilot (protocol constant)."""
    rng = np.random.default_rng(_PILOT_SEED)
    sym = rng.integers(0, 4, size=(N, M))
    x = np.exp(1j * (np.pi / 4 + sym * np.pi / 2)).astype(np.complex64)
    x /= np.sqrt(N * M)                     # so each column of Phi has unit norm
    return x


def build_otfs_operator(x: np.ndarray, N: int, M: int) -> np.ndarray:
    """
    True OTFS twisted-convolution operator Phi (L x L), L = N*M.
    Column (k',l') is the DD pilot circularly shifted by (k',l'):
        Phi[:, k'*M + l'] = vec( roll(x, (k', l')) ).
    """
    L = N * M
    Phi = np.zeros((L, L), dtype=np.complex64)
    for kp in range(N):
        for lp in range(M):
            col = np.roll(np.roll(x, kp, axis=0), lp, axis=1).reshape(-1)
            Phi[:, kp * M + lp] = col
    return Phi


def doppler_max_bin(v_kmh: float, N: int) -> float:
    """Normalised max Doppler as a fraction of the Doppler grid half-width."""
    return (v_kmh / V_REF) * (N / 2 - 1)


def sample_dd_channel(scenario, N: int, M: int, rng: np.random.Generator):
    """
    Sample a sparse DD channel vector h in C^L and mobility embedding s.

    `scenario` is either a key of SCENARIOS or a config dict (used by the
    velocity-sweep / generalization studies to pin an exact velocity).
    """
    cfg = scenario if isinstance(scenario, dict) else SCENARIOS[scenario]
    v = rng.uniform(*cfg["v_range"])
    P = int(rng.integers(cfg["n_paths"][0], cfg["n_paths"][1] + 1))
    k_max = doppler_max_bin(v, N)
    L = N * M
    h = np.zeros(L, dtype=np.complex64)

    # --- delays: exponential PDP with per-scenario RMS delay spread ---
    ds_bins = max(1.0, cfg["ds_frac"] * M)
    delays = np.minimum(rng.exponential(ds_bins, size=P).astype(int), M - 1)
    powers = np.exp(-delays / ds_bins).astype(np.float32)
    powers /= powers.sum()

    # --- Doppler: Clarke/Jakes, nu = nu_max cos(theta) ---
    theta = rng.uniform(0, 2 * np.pi, size=P)
    dopplers = np.rint(k_max * np.cos(theta)).astype(int) % N

    for p in range(P):
        pw = powers[p]
        if p == 0 and cfg["k_factor"] > 0:                 # Rician LoS on first tap
            k = cfg["k_factor"]
            los = np.sqrt(k / (k + 1) * pw)
            nlos = np.sqrt(pw / (k + 1) / 2) * (rng.standard_normal() + 1j * rng.standard_normal())
            g = los + nlos
        else:
            g = np.sqrt(pw / 2) * (rng.standard_normal() + 1j * rng.standard_normal())
        h[dopplers[p] * M + delays[p]] += g

    s = np.array([k_max / N,                         # normalised Doppler spread
                  delays.max() / M if P else 0.0,    # normalised delay spread
                  0.0,                               # SNR placeholder (set later)
                  v / V_REF,                         # normalised velocity
                  cfg["k_factor"] / 10.0],           # Rician factor
                 dtype=np.float32)
    return h, s


def build_row_compressor(Q: int, Qc: int) -> np.ndarray:
    """
    Fixed orthonormal ROW-COMPRESSION matrix S (Qc x Q), Qc < Q.

    Applied to both the observation and the operator:  y_c = S y,  Phi_c = S Phi.
    This reduces the number of observed rows from Q to Qc, which (i) shrinks the
    OAMP Q x Q eigendecomposition -> faster, and (ii) lowers pilot/feedback
    overhead. S is a protocol constant (fixed seed) so it costs NO parameters.
    """
    rng = np.random.default_rng(_PILOT_SEED + 2)
    G = rng.standard_normal((Qc, Q)) + 1j * rng.standard_normal((Qc, Q))
    U, _, Vh = np.linalg.svd(G, full_matrices=False)     # orthonormalise rows
    return (U @ Vh).astype(np.complex64)


class ClientDataset:
    """One vehicle: channels, pilot observations, shared OTFS operator, embedding."""
    def __init__(self, scenario, N, M, Phi_np, obs_idx, snr_db, n_samples, seed,
                 frames=1, S=None):
        self.scenario = scenario
        self.N, self.M, self.L = N, M, N * M
        self.snr_db = snr_db
        self.frames = frames
        Phi_obs = Phi_np[obs_idx, :]                  # (Q, L) TRUE OTFS operator
        if S is not None:                             # optional row compression
            Phi_obs = (S @ Phi_obs).astype(np.complex64)
        self.Phi_np = Phi_obs
        self.Q = Phi_obs.shape[0]
        self.row_S = S                                # row-compression matrix (or None)
        self.Phi = torch.from_numpy(self.Phi_np)
        rng = np.random.default_rng(seed)

        H, S, Y = [], [], []
        sigma2 = 1.0
        for _ in range(n_samples):
            h0, s = sample_dd_channel(scenario, N, M, rng)
            seq_h, seq_y = [], []
            h_prev = h0
            for f in range(frames):
                if f == 0:
                    h = h0
                else:                                # AR(1) temporal evolution
                    innov = 0.15 * (rng.standard_normal(self.L)
                                    + 1j * rng.standard_normal(self.L)).astype(np.complex64)
                    h = 0.98 * h_prev + innov * (np.abs(h_prev) > 0)
                h_prev = h
                y_clean = self.Phi_np @ h
                sig_p = np.mean(np.abs(y_clean) ** 2) + 1e-9
                sigma2 = sig_p / (10 ** (snr_db / 10))
                noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(self.Q)
                                               + 1j * rng.standard_normal(self.Q))
                seq_h.append(h)
                seq_y.append((y_clean + noise).astype(np.complex64))
            H.append(np.stack(seq_h)); Y.append(np.stack(seq_y)); S.append(s)

        self.H = torch.from_numpy(np.stack(H))       # (Ns,F,L) complex
        self.Y = torch.from_numpy(np.stack(Y))       # (Ns,F,Q) complex
        self.S = torch.from_numpy(np.stack(S))       # (Ns,d) float
        # Populate the SNR slot of the mobility embedding. sample_dd_channel()
        # cannot know the operating SNR, so it leaves index 2 as a placeholder;
        # filling it here makes all five embedding features live.
        self.S[:, 2] = float(snr_db) / 30.0
        self.sigma2 = torch.tensor(float(sigma2), dtype=torch.float32)

    def __len__(self):
        return self.H.shape[0]

    def batch(self, idx):
        return self.H[idx], self.Y[idx], self.S[idx]


def make_velocity_federation(velocities, N, M, Q, snr_db, samples_per_client,
                             frames=1, seed=0, base="highway", row_compress=1.0):
    """
    One client per *exact* velocity (km/h). Used for:
      * NMSE vs velocity curves,
      * zero-shot generalization to velocities never seen in training.
    All other channel parameters are held at the `base` scenario so velocity
    is the only variable.
    """
    x = make_pilot(N, M)
    Phi_full = build_otfs_operator(x, N, M)
    L = N * M
    obs_rng = np.random.default_rng(_PILOT_SEED + 1)
    obs_idx = np.sort(obs_rng.choice(L, size=Q, replace=False)) if Q < L else np.arange(L)
    S = None
    if row_compress < 1.0:
        S = build_row_compressor(Q, max(1, int(round(row_compress * Q))))

    clients = []
    for i, v in enumerate(velocities):
        cfg = dict(SCENARIOS[base]); cfg["v_range"] = (v, v)     # pin the velocity
        c = ClientDataset(cfg, N, M, Phi_full, obs_idx, snr_db, samples_per_client,
                          seed=2000 + i + 97 * seed, frames=frames, S=S)
        c.scenario = f"v={v:.0f}"                                # label for reporting
        c.velocity = v
        clients.append(c)
    return clients


def make_federation(n_clients, N, M, Q, snr_db, samples_per_client, frames=1,
                    seed=0, row_compress=1.0):
    """
    Non-IID federation with a single shared, true OTFS pilot operator.

    row_compress : fraction in (0,1]. If <1, the Q observed rows are compressed
                   to Qc = round(row_compress * Q) via a fixed orthonormal
                   matrix (speeds up OAMP and lowers overhead).
    """
    x = make_pilot(N, M)
    Phi_full = build_otfs_operator(x, N, M)          # (L, L) shared operator
    L = N * M
    # reduced pilot overhead: observe Q of the L DD bins (fixed pattern)
    obs_rng = np.random.default_rng(_PILOT_SEED + 1)
    obs_idx = np.sort(obs_rng.choice(L, size=Q, replace=False)) if Q < L else np.arange(L)

    S = None
    if row_compress < 1.0:
        Qc = max(1, int(round(row_compress * Q)))
        S = build_row_compressor(Q, Qc)

    names = list(SCENARIOS.keys())
    clients = []
    for i in range(n_clients):
        sc = names[i % len(names)]
        clients.append(ClientDataset(sc, N, M, Phi_full, obs_idx, snr_db,
                                     samples_per_client, seed=1000 + i + 97 * seed,
                                     frames=frames, S=S))
    return clients
