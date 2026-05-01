"""
ECEN 432 - LAB 6: Automation of Pipelined ADC Calibration
=========================================================
4-stage pipelined ADC modelled in two architectures:

    * 2 bit /stage, no redundancy        -> 8-bit ENOB target
    * 2.5 bit/stage, 1-bit redundancy    -> 8-bit ENOB target

Non-idealities injected per stage:
    - Comparator offsets       sigma_OFF  (default 30 mV)
    - Residue-amplifier gain   sigma_G    (default 30 mV/V = 3 %)
    - 3rd-order residue NL     sigma_NL3  (bonus, default 30 mV/V)

Five calibration schemes operate on per-stage 'knobs' (offset trim, gain trim):

    1. ACDM      Adaptive Coordinate Descent with Momentum  (proposed)
    2. LMS       SPSA-style stochastic gradient ascent
    3. AI-PTS    Multi-Centroid AI-Loops + Parallel Thompson Sampling
    4. RL        Tabular Q-learning per knob
    5. Agentic   Pool of 4 stage-specialised agents + supervisor

The bonus implements an LMS-adapted per-stage 3rd-order canceller.

Outputs (PNG + JSON) are written to the parent directory of this script.

Run:
    python3 code/lab6_pipeline_adc.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 7
np.random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.dirname(HERE)              # one level up - lab root
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Global ADC + simulation parameters
# ---------------------------------------------------------------------------
VREF       = 1.0
N_STAGES   = 4
N_SAMPLES  = 4096                  # FFT length (power of 2)
SIG_BIN    = 117                   # prime, coprime with N_SAMPLES (coherent)
SIG_AMPL   = 0.9 * VREF            # near full-scale sinusoid
FS         = 1.0                   # normalised sampling rate
SIG_FREQ   = SIG_BIN * FS / N_SAMPLES

OFFSET_STD = 30e-3                 # 30 mV  comparator offset std
GAIN_STD   = 30e-3                 # 30 mV/V (= 3 %) residue-gain mismatch std
NL3_STD    = 30e-3                 # bonus 3rd-order coefficient std

MC_RUNS_BASELINE = 100             # part (ii)  Monte-Carlo runs
MC_RUNS_ALGO     = 20              # part (iii) Monte-Carlo runs (per algorithm)
N_ITERS          = 100             # calibration iterations


# ---------------------------------------------------------------------------
# 1.  PIPELINE STAGE
# ---------------------------------------------------------------------------
@dataclass
class StageConfig:
    """Static configuration for one pipelined stage (mode-dependent)."""
    thr_ideal:  np.ndarray     # comparator thresholds
    dac_ideal:  np.ndarray     # MDAC sub-DAC output levels (per code)
    centres:    np.ndarray     # ideal digital reconstruction centres
    gain:       float          # ideal residue gain
    n_codes:    int            # number of output codes


def _stage_config(mode: str) -> StageConfig:
    """Build the static configuration for either '2b' or '2.5b' stages."""
    if mode == "2b":
        thr  = np.array([-0.5, 0.0, 0.5]) * VREF
        dac  = np.array([-0.75, -0.25, 0.25, 0.75]) * VREF
        return StageConfig(thr, dac, dac.copy(), 4.0, 4)
    if mode == "2.5b":
        thr  = np.array([-0.625, -0.375, -0.125, 0.125, 0.375, 0.625]) * VREF
        dac  = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]) * VREF
        return StageConfig(thr, dac, dac.copy(), 4.0, 7)
    raise ValueError(f"unknown stage mode: {mode}")


class PipelineStage:
    """One pipelined ADC stage with injected non-idealities and trim knobs."""

    def __init__(self, mode: str = "2b",
                 offset_std: float = OFFSET_STD,
                 gain_std:   float = GAIN_STD,
                 nl3_std:    float = 0.0):
        self.mode = mode
        cfg = _stage_config(mode)
        self.cfg = cfg

        # Physical impairments (drawn once per stage)
        self.offset = np.random.normal(0.0, offset_std, size=cfg.thr_ideal.size)
        self.gain   = cfg.gain * (1.0 + np.random.normal(0.0, gain_std))
        self.nl3    = np.random.normal(0.0, nl3_std) if nl3_std > 0 else 0.0

        # Calibration knobs (algorithms tune these)
        self.knob_offset = np.zeros_like(self.offset)   # per comparator
        self.knob_gain   = 0.0                          # multiplicative trim
        self.knob_nl3    = 0.0                          # NL3 canceller coeff.

    # ---- Convenience -----------------------------------------------------
    @property
    def n_offset_knobs(self) -> int:
        return self.offset.size

    def process(self, v_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantise + amplify residue.  Returns (raw_code, v_residue)."""
        thr = self.cfg.thr_ideal + self.offset - self.knob_offset
        raw = np.sum(v_in[:, None] >= thr[None, :], axis=1)
        v_dac = self.cfg.dac_ideal[raw]
        gain  = self.gain * (1.0 - self.knob_gain)
        v_res = gain * (v_in - v_dac)
        # Optional 3rd-order non-linearity (and its background canceller)
        if self.nl3 != 0.0 or self.knob_nl3 != 0.0:
            v_res = v_res + (self.nl3 - self.knob_nl3) * v_res ** 3
        return raw, v_res


# ---------------------------------------------------------------------------
# 2.  PIPELINE ADC
# ---------------------------------------------------------------------------
class PipelineADC:
    """4-stage pipelined ADC with weighted-sum digital back-end."""

    def __init__(self, mode: str = "2b",
                 n_stages:   int   = N_STAGES,
                 offset_std: float = OFFSET_STD,
                 gain_std:   float = GAIN_STD,
                 nl3_std:    float = 0.0):
        self.mode      = mode
        self.n_stages  = n_stages
        # Surface the std actually used so calibration algorithms can scale
        # their internal step sizes to match the impairment magnitude.
        self.offset_std = offset_std
        self.gain_std   = gain_std
        self.stages = [PipelineStage(mode, offset_std, gain_std, nl3_std)
                       for _ in range(n_stages)]

        self.cfg     = self.stages[0].cfg
        self.weights = np.array([4.0 ** (-k) for k in range(n_stages)])

    def convert(self, v_in: np.ndarray) -> np.ndarray:
        """Reconstruct an analog estimate of v_in using ideal centres."""
        residue = np.asarray(v_in, dtype=float).copy()
        est     = np.zeros_like(residue)
        for k, stage in enumerate(self.stages):
            raw, residue = stage.process(residue)
            est += self.weights[k] * stage.cfg.centres[raw]
        return est


# ---------------------------------------------------------------------------
# 3.  METRICS  + INPUT GENERATION
# ---------------------------------------------------------------------------
def coherent_sin(n: int = N_SAMPLES, ampl: float = SIG_AMPL,
                 bin_idx: int = SIG_BIN) -> np.ndarray:
    """Coherent (closed-period) sinusoid sampled at the integer bin index."""
    t = np.arange(n)
    return ampl * np.sin(2.0 * np.pi * bin_idx * t / n)


def sndr_db(x_out: np.ndarray) -> float:
    """SNDR (dB) of x_out using a Hann window and a 7-bin signal cluster."""
    x = np.asarray(x_out, dtype=float)
    x = x - np.mean(x)
    N = x.size
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    P = np.abs(X) ** 2
    P[0] = 0.0                                  # remove residual DC
    k_sig = int(np.argmax(P))
    sig_mask = np.zeros_like(P, dtype=bool)
    sig_mask[max(0, k_sig - 3):k_sig + 4] = True
    sig_pwr = float(np.sum(P[sig_mask]))
    nd_pwr  = float(np.sum(P[~sig_mask]))
    if sig_pwr <= 0 or nd_pwr <= 0:
        return 0.0
    return 10.0 * np.log10(sig_pwr / nd_pwr)


# Re-use the same input across SNDR queries to avoid wasted RAM allocs
_PROBE_SIN = coherent_sin()


def run_sndr(adc: PipelineADC,
             v_in: np.ndarray | None = None) -> float:
    """SNDR of the ADC's reconstruction of a near-VFS sinusoid."""
    if v_in is None:
        v_in = _PROBE_SIN
    return sndr_db(adc.convert(v_in))


# ---------------------------------------------------------------------------
# 4.  KNOB-VECTOR HELPERS  (uniform algorithm interface)
# ---------------------------------------------------------------------------
def _n_knobs(adc: PipelineADC) -> int:
    return adc.n_stages * (adc.stages[0].n_offset_knobs + 1)


def _knob_vector(adc: PipelineADC) -> np.ndarray:
    """Pack all stage knobs into a single 1-D vector."""
    parts: list[np.ndarray] = []
    for s in adc.stages:
        parts.append(s.knob_offset)
        parts.append(np.array([s.knob_gain]))
    return np.concatenate(parts)


def _apply_knob_vector(adc: PipelineADC, v: np.ndarray) -> None:
    """Inverse of `_knob_vector`."""
    idx = 0
    for s in adc.stages:
        n = s.n_offset_knobs
        s.knob_offset = v[idx:idx + n].astype(float, copy=True)
        idx += n
        s.knob_gain = float(v[idx])
        idx += 1


def _knob_scales(adc: PipelineADC) -> np.ndarray:
    """Per-knob 'characteristic magnitude'.

    Used by every algorithm to scale its step sizes so the same code works
    for the 1x, 4x and intermediate impairment budgets.
    """
    scales: list[float] = []
    for s in adc.stages:
        scales.extend([adc.offset_std] * s.n_offset_knobs)
        scales.append(adc.gain_std)
    return np.array(scales, dtype=float)


# ===========================================================================
# 5.  CALIBRATION ALGORITHMS  (uniform signature: algo(adc, n_iters)->curve)
# ===========================================================================
#  All algorithms drive the *same* SNDR objective and write only the knobs.
#  Each returns an (n_iters,) curve recording the SNDR after each iteration.

# ---------------------------------------------------------------------------
# 5.1  ACDM - Adaptive Coordinate Descent with Momentum  [PROPOSED]
# ---------------------------------------------------------------------------
def calibrate_acdm(adc: PipelineADC, n_iters: int = N_ITERS) -> np.ndarray:
    """ACDM - my proposed scheme.

    For each knob, probe +-step.  If either side improves SNDR:
        - move there
        - grow that knob's step (momentum) up to a per-knob ceiling
    Otherwise:
        - stay put
        - shrink that knob's step toward a tiny floor (annealing)

    Round-robin across knobs.  Cost = 2 SNDR evals / iteration.
    """
    n      = _n_knobs(adc)
    scales = _knob_scales(adc)
    step   = 0.5 * scales.copy()             # initial step ~ 0.5 sigma
    step_max  = 4.0 * scales
    step_min  = 1e-4 * scales
    grow,    shrink = 1.6, 0.5

    sndr_curve = np.zeros(n_iters)
    best       = run_sndr(adc)

    for it in range(n_iters):
        k = it % n
        v0 = _knob_vector(adc)

        # Probe +step
        v_p = v0.copy(); v_p[k] += step[k]
        _apply_knob_vector(adc, v_p)
        s_p = run_sndr(adc)
        # Probe -step
        v_m = v0.copy(); v_m[k] -= step[k]
        _apply_knob_vector(adc, v_m)
        s_m = run_sndr(adc)

        if max(s_p, s_m) > best + 1e-3:
            if s_p >= s_m:
                _apply_knob_vector(adc, v_p)
                best = s_p
            else:
                _apply_knob_vector(adc, v_m)
                best = s_m
            step[k] = min(step[k] * grow, step_max[k])
        else:
            _apply_knob_vector(adc, v0)
            step[k] = max(step[k] * shrink, step_min[k])

        sndr_curve[it] = best
    return sndr_curve


# ---------------------------------------------------------------------------
# 5.2  LMS - SPSA-flavoured stochastic gradient ascent
# ---------------------------------------------------------------------------
def calibrate_lms(adc: PipelineADC, n_iters: int = N_ITERS) -> np.ndarray:
    """SPSA variant of LMS knob update.

    Each iteration:
      - draw a random Bernoulli +-1 perturbation `d` over all knobs
      - evaluate SNDR at +-c*d (2 calls)
      - estimate the directional derivative: g = (S+ - S-) / (2*c*d)
      - update knobs by a sign-LMS step in +g direction

    Independent of dimension (always 2 evals/iter), well-known calibration
    workhorse and a clean realisation of the LMS / gradient-descent prompt.
    """
    n         = _n_knobs(adc)
    scales    = _knob_scales(adc)
    c_perturb = 0.30 * scales         # SPSA probe magnitude
    mu        = 0.10                  # learning-rate (relative to scale)

    sndr_curve = np.zeros(n_iters)
    best = run_sndr(adc)

    for it in range(n_iters):
        v0    = _knob_vector(adc)
        delta = np.random.choice((-1.0, 1.0), size=n)
        # Probe +
        _apply_knob_vector(adc, v0 + c_perturb * delta)
        s_p = run_sndr(adc)
        # Probe -
        _apply_knob_vector(adc, v0 - c_perturb * delta)
        s_m = run_sndr(adc)
        # Two-sided SPSA gradient
        g = (s_p - s_m) / (2.0 * c_perturb * delta)
        # Sign-LMS scaled by knob magnitude
        step  = mu * scales * np.sign(g)
        v_new = v0 + step
        _apply_knob_vector(adc, v_new)
        cur = run_sndr(adc)
        # Reject obviously bad updates (rare with proper scaling)
        if cur < best - 5.0:
            _apply_knob_vector(adc, v0)
            cur = run_sndr(adc)
        best = max(best, cur)
        sndr_curve[it] = cur
    return sndr_curve


# ---------------------------------------------------------------------------
# 5.3  AI-Loops with Parallel Thompson Sampling
# ---------------------------------------------------------------------------
def calibrate_ai_pts(adc: PipelineADC, n_iters: int = N_ITERS,
                     M: int = 9, Ns: int = 3,
                     itrans: int = 20, gamma: float = 0.05) -> np.ndarray:
    """Multi-centroid AI-loops with parallel Thompson sampling.

    Centroids per knob span +-3*sigma scaled to that knob's magnitude.
    During the transient phase all M centroids are evaluated to seed the
    Beta posteriors; after that, only the top Ns candidates per Thompson
    sample are evaluated.

    Round-robin across knobs.  Each candidate uses a small "neural"
    nonlinear correction in addition to the centroid pull (keeping with
    the AI-loop description in the lab manual).
    """
    n      = _n_knobs(adc)
    scales = _knob_scales(adc)

    centroid_grid = np.linspace(-3.0, 3.0, M)               # in units of sigma
    alpha = np.ones((n, M))
    beta  = np.ones((n, M))
    sndr_curve = np.zeros(n_iters)

    # Two tiny linear "neural" weights per centroid (input = current knob, error)
    Wnn = np.random.normal(0.0, 0.2, size=(M, 2))           # bank of M arms

    for it in range(n_iters):
        eta = 1.0 if it < itrans else 1.0 / (gamma * (it - itrans + 1))
        eta = min(1.0, max(eta, 0.05))

        base_sndr = run_sndr(adc)
        sndr_curve[it] = base_sndr

        k  = it % n                              # round-robin coordinate
        v0 = _knob_vector(adc)
        # Choose subset (full sweep during transient)
        if it < itrans:
            subset = np.arange(M)
        else:
            samples = np.random.beta(alpha[k], beta[k])
            subset  = np.argsort(-samples)[:Ns]

        best_sndr   = base_sndr
        best_val    = v0[k]
        best_m      = None
        # Use SNDR shortfall as the "error" feature for the AI-loop NN
        ideal_proxy = 60.0
        err_feat    = ideal_proxy - base_sndr

        for m in subset:
            target  = centroid_grid[m] * scales[k]
            nn_corr = scales[k] * (Wnn[m, 0] * v0[k] / max(scales[k], 1e-12)
                                   + Wnn[m, 1] * np.tanh(err_feat / 10.0))
            cand = v0.copy()
            cand[k] = v0[k] + eta * (target - v0[k] + 0.05 * nn_corr)
            _apply_knob_vector(adc, cand)
            s = run_sndr(adc)
            if s > best_sndr:
                best_sndr, best_val, best_m = s, cand[k], m

        # Apply winner (might be unchanged)
        v_new = v0.copy(); v_new[k] = best_val
        _apply_knob_vector(adc, v_new)
        # Posterior update (Beta-Bernoulli)
        for m in subset:
            if m == best_m:
                alpha[k, m] += 1
            else:
                beta[k, m] += 1
        sndr_curve[it] = best_sndr
    return sndr_curve


# ---------------------------------------------------------------------------
# 5.4  Reinforcement Learning - tabular Q-learning per knob
# ---------------------------------------------------------------------------
def calibrate_rl(adc: PipelineADC, n_iters: int = N_ITERS,
                 eps0: float = 0.6, lr: float = 0.5, df: float = 0.9,
                 state_bins: int = 7) -> np.ndarray:
    """One independent tabular agent per knob.

    State  = quantised current knob value.
    Action = discrete additive step from a per-knob set scaled to sigma.
    Reward = delta-SNDR after taking the action.
    """
    n      = _n_knobs(adc)
    scales = _knob_scales(adc)
    A_unit = np.array([-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0])   # in units of sigma
    A      = A_unit.size
    Q      = np.zeros((n, state_bins, A))

    edges = np.linspace(-3.0, 3.0, state_bins + 1)         # in units of sigma

    def state_of(v: float, sc: float) -> int:
        return int(np.clip(np.digitize(v / max(sc, 1e-12), edges) - 1,
                           0, state_bins - 1))

    sndr_curve = np.zeros(n_iters)
    prev_sndr  = run_sndr(adc)
    best_sndr  = prev_sndr

    for it in range(n_iters):
        eps = max(0.05, eps0 * (1.0 - it / n_iters))
        v0  = _knob_vector(adc)
        k   = it % n
        s   = state_of(v0[k], scales[k])
        if np.random.rand() < eps:
            a = np.random.randint(A)
        else:
            a = int(np.argmax(Q[k, s]))

        v_new      = v0.copy()
        v_new[k]   = v0[k] + 0.4 * scales[k] * A_unit[a]    # 0.4 sigma per step
        _apply_knob_vector(adc, v_new)
        new_sndr   = run_sndr(adc)
        r          = new_sndr - prev_sndr
        s2         = state_of(v_new[k], scales[k])
        Q[k, s, a] = Q[k, s, a] + lr * (r + df * np.max(Q[k, s2]) - Q[k, s, a])
        prev_sndr  = new_sndr
        best_sndr  = max(best_sndr, new_sndr)
        sndr_curve[it] = best_sndr
    return sndr_curve


# ---------------------------------------------------------------------------
# 5.5  Agentic AI - 4 stage-specialised agents + supervisor
# ---------------------------------------------------------------------------
def calibrate_agentic(adc: PipelineADC, n_iters: int = N_ITERS) -> np.ndarray:
    """One agent per stage; supervisor routes activations.

    Each agent rotates through four roles:
        gradient    - finite-difference gradient ascent on its knobs
        centroid    - centroid-pull toward a learnt working point
        perturb     - small random kick to escape local plateaus
        consolidate - revert if the most recent action made things worse
    The supervisor tracks an EMA of each agent's recent SNDR contribution
    and selects the next-active agent via a softmax over those scores.
    """
    n_stages = adc.n_stages
    priority = np.ones(n_stages)
    sndr_curve = np.zeros(n_iters)
    best_sndr  = run_sndr(adc)
    sigma_off  = adc.offset_std
    sigma_gn   = adc.gain_std
    roles = ("gradient", "centroid", "perturb", "consolidate")

    for it in range(n_iters):
        # Supervisor: softmax over running priorities
        p = np.exp(priority - priority.max()); p /= p.sum()
        a_idx = np.random.choice(n_stages, p=p)
        role  = roles[(it // max(1, n_iters // 8)) % len(roles)]
        s     = adc.stages[a_idx]
        v0_off, v0_gain = s.knob_offset.copy(), s.knob_gain
        base = run_sndr(adc)

        if role == "gradient":
            eps = 0.2 * sigma_off
            g_off = np.zeros_like(s.knob_offset)
            for j in range(s.knob_offset.size):
                s.knob_offset[j] += eps
                g_off[j] = (run_sndr(adc) - base) / eps
                s.knob_offset[j] -= eps
            s.knob_gain += 0.2 * sigma_gn
            g_gain = (run_sndr(adc) - base) / (0.2 * sigma_gn)
            s.knob_gain -= 0.2 * sigma_gn
            norm = np.linalg.norm(np.append(g_off, g_gain)) + 1e-9
            s.knob_offset = v0_off + 0.3 * sigma_off * g_off / norm
            s.knob_gain   = v0_gain + 0.3 * sigma_gn  * g_gain / norm
        elif role == "centroid":
            # Pull toward a random anchor in +-2 sigma neighbourhood
            target_off  = np.random.uniform(-2 * sigma_off, 2 * sigma_off,
                                            size=s.knob_offset.size)
            target_gain = np.random.uniform(-2 * sigma_gn, 2 * sigma_gn)
            eta = 0.30
            s.knob_offset = v0_off  + eta * (target_off  - v0_off)
            s.knob_gain   = v0_gain + eta * (target_gain - v0_gain)
        elif role == "perturb":
            s.knob_offset = v0_off  + np.random.normal(0.0, 0.2 * sigma_off,
                                                       size=s.knob_offset.size)
            s.knob_gain   = v0_gain + np.random.normal(0.0, 0.2 * sigma_gn)
        # 'consolidate' takes no action this iteration.

        new = run_sndr(adc)
        if new < base - 0.5:                 # revert harmful actions
            s.knob_offset, s.knob_gain = v0_off, v0_gain
            new = base
        priority[a_idx] = 0.9 * priority[a_idx] + 0.1 * (new - base)
        best_sndr = max(best_sndr, new)
        sndr_curve[it] = best_sndr
    return sndr_curve


# ===========================================================================
# 6.  BONUS - 3rd-order residue NL canceller
# ===========================================================================
#  Model     : V_res = G*(V_in - V_dac) + a3 * (G*(V_in - V_dac))**3
#  Canceller : V_res' = V_res - knob_nl3 * V_res**3
#  Optimum   : knob_nl3 ~ a3   (exact to first order in a3)
# ---------------------------------------------------------------------------
def calibrate_nl3(adc: PipelineADC, n_iters: int = N_ITERS) -> np.ndarray:
    """Background adaptation of per-stage knob_nl3 (ACDM-style).

    For each stage's knob_nl3 we probe +-step and accept the better side.
    The step size grows after a successful move (momentum) and shrinks
    otherwise (line search).  This converges to knob_nl3 ~ a3 per stage,
    which to first order cancels the residue cubic.

    The fixed LMS rate originally tried here is unstable - the gradient
    of SNDR w.r.t. knob_nl3 has magnitude O(100 dB/unit), so any naive
    learning rate either crawls or overshoots.  The adaptive +/- probe
    automatically picks a stable step.
    """
    n_st = adc.n_stages
    step      = np.full(n_st, 0.20 * NL3_STD)        # ~6 mV initial step
    step_max  = 1.00 * NL3_STD                       # ~30 mV cap
    step_min  = 0.005 * NL3_STD                      # ~0.15 mV floor
    grow, shrink = 1.6, 0.5

    sndr_curve = np.zeros(n_iters)
    best = run_sndr(adc)
    for it in range(n_iters):
        for k_s, s in enumerate(adc.stages):
            base = run_sndr(adc)
            s.knob_nl3 += step[k_s]
            s_p = run_sndr(adc)
            s.knob_nl3 -= 2.0 * step[k_s]
            s_m = run_sndr(adc)
            s.knob_nl3 += step[k_s]              # restore baseline

            if max(s_p, s_m) > base + 1e-3:
                if s_p >= s_m:
                    s.knob_nl3 += step[k_s]
                    base = s_p
                else:
                    s.knob_nl3 -= step[k_s]
                    base = s_m
                step[k_s] = min(step[k_s] * grow, step_max)
            else:
                step[k_s] = max(step[k_s] * shrink, step_min)
            best = max(best, base)
        sndr_curve[it] = best
    return sndr_curve


# ===========================================================================
# 7.  EXPERIMENT DRIVERS
# ===========================================================================
ALGORITHMS = {
    "ACDM":    calibrate_acdm,
    "LMS":     calibrate_lms,
    "AI-PTS":  calibrate_ai_pts,
    "RL":      calibrate_rl,
    "Agentic": calibrate_agentic,
}
ALGO_COLORS = {
    "ACDM":    "tab:purple",
    "LMS":     "tab:blue",
    "AI-PTS":  "tab:orange",
    "RL":      "tab:green",
    "Agentic": "tab:red",
}


def fresh_adc(mode: str, off_std: float, gn_std: float,
              nl3_std: float = 0.0) -> PipelineADC:
    return PipelineADC(mode=mode, n_stages=N_STAGES,
                       offset_std=off_std, gain_std=gn_std, nl3_std=nl3_std)


def baseline_sweep(mode: str, off_std: float, gn_std: float,
                   runs: int = MC_RUNS_BASELINE) -> dict:
    """Per-experiment ideal/uncalibrated SNDR statistics."""
    ideal_sndrs, nocal_sndrs = [], []
    for _ in range(runs):
        ideal_sndrs.append(run_sndr(PipelineADC(mode=mode,
                                                offset_std=0.0, gain_std=0.0)))
        nocal_sndrs.append(run_sndr(fresh_adc(mode, off_std, gn_std)))
    return {
        "ideal_mean": float(np.mean(ideal_sndrs)),
        "ideal_std":  float(np.std(ideal_sndrs)),
        "nocal_mean": float(np.mean(nocal_sndrs)),
        "nocal_std":  float(np.std(nocal_sndrs)),
        "nocal_samples": np.asarray(nocal_sndrs, dtype=float),
    }


def algorithm_sweep(algo, mode: str, off_std: float, gn_std: float,
                    runs: int = MC_RUNS_ALGO,
                    iters: int = N_ITERS) -> tuple[np.ndarray, np.ndarray]:
    """Average SNDR-vs-iteration curve and 1-sigma envelope for one algorithm."""
    curves = np.zeros((runs, iters))
    for r in range(runs):
        # Re-seed per run for reproducibility independent of algorithm order
        np.random.seed(RNG_SEED + r * 13)
        adc = fresh_adc(mode, off_std, gn_std)
        np.random.seed(RNG_SEED + r * 13 + 1)
        curves[r] = algo(adc, n_iters=iters)
    return curves.mean(axis=0), curves.std(axis=0)


# ---------------------------------------------------------------------------
# 7.1  Plot helpers
# ---------------------------------------------------------------------------
def plot_ideal_psd(out_path: str) -> float:
    """Part (i.a): PSD of the ideal 4-stage 2b/stage ADC's output."""
    adc   = PipelineADC(mode="2b", offset_std=0.0, gain_std=0.0)
    v_in  = coherent_sin()
    v_out = adc.convert(v_in)
    sndr  = sndr_db(v_out)

    x = v_out - np.mean(v_out)
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    P = np.abs(X) ** 2
    P /= P.max()
    f = np.fft.rfftfreq(x.size, d=1.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(f, 10.0 * np.log10(P + 1e-20), color="tab:blue", lw=0.9)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-110, 5)
    ax.set_xlabel("Normalised frequency  (f / f_s)")
    ax.set_ylabel("Magnitude (dB, normalised)")
    ax.set_title(
        f"(i.a) Ideal 4-stage 2 b/stage pipeline ADC - "
        f"sinusoid at -{20*np.log10(VREF/SIG_AMPL):.1f} dBFS, "
        f"SNDR = {sndr:.2f} dB")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return sndr


def plot_mc_distribution(out_path: str,
                         baseline_2b: dict,
                         baseline_25b: dict | None = None) -> None:
    """Part (ii): histogram of SNDR over 100 Monte-Carlo runs."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bins = np.linspace(20, 60, 41)
    ax.hist(baseline_2b["nocal_samples"], bins=bins, alpha=0.65,
            label=f"2 b/stage  mu={baseline_2b['nocal_mean']:.2f} dB",
            color="tab:blue", edgecolor="white")
    if baseline_25b is not None:
        ax.hist(baseline_25b["nocal_samples"], bins=bins, alpha=0.55,
                label=f"2.5 b/stage  mu={baseline_25b['nocal_mean']:.2f} dB",
                color="tab:orange", edgecolor="white")
    ax.axvline(baseline_2b["ideal_mean"], color="black", ls="--", lw=1,
               label=f"Ideal 2 b/stage = {baseline_2b['ideal_mean']:.2f} dB")
    if baseline_25b is not None:
        ax.axvline(baseline_25b["ideal_mean"], color="gray", ls="--", lw=1,
                   label=f"Ideal 2.5 b/stage = {baseline_25b['ideal_mean']:.2f} dB")
    ax.set_xlabel("SNDR (dB)")
    ax.set_ylabel("Count (out of 100 MC runs)")
    ax.set_title("(ii) SNDR distribution with comparator offset (sigma=30 mV) "
                 "and gain mismatch (sigma=30 mV/V)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_iteration_curves(out_path: str,
                          experiments: list[tuple],
                          curves: dict,
                          baselines: dict,
                          suptitle: str) -> None:
    """Part (iii)+(iv): SNDR-vs-iteration for every algorithm / experiment."""
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (label, mode, o_std, g_std) in zip(axes, experiments):
        bl = baselines[label]
        ax.axhline(bl["ideal_mean"], color="black", ls="--", lw=1,
                   label=f"Ideal ({bl['ideal_mean']:.1f} dB)")
        ax.axhline(bl["nocal_mean"], color="gray",  ls=":",  lw=1,
                   label=f"No cal ({bl['nocal_mean']:.1f} dB)")
        for name in ALGORITHMS:
            mean = curves[label][name]["mean"]
            ax.plot(np.arange(len(mean)) + 1, mean,
                    color=ALGO_COLORS[name], lw=1.5, label=name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("SNDR (dB)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.2  Convergence-speed scoring
# ---------------------------------------------------------------------------
def iters_to_fraction(curve: np.ndarray, start: float, end: float,
                      frac: float) -> int:
    """First iteration index that reaches `start + frac*(end-start)`."""
    if end <= start:
        return -1
    thr = start + frac * (end - start)
    hit = np.where(curve >= thr)[0]
    return int(hit[0]) if hit.size else len(curve)


# ===========================================================================
# 8.  MAIN
# ===========================================================================
def main():
    t0 = time.time()
    print("ECEN 432 - Lab 6 :  Pipeline ADC Calibration Study")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Part (i.a) - ideal-ADC PSD plot
    # -------------------------------------------------------------------
    print("\n[part i.a] Ideal 2 b/stage SNDR for sinusoid near VFS")
    sndr_ia = plot_ideal_psd(os.path.join(OUT_DIR, "q_ia_ideal_psd.png"))
    print(f"  ideal SNDR (no errors)    : {sndr_ia:6.2f} dB")

    # -------------------------------------------------------------------
    # Part (ii) - Monte-Carlo SNDR with 30 mV/30 mV/V impairments
    # -------------------------------------------------------------------
    print("\n[part ii ] Monte-Carlo SNDR (offset 30 mV, gain 30 mV/V)")
    bl_2b   = baseline_sweep("2b",   OFFSET_STD,    GAIN_STD,    MC_RUNS_BASELINE)
    bl_25b  = baseline_sweep("2.5b", OFFSET_STD,    GAIN_STD,    MC_RUNS_BASELINE)
    bl_2b4  = baseline_sweep("2b",   4*OFFSET_STD,  4*GAIN_STD,  MC_RUNS_BASELINE)
    bl_25b4 = baseline_sweep("2.5b", 4*OFFSET_STD,  4*GAIN_STD,  MC_RUNS_BASELINE)

    print(f"  2 b/stage   ideal = {bl_2b['ideal_mean']:6.2f} dB   "
          f"nocal = {bl_2b['nocal_mean']:6.2f} +/- {bl_2b['nocal_std']:.2f} dB")
    print(f"  2.5 b/stage ideal = {bl_25b['ideal_mean']:6.2f} dB   "
          f"nocal = {bl_25b['nocal_mean']:6.2f} +/- {bl_25b['nocal_std']:.2f} dB")
    plot_mc_distribution(os.path.join(OUT_DIR, "q_ii_mc_distribution.png"),
                         bl_2b, bl_25b)

    # Quadrupled-std variants are needed only for the iteration plot in (iv)
    print(f"\n[part iv ] 4x std  (offset 120 mV, gain 120 mV/V)")
    print(f"  2 b/stage   nocal = {bl_2b4['nocal_mean']:6.2f} +/- "
          f"{bl_2b4['nocal_std']:.2f} dB")
    print(f"  2.5 b/stage nocal = {bl_25b4['nocal_mean']:6.2f} +/- "
          f"{bl_25b4['nocal_std']:.2f} dB")

    # -------------------------------------------------------------------
    # Part (iii) + (iv) - calibration runs
    # -------------------------------------------------------------------
    experiments = [
        ("2b, std=1x",   "2b",   OFFSET_STD,   GAIN_STD),
        ("2.5b, std=1x", "2.5b", OFFSET_STD,   GAIN_STD),
        ("2b, std=4x",   "2b",   4*OFFSET_STD, 4*GAIN_STD),
        ("2.5b, std=4x", "2.5b", 4*OFFSET_STD, 4*GAIN_STD),
    ]
    baselines = {
        "2b, std=1x":   bl_2b,
        "2.5b, std=1x": bl_25b,
        "2b, std=4x":   bl_2b4,
        "2.5b, std=4x": bl_25b4,
    }
    curves: dict = {label: {} for label, *_ in experiments}

    for label, mode, o_std, g_std in experiments:
        print(f"\n[part iii/iv] Calibration  -  {label}")
        for name, algo in ALGORITHMS.items():
            t_alg = time.time()
            mean, std = algorithm_sweep(algo, mode, o_std, g_std,
                                        runs=MC_RUNS_ALGO, iters=N_ITERS)
            curves[label][name] = {"mean": mean, "std": std}
            print(f"  {name:8s}  final = {mean[-1]:6.2f} dB  "
                  f"(t = {time.time() - t_alg:5.1f} s)")

    plot_iteration_curves(
        os.path.join(OUT_DIR, "q_iii_iv_curves.png"),
        experiments, curves, baselines,
        "Pipeline ADC calibration   -  SNDR vs Iteration")

    # Single-experiment plots for clarity in the report
    plot_iteration_curves(
        os.path.join(OUT_DIR, "q_iii_curves_2b_1x.png"),
        [experiments[0]], curves, baselines,
        "(iii) 2 b/stage,  sigma_off=30 mV, sigma_g=30 mV/V")
    plot_iteration_curves(
        os.path.join(OUT_DIR, "q_iv_curves_25b_1x.png"),
        [experiments[1]], curves, baselines,
        "(iv) 2.5 b/stage,  sigma_off=30 mV, sigma_g=30 mV/V")
    plot_iteration_curves(
        os.path.join(OUT_DIR, "q_iv_curves_2b_4x.png"),
        [experiments[2]], curves, baselines,
        "(iv) 2 b/stage,  sigma_off=120 mV, sigma_g=120 mV/V")
    plot_iteration_curves(
        os.path.join(OUT_DIR, "q_iv_curves_25b_4x.png"),
        [experiments[3]], curves, baselines,
        "(iv) 2.5 b/stage,  sigma_off=120 mV, sigma_g=120 mV/V")

    # -------------------------------------------------------------------
    # Bonus  -  3rd-order non-linearity cancellation
    # -------------------------------------------------------------------
    print("\n[bonus] 3rd-order residue non-linearity cancellation")
    np.random.seed(RNG_SEED + 999)
    adc_nl_off = PipelineADC(mode="2.5b", offset_std=0.0, gain_std=0.0,
                             nl3_std=NL3_STD)
    sndr_nl_before = run_sndr(adc_nl_off)
    bonus_curve    = calibrate_nl3(adc_nl_off, n_iters=N_ITERS)
    sndr_nl_after  = bonus_curve[-1]

    # Reference: ideal ADC without NL3 at all
    np.random.seed(RNG_SEED + 999)
    adc_nl_ref     = PipelineADC(mode="2.5b", offset_std=0.0, gain_std=0.0,
                                 nl3_std=0.0)
    sndr_nl_ideal  = run_sndr(adc_nl_ref)
    print(f"  ideal (no NL3)              : {sndr_nl_ideal:6.2f} dB")
    print(f"  with NL3 (no canceller)     : {sndr_nl_before:6.2f} dB")
    print(f"  with NL3 + canceller        : {sndr_nl_after :6.2f} dB")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.axhline(sndr_nl_ideal,  color="black",  ls="--", lw=1,
               label=f"Ideal (no NL3) = {sndr_nl_ideal:.2f} dB")
    ax.axhline(sndr_nl_before, color="gray",   ls=":",  lw=1,
               label=f"With NL3, no cal = {sndr_nl_before:.2f} dB")
    ax.plot(np.arange(N_ITERS) + 1, bonus_curve, color="tab:purple", lw=1.6,
            label="Background NL3 canceller")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SNDR (dB)")
    ax.set_title("Bonus: 3rd-order residue non-linearity (sigma_NL3 = 30 mV/V)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bonus_nl3.png"), dpi=130)
    plt.close(fig)

    # -------------------------------------------------------------------
    # JSON summary  -  feeds the report
    # -------------------------------------------------------------------
    summary = {
        "config": {
            "n_stages":   N_STAGES,
            "n_samples":  N_SAMPLES,
            "sig_bin":    SIG_BIN,
            "sig_ampl":   SIG_AMPL,
            "offset_std": OFFSET_STD,
            "gain_std":   GAIN_STD,
            "nl3_std":    NL3_STD,
            "mc_runs_baseline": MC_RUNS_BASELINE,
            "mc_runs_algo":     MC_RUNS_ALGO,
            "n_iters":    N_ITERS,
        },
        "ia_ideal_sndr_db": sndr_ia,
        "experiments": {},
        "bonus_nl3": {
            "sndr_ideal_db":  sndr_nl_ideal,
            "sndr_before_db": sndr_nl_before,
            "sndr_after_db":  sndr_nl_after,
        },
    }
    for label, mode, o_std, g_std in experiments:
        bl = baselines[label]
        per_algo = {}
        for name in ALGORITHMS:
            mean = curves[label][name]["mean"]
            per_algo[name] = {
                "final_sndr_db":  float(mean[-1]),
                "iter_to_50pct":  iters_to_fraction(mean, bl["nocal_mean"],
                                                    bl["ideal_mean"], 0.5),
                "iter_to_90pct":  iters_to_fraction(mean, bl["nocal_mean"],
                                                    bl["ideal_mean"], 0.9),
            }
        summary["experiments"][label] = {
            "mode":          mode,
            "offset_std":    o_std,
            "gain_std":      g_std,
            "ideal_sndr_db": bl["ideal_mean"],
            "nocal_sndr_db": bl["nocal_mean"],
            "nocal_std_db":  bl["nocal_std"],
            "algorithms":    per_algo,
        }

    with open(os.path.join(OUT_DIR, "lab6_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nElapsed: {time.time() - t0:.1f} s")
    print("Artifacts written to:")
    for p in sorted(os.listdir(OUT_DIR)):
        if p.startswith(("q_", "bonus_", "lab6_summary")):
            print(f"  {os.path.join(OUT_DIR, p)}")


if __name__ == "__main__":
    main()
