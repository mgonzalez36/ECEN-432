import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Global parameters
VFS = 1.0 # full-scale (normalised)
G = 4.0 # inter-stage gain (=2^N for N=2)

# Sub-ADC thresholds (centered input: x_c = Vin − VFS/2)
# 2.5-bit stage  : 6 comparators at midpoints of 7 DAC levels (DAC step = VFS/8)
THR_25 = np.array([-5,-3,-1, 1, 3, 5]) * VFS/16
# 2.0-bit stage  : 3 comparators, codes {-3,-1,+1,+3}, DAC step = VFS/4
THR_2 = np.array([-1, 0, 1]) * VFS/4
DAC_STEP = VFS/8 # unit weight in digital combiner

# Coherent-sampling single-tone (rectangular window → no spectral leakage)
N_FFT = 2**14 # 16 384 samples
M_CYC = 1201 # prime; gcd(M_CYC, N_FFT)=1 ⇒ coherent
FS    = 1.0
FIN   = M_CYC * FS / N_FFT
n_arr = np.arange(N_FFT)

A_IN  = 0.49 * VFS
DC_IN = VFS / 2
vin_tone = DC_IN + A_IN * np.sin(2*np.pi * FIN * n_arr / FS)

# Pre-computed level-correction term used for ENOB (IEEE-1241)
ENOB_LEVEL_DB = 20*np.log10(VFS / (2*A_IN)) # = +0.176 dB at A=0.49 VFS

# Sub-ADC and residue-amp models
def sub_adc(x_c, redund=True, offset=0.0, eps=0.0, a2=0.0, a3=0.0):
    """Effective comparator input (common-mode error model):
           x_eff = (1+eps)·x + offset + a2·x² + a3·x³
       then thresholded against the (redundant or non-redundant) ladder.
    """
    x_eff = (1+eps)*x_c + offset + a2*x_c**2 + a3*x_c**3
    if redund:                              # 2.5-bit (7 codes, ±3..±3)
        d = np.sum(x_eff[..., None] > THR_25, axis=-1) - 3
    else:                                   # 2.0-bit (4 codes, {-3,-1,+1,+3})
        d = 2*np.sum(x_eff[..., None] > THR_2, axis=-1) - 3
    return d.astype(np.int8)

def residue_amp(x_c, d, eps=0.0, Voff=0.0, a2=0.0, a3=0.0):
    """Residue-amp behavioural model (mirrors q3.py):
           Vres_c = (1+eps)·G·err + Voff + a2·err² + a3·err³
       with err = x_c − d·DAC_STEP.
    """
    err = x_c - d*DAC_STEP
    return (1+eps)*G*err + Voff + a2*err**2 + a3*err**3

# 3-stage pipeline (stage 3 is back-end flash, no MDAC)
def default_params():
    return [{'subadc': {}, 'ra': {}},
            {'subadc': {}, 'ra': {}},
            {'subadc': {}}]

def pipeline_adc(vin_u, params, redund=True):
    x1 = vin_u - VFS/2
    d1 = sub_adc(x1, redund, **params[0]['subadc'])
    r1 = residue_amp(x1, d1, **params[0]['ra'])

    d2 = sub_adc(r1, redund, **params[1]['subadc'])
    r2 = residue_amp(r1, d2, **params[1]['ra'])

    d3 = sub_adc(r2, redund, **params[2]['subadc'])
    # Weighted digital sum → centered estimate → back to unipolar
    x_est = DAC_STEP * (d1 + d2/G + d3/G**2)
    return x_est + VFS/2

# SNDR / ENOB from coherent-sampled FFT
def compute_metrics(vadc):
    y  = vadc - vadc.mean()
    Y  = np.fft.fft(y) / N_FFT
    Ps = np.abs(Y)**2
    Ps = Ps[:N_FFT//2].copy()
    Ps[1:] *= 2 # one-sided (Nyquist excluded by slicing)
    Ps[0] = 0.0 # DC explicitly excluded after mean removal
    Psig = Ps[M_CYC]
    Pnd  = Ps.sum() - Psig
    SNDR = 10*np.log10(Psig / max(Pnd, 1e-30))
    ENOB = (SNDR + ENOB_LEVEL_DB - 1.76) / 6.02
    return SNDR, ENOB, Ps

# B: Ramp
vin_ramp  = np.linspace(0, VFS, 4097)
vadc_ramp_redund = pipeline_adc(vin_ramp, default_params(), redund=True)

# Without redundancy + a moderate 1st-sub-ADC offset → missing codes
p_nr = default_params()
p_nr[0]['subadc'] = {'offset': 0.04 * VFS}    # < redundancy window had it existed
vadc_ramp_no_redund = pipeline_adc(vin_ramp, p_nr, redund=False)

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
axs[0].plot(vin_ramp/VFS, vadc_ramp_redund/VFS, lw=1)
axs[0].plot([0, 1], [0, 1], 'k--', alpha=.5, label='Ideal y=x')
axs[0].set_xlabel(r'$V_{in}/V_{FS}$')
axs[0].set_ylabel(r'$\hat V_{in}/V_{FS}$')
axs[0].set_title('B1: w/ redundancy (2.5-b/stage), ideal stages')
axs[0].grid(True); axs[0].legend()

axs[1].plot(vin_ramp/VFS, vadc_ramp_no_redund/VFS, lw=1, color='C3')
axs[1].plot([0, 1], [0, 1], 'k--', alpha=.5, label='Ideal y=x')
axs[1].set_xlabel(r'$V_{in}/V_{FS}$')
axs[1].set_ylabel(r'$\hat V_{in}/V_{FS}$')
axs[1].set_title('B2: w/o redundancy + 1st sub-ADC Voff = 0.04 VFS\n' '(missing codes & wide-bin DNL exposed)')
axs[1].grid(True); axs[1].legend()
plt.tight_layout()


# C: Single-tone PSD, SNDR, ENOB (ideal)
vadc_tone = pipeline_adc(vin_tone, default_params(), redund=True)
SNDR_i, ENOB_i, Ps_i = compute_metrics(vadc_tone)
freqs  = np.arange(N_FFT//2) * FS / N_FFT
psd_db = 10*np.log10(Ps_i / Ps_i[M_CYC] + 1e-30)

plt.figure(figsize=(8.5, 5))
plt.plot(freqs, psd_db, lw=.8)
plt.xlabel(r'$f/f_s$')
plt.ylabel('PSD (dBc)')
plt.title(f'C: Ideal pipeline | SNDR = {SNDR_i:.2f} dB, ENOB = {ENOB_i:.2f} b ')
plt.grid(True); plt.tight_layout()
print(f"C: Ideal pipeline at A_pk = {A_IN/VFS:.3f} VFS  "
      f"({20*np.log10(2*A_IN/VFS):+.3f} dBFS)")
print(f"    SNDR = {SNDR_i:5.2f} dB    ENOB(level-corr) = {ENOB_i:4.2f} b   "
      f"[7-bit theory ≈ 7.00 b]")


# Sweep helper (parts d & e)
def sweep_param(s_idx, blk, par, vals, redund):
    out = np.zeros_like(vals, dtype=float)
    for i, v in enumerate(vals):
        p = default_params()
        p[s_idx][blk] = {par: v}
        out[i], _, _ = compute_metrics(pipeline_adc(vin_tone, p, redund=redund))
    return out

SWEEPS = [
    ('1st sub-ADC $V_{off}$',  0, 'subadc', 'offset', np.linspace(-0.20, 0.20, 41)*VFS),
    ('1st sub-ADC $\\epsilon$', 0, 'subadc', 'eps',    np.linspace(-0.40, 0.40, 41)),
    ('1st sub-ADC $\\alpha_2$', 0, 'subadc', 'a2',     np.linspace(-1.0,  1.0,  41)/VFS**2),
    ('1st RA $\\epsilon$',      0, 'ra',     'eps',    np.linspace(-0.05, 0.05, 41)),
    ('1st RA $V_{off}$',        0, 'ra',     'Voff',   np.linspace(-0.50, 0.50, 41)*VFS),
    ('1st RA $\\alpha_3$',      0, 'ra',     'a3',     np.linspace(-50.0, 50.0, 41)/VFS**3),
]

# D: w/ redundancy
sndr_d = {}
fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, (nm, si, blk, par, vals) in zip(axs.flat, SWEEPS):
    s = sweep_param(si, blk, par, vals, redund=True)
    sndr_d[nm] = (vals, s)
    ax.plot(vals, s, 'o-', lw=1.4, ms=4)
    ax.axhline(SNDR_i, color='k', ls=':', lw=0.7, label=f'ideal {SNDR_i:.1f} dB')
    ax.set_xlabel(nm); ax.set_ylabel('SNDR (dB)')
    ax.grid(True); ax.legend(fontsize=8, loc='lower center')
fig.suptitle('D: SNDR vs single-parameter sweeps w/ redundancy (2.5-b/stage)',
             fontsize=12)
fig.tight_layout()


# E: w/ redundancy
fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, (nm, si, blk, par, vals) in zip(axs.flat, SWEEPS):
    s_e = sweep_param(si, blk, par, vals, redund=False)
    ax.plot(vals, sndr_d[nm][1], 'o-',  lw=1.4, ms=4, label='2.5-b (w/ redundancy)')
    ax.plot(vals, s_e,           's--', lw=1.4, ms=4, label='2.0-b (no redundancy)')
    ax.set_xlabel(nm); ax.set_ylabel('SNDR (dB)')
    ax.grid(True); ax.legend(fontsize=8, loc='lower center')
fig.suptitle('E: Redundancy comparison — sub-ADC errors are mitigated, '
             'residue-amp errors are NOT', fontsize=12)
fig.tight_layout()


# F: Convexity: 2-D landscape + calibration diagram
eps_grid = np.linspace(-0.05, 0.05, 41)
vof_grid = np.linspace(-0.10, 0.10, 41) * VFS
EPS, VOF = np.meshgrid(eps_grid, vof_grid)
SNDR_2D = np.empty_like(EPS)
for i in range(EPS.shape[0]):
    for j in range(EPS.shape[1]):
        p = default_params()
        p[0]['ra']     = {'eps':    EPS[i, j]}
        p[0]['subadc'] = {'offset': VOF[i, j]}
        SNDR_2D[i, j] = compute_metrics(pipeline_adc(vin_tone, p, redund=True))[0]

plt.figure(figsize=(7.5, 6))
pc = plt.pcolormesh(EPS, VOF/VFS, SNDR_2D, shading='auto', cmap='viridis')
plt.colorbar(pc, label='SNDR (dB)')
plt.contour(EPS, VOF/VFS, SNDR_2D, 12, colors='k', linewidths=0.4)
plt.xlabel(r'1st RA gain error $\epsilon_{RA1}$')
plt.ylabel(r'1st sub-ADC $V_{off,1}/V_{FS}$')
plt.title('F: SNDR landscape over two interacting parameters\n'
          '(Non-Convex)')
plt.tight_layout()

# 3-D surface view of the same SNDR landscape (z = SNDR in dB)
fig = plt.figure(figsize=(8.5, 6.5))
ax3d = fig.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(EPS, VOF/VFS, SNDR_2D, cmap='viridis',
                         edgecolor='k', linewidth=0.2, alpha=0.92,
                         rstride=1, cstride=1, antialiased=True)
ax3d.contour(EPS, VOF/VFS, SNDR_2D, 12, zdir='z',
             offset=SNDR_2D.min(), cmap='viridis', linewidths=0.6)
fig.colorbar(surf, ax=ax3d, shrink=0.6, pad=0.1, label='SNDR (dB)')
ax3d.set_xlabel(r'1st RA gain error $\epsilon_{RA1}$')
ax3d.set_ylabel(r'1st sub-ADC $V_{off,1}/V_{FS}$')
ax3d.set_zlabel('SNDR (dB)')
ax3d.set_title('F: 3-D SNDR landscape\n')
ax3d.view_init(elev=28, azim=-58)
plt.tight_layout()
plt.show()