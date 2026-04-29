import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

np.set_printoptions(precision=4, suppress=True)

VFS = 1.0

class FlashADC:
    """Flash ADC with per-comparator offset, gain, 2nd/3rd-order nonlinearity."""
    CONFIGS = {
        '2p5bit': dict(n_comp=6, n_bits=3),   # 2.5-bit w/ 1-bit redundancy
        '2bit':   dict(n_comp=3, n_bits=2),   # Conventional 2-bit
        '3bit':   dict(n_comp=7, n_bits=3),   # Conventional 3-bit
    }

    def __init__(self, arch='2p5bit', VFS=VFS):
        cfg = self.CONFIGS[arch]
        self.arch    = arch
        self.VFS     = VFS
        self.n_comp  = cfg['n_comp']
        self.n_bits  = cfg['n_bits']
        self.LSB     = VFS / (2 ** self.n_bits)
        self.Vref    = np.arange(1, self.n_comp + 1) * self.LSB
        self.n_codes = self.n_comp + 1
        self.os = np.zeros(self.n_comp)
        self.gn = np.zeros(self.n_comp)
        self.a2 = np.zeros(self.n_comp)
        self.a3 = np.zeros(self.n_comp)

    def randomize_errors(self, os_sig=0, gn_sig=0, a2_sig=0, a3_sig=0, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.os = rng.standard_normal(self.n_comp) * os_sig
        self.gn = rng.standard_normal(self.n_comp) * gn_sig
        self.a2 = rng.standard_normal(self.n_comp) * a2_sig
        self.a3 = rng.standard_normal(self.n_comp) * a3_sig

    def convert(self, Vin):
        Vin   = np.atleast_1d(Vin).astype(float)
        V     = Vin[:, None]
        V_eff = (1 + self.gn) * V + self.os + self.a2 * V**2 + self.a3 * V**3
        therm = (V_eff >= self.Vref).astype(int)
        code  = therm.sum(axis=1)
        return code, therm

    def reconstruct(self, code):
        """Mid-bin reconstruction (top bin uses [n_comp*LSB, VFS])."""
        code = np.asarray(code)
        Vrec = np.zeros_like(code, dtype=float)
        for k in range(self.n_codes):
            lo = k * self.LSB
            hi = (k + 1) * self.LSB if k < self.n_comp else self.VFS
            Vrec[code == k] = 0.5 * (lo + hi)
        return Vrec


# DNL / INL
def code_transitions(adc, Vin_sweep, codes):
    """Vin at which each k -> k+1 transition occurs."""
    T = np.full(adc.n_comp, np.nan)
    for k in range(adc.n_comp):
        idx = np.where((codes[:-1] < k + 1) & (codes[1:] >= k + 1))[0]
        if idx.size > 0:
            T[k] = 0.5 * (Vin_sweep[idx[0]] + Vin_sweep[idx[0] + 1])
    return T


def compute_dnl_inl(adc, Vin_sweep, codes):
    T       = code_transitions(adc, Vin_sweep, codes)
    T_ideal = np.arange(1, adc.n_comp + 1) * adc.LSB
    inl     = (T - T_ideal) / adc.LSB
    dnl     = np.diff(T) / adc.LSB - 1.0
    return dnl, inl, T


# SNDR / ENOB
def compute_sndr_enob(x, Fs, nbins_sig=3, nbins_dc=5):
    x = np.asarray(x)
    N = len(x)
    w = windows.blackmanharris(N)
    X = np.fft.rfft((x - x.mean()) * w) / (w.sum() / 2)
    psd       = np.abs(X) ** 2
    psd_dbfs  = 10 * np.log10(psd / (psd.max() + 1e-30) + 1e-30)
    freqs     = np.fft.rfftfreq(N, 1.0 / Fs)

    k_sig = np.argmax(psd[nbins_dc + 1:]) + nbins_dc + 1
    sig_mask = np.zeros_like(psd, dtype=bool)
    sig_mask[max(0, k_sig - nbins_sig): k_sig + nbins_sig + 1] = True
    dc_mask = np.zeros_like(psd, dtype=bool)
    dc_mask[:nbins_dc + 1] = True
    nd_mask = ~(sig_mask | dc_mask)

    P_sig    = psd[sig_mask].sum()
    P_nd     = psd[nd_mask].sum()
    sndr_db  = 10 * np.log10(P_sig / (P_nd + 1e-30))
    enob     = (sndr_db - 1.76) / 6.02
    return sndr_db, enob, freqs, psd_dbfs


# A: Design
adc = FlashADC('2p5bit')
print("Part A: 2.5-bit Flash ADC with 1-bit Interstage Redundancy")
print(f"Number of comparators: {adc.n_comp}")
print(f"Number of output codes: {adc.n_codes} (3-bit word; code 7 unused)")
print(f"LSB: VFS/2^3 = {adc.LSB*1e3:.3f} mV")
print(f"Reference levels (V): {adc.Vref}\n")

# B: Ideal transfer characteristic
Vin_sweep = np.linspace(0, VFS, 20001)
codes_ideal, _ = adc.convert(Vin_sweep)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.step(Vin_sweep, codes_ideal, where='post', color='C0', lw=1.5)
ax.set_xlabel('Vin [V]')
ax.set_ylabel('Digital Output Code')
ax.set_title('Part B: 2.5-bit Flash Transfer Characteristic')
ax.set_yticks(range(adc.n_codes))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# C: Quantization error
Vrec_ideal = adc.reconstruct(codes_ideal)
Qerr = Vin_sweep - Vrec_ideal

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(Vin_sweep, Qerr * 1e3, color='C3', lw=1)
ax.axhline(+adc.LSB / 2 * 1e3, color='k', ls='--', label=f'±LSB/2')
ax.axhline(-adc.LSB / 2 * 1e3, color='k', ls='--')
ax.set_xlabel('Vin [V]')
ax.set_ylabel('Quantization Error  Vin − Vrec  [mV]')
ax.set_title('Part C: Quantization Error vs Input')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

print(f"Part C: Q_err min = {Qerr.min()*1e3:+.2f} mV   "
      f"Q_err max = {Qerr.max()*1e3:+.2f} mV")
print("  (Top bin is 2·LSB wide → |Qerr| reaches LSB in [0.75, 1.0] V.)\n")


# D: 100 random-error realisations
os_sig = 8e-3 # σ comparator offset (≈ 0.064 LSB)
gn_sig = 5e-2 # σ fractional gain error (0.5 %)
a2_sig = 5e-3 # σ 2nd-order coefficient (1/V)
a3_sig = 5e-3 # σ 3rd-order coefficient (1/V^2)

rng_d = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(9, 4.5))
for _ in range(100):
    adc_d = FlashADC('2p5bit')
    adc_d.randomize_errors(os_sig, gn_sig, a2_sig, a3_sig, rng_d)
    codes_d, _ = adc_d.convert(Vin_sweep)
    ax.plot(Vin_sweep, codes_d, color='k', lw=0.4, alpha=0.15)
ax.plot(Vin_sweep, codes_ideal, color='r', lw=0.4, label='ideal')
ax.set_xlabel('Vin [V]')
ax.set_ylabel('Output code')
ax.set_title('Part D: 100 Random Error Realizations vs Ideal')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# E: DNL / INL with offset-only errors
rng_e = np.random.default_rng(7)
adc_e = FlashADC('2p5bit')
adc_e.randomize_errors(os_sig=os_sig, rng=rng_e)
codes_e, _ = adc_e.convert(Vin_sweep)
dnl, inl, T = compute_dnl_inl(adc_e, Vin_sweep, codes_e)

print("Part E: DNL / INL with comparator-offset-only errors "
      f"(σ_os = {os_sig*1e3:.1f} mV)")
print(f"Transition voltages: {T}")
print(f"DNL [LSB]: {dnl}")
print(f"INL [LSB]: {inl}")
print(f"|DNL|_pk = {np.max(np.abs(dnl)):.3f}   |INL|_pk = {np.max(np.abs(inl)):.3f}\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.stem(np.arange(len(dnl)), dnl, basefmt=' ')
ax1.axhline(0, color='k', lw=0.6)
ax1.set_xlabel('Transition index k')
ax1.set_ylabel('DNL [LSB]')
ax1.set_title('Part E: DNL (offset-only)')
ax1.grid(alpha=0.3)

ax2.stem(np.arange(1, len(inl) + 1), inl, basefmt=' ')
ax2.axhline(0, color='k', lw=0.6)
ax2.set_xlabel('Transition index k')
ax2.set_ylabel('INL [LSB]')
ax2.set_title('Part E: INL (offset-only)')
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# (f) Single-tone PSD / SNDR / ENOB (coherent sampling)
Fs = 1.0e6
N_fft = 16384
M = 1123                # prime, coprime with 2^14  -> coherent
f_in = M * Fs / N_fft      # ≈ 68.54 kHz
A_in = 0.45 * VFS
Vdc = VFS / 2
t = np.arange(N_fft) / Fs
Vin_sin = Vdc + A_in * np.sin(2 * np.pi * f_in * t)

adc_f = FlashADC('2p5bit')
codes_f, _ = adc_f.convert(Vin_sin)
Vrec_f = adc_f.reconstruct(codes_f)

sndr_db, enob, freqs, psd_dbfs = compute_sndr_enob(Vrec_f, Fs)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
nplot = 512
ax1.plot(t[:nplot] * 1e6, Vin_sin[:nplot], 'b-', lw=1.3, label='Vin')
ax1.plot(t[:nplot] * 1e6, Vrec_f[:nplot],  'r.', ms=4,   label='ADC output')
ax1.set_xlabel('Time [µs]')
ax1.set_ylabel('Voltage [V]')
ax1.set_title(f'Part F: TD Plot Fin = {f_in/1e3:.3f} kHz)')
ax1.grid(alpha=0.3)
ax1.legend()

ax2.plot(freqs / 1e3, psd_dbfs, color='k', lw=0.8)
ax2.set_xlabel('Frequency [kHz]')
ax2.set_ylabel('PSD [dBFS]')
ax2.set_title(f'Output PSD: SNDR = {sndr_db:.2f} dB, ENOB = {enob:.2f} b')
ax2.grid(alpha=0.3)
ax2.set_ylim(-120, 5)
plt.tight_layout()
plt.show()

print(f"Part F: SNDR = {sndr_db:.2f} dB, ENOB = {enob:.2f} bits "
      f"(ideal comparators; bounded by quantization + wide-top-bin).\n")

# G: Monte Carlo: 2.5-bit (w/ redundancy) vs 3-bit (conventional)
n_mc     = 50
sndr_A   = np.zeros(n_mc);  enob_A  = np.zeros(n_mc)
sndr_B   = np.zeros(n_mc);  enob_B  = np.zeros(n_mc)
dnlpk_A  = np.zeros(n_mc);  inlpk_A = np.zeros(n_mc)
dnlpk_B  = np.zeros(n_mc);  inlpk_B = np.zeros(n_mc)

master_rng = np.random.default_rng(2024)
for i in range(n_mc):
    seed_i = int(master_rng.integers(0, 2**32 - 1))
    rng_A  = np.random.default_rng(seed_i)
    rng_B  = np.random.default_rng(seed_i + 999983)

    adc_A = FlashADC('2p5bit')
    adc_B = FlashADC('3bit')
    adc_A.randomize_errors(os_sig, gn_sig, a2_sig, a3_sig, rng_A)
    adc_B.randomize_errors(os_sig, gn_sig, a2_sig, a3_sig, rng_B)

    # DNL / INL from DC sweep
    cA_sw, _ = adc_A.convert(Vin_sweep)
    cB_sw, _ = adc_B.convert(Vin_sweep)
    dA, iA, _ = compute_dnl_inl(adc_A, Vin_sweep, cA_sw)
    dB, iB, _ = compute_dnl_inl(adc_B, Vin_sweep, cB_sw)
    dnlpk_A[i] = np.nanmax(np.abs(dA)); inlpk_A[i] = np.nanmax(np.abs(iA))
    dnlpk_B[i] = np.nanmax(np.abs(dB)); inlpk_B[i] = np.nanmax(np.abs(iB))

    # SNDR / ENOB from single tone
    cA_sn, _ = adc_A.convert(Vin_sin)
    cB_sn, _ = adc_B.convert(Vin_sin)
    VrA = adc_A.reconstruct(cA_sn)
    VrB = adc_B.reconstruct(cB_sn)
    sndr_A[i], enob_A[i], _, _ = compute_sndr_enob(VrA, Fs)
    sndr_B[i], enob_B[i], _, _ = compute_sndr_enob(VrB, Fs)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
b = 14
axes[0,0].hist(sndr_A, bins=b, alpha=0.6, color='C0', edgecolor='k',
               label='2.5-bit (w/ redund.)')
axes[0,0].hist(sndr_B, bins=b, alpha=0.6, color='C3', edgecolor='k',
               label='3-bit conventional')
axes[0,0].set_xlabel('SNDR [dB]'); axes[0,0].set_ylabel('Count')
axes[0,0].set_title('SNDR'); axes[0,0].grid(alpha=0.3); axes[0,0].legend()

axes[0,1].hist(enob_A, bins=b, alpha=0.6, color='C0', edgecolor='k', label='2.5-bit')
axes[0,1].hist(enob_B, bins=b, alpha=0.6, color='C3', edgecolor='k', label='3-bit')
axes[0,1].set_xlabel('ENOB [bits]'); axes[0,1].set_ylabel('Count')
axes[0,1].set_title('ENOB'); axes[0,1].grid(alpha=0.3); axes[0,1].legend()

axes[1,0].hist(dnlpk_A, bins=b, alpha=0.6, color='C0', edgecolor='k', label='2.5-bit')
axes[1,0].hist(dnlpk_B, bins=b, alpha=0.6, color='C3', edgecolor='k', label='3-bit')
axes[1,0].set_xlabel('|DNL|_pk [LSB]'); axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Peak DNL'); axes[1,0].grid(alpha=0.3); axes[1,0].legend()

axes[1,1].hist(inlpk_A, bins=b, alpha=0.6, color='C0', edgecolor='k', label='2.5-bit')
axes[1,1].hist(inlpk_B, bins=b, alpha=0.6, color='C3', edgecolor='k', label='3-bit')
axes[1,1].set_xlabel('|INL|_pk [LSB]'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Peak INL'); axes[1,1].grid(alpha=0.3); axes[1,1].legend()

plt.suptitle(f'Part G: Monte Carlo ({n_mc} runs)', y=1.00)
plt.tight_layout()
plt.show()

print("Part G: Monte Carlo summary")
print(f"  2.5-bit : SNDR = {sndr_A.mean():5.2f} ± {sndr_A.std():4.2f} dB  |  "
      f"ENOB = {enob_A.mean():4.2f} ± {enob_A.std():4.2f} b")
print(f"  3-bit   : SNDR = {sndr_B.mean():5.2f} ± {sndr_B.std():4.2f} dB  |  "
      f"ENOB = {enob_B.mean():4.2f} ± {enob_B.std():4.2f} b")
print(f"  2.5-bit : |DNL|_pk = {dnlpk_A.mean():4.3f} ± {dnlpk_A.std():4.3f} LSB  |  "
      f"|INL|_pk = {inlpk_A.mean():4.3f} ± {inlpk_A.std():4.3f} LSB")
print(f"  3-bit   : |DNL|_pk = {dnlpk_B.mean():4.3f} ± {dnlpk_B.std():4.3f} LSB  |  "
      f"|INL|_pk = {inlpk_B.mean():4.3f} ± {inlpk_B.std():4.3f} LSB")