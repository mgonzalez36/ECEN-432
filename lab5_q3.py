import numpy as np
import matplotlib.pyplot as plt

# Global parameters
VFS = 1.0 # full-scale voltage (normalized)
G = 4.0 # ideal residue-amp gain

# Coherent-sampling FFT parameters (no leakage -> no windowing)
N = 2**14 # # of samples (16384)
Mcycles = 1201 # integer, prime, coprime w/ N
fs = 1.0 # normalized sample rate
fin = Mcycles * fs / N # input tone frequency

# Single-tone: centered at VFS/2, peak-to-peak ~ 0.9*VFS (near VFS)
DC_IN = VFS / 2
A_IN = 0.45 * VFS
n = np.arange(N)
vin_tone = DC_IN + A_IN * np.sin(2*np.pi*fin*n/fs)

# A: Behavioral model: Vout = (1+eps)*G*Vin + Voff + a2*Vin^2 + a3*Vin^3
def residue_amp(vin, eps=0.0, Voff=0.0, a2=0.0, a3=0.0, G=G):
    return (1+eps)*G*vin + Voff + a2*vin**2 + a3*vin**3

# Setup table (eps, Voffset, a2, a3) ; a2,a3 scaled by VFS^2, VFS^3
SETUPS = {
    1: dict(eps=0.05, Voff=0.2*VFS, a2=0.00,            a3=0.00),
    2: dict(eps=0.05, Voff=0.3*VFS, a2=0.02/VFS**2,     a3=0.01/VFS**3),
    3: dict(eps=0.05, Voff=0.4*VFS, a2=0.05/VFS**2,     a3=0.03/VFS**3),
}

# Spectrum / metric helper
#   Returns single-sided power spectrum, SNDR, ENOB, SFDR, HD2, HD3
#   Uses coherent sampling -> rectangular window
def spectrum_metrics(vout):
    y  = vout - np.mean(vout)                 # remove DC
    Y  = np.fft.fft(y)/N
    P  = np.abs(Y)**2
    Ps = P[:N//2].copy(); Ps[1:] *= 2          # single-sided power

    # Harmonic bin (fold if aliased back to [0, N/2])
    def hbin(k):
        b = (k*Mcycles) % N
        return b if b <= N//2 else N-b

    sig_bin = Mcycles
    Psig    = Ps[sig_bin]
    HD2     = 10*np.log10(Ps[hbin(2)]/Psig + 1e-30)
    HD3     = 10*np.log10(Ps[hbin(3)]/Psig + 1e-30)

    # Noise+distortion = total spectrum minus the signal bin
    Pnd  = Ps.sum() - Psig
    SNDR = 10*np.log10(Psig/max(Pnd, 1e-30))
    ENOB = (SNDR - 1.76)/6.02

    # SFDR = signal / largest non-signal bin
    mask = np.ones_like(Ps, dtype=bool); mask[sig_bin] = False
    SFDR = 10*np.log10(Psig / Ps[mask].max())

    freqs  = np.arange(N//2)*fs/N
    psd_db = 10*np.log10(Ps/Psig + 1e-30)      # normalize to carrier (dBc)
    return dict(f=freqs, psd=psd_db, SNDR=SNDR, ENOB=ENOB, SFDR=SFDR,
                HD2=HD2, HD3=HD3, hb2=hbin(2), hb3=hbin(3))

# B: Ideal transfer function
vin_sweep   = np.linspace(0, VFS, 1001)
vout_ideal  = residue_amp(vin_sweep)   # pure G*Vin

plt.figure(figsize=(7,5))
plt.plot(vin_sweep/VFS, vout_ideal/VFS, lw=2, label=f'Ideal G={G:.0f}')
plt.xlabel(r'$V_{in}/V_{FS}$'); plt.ylabel(r'$V_{out}/V_{FS}$')
plt.title('B: Ideal Residue-Amp Transfer Function')
plt.grid(True); plt.legend(); plt.tight_layout()

# C: Ideal PSD / SNDR / ENOB
m_ideal = spectrum_metrics(residue_amp(vin_tone))
plt.figure(figsize=(8,5))
plt.plot(m_ideal['f'], m_ideal['psd'])
plt.xlabel(r'$f/f_s$'); plt.ylabel('PSD (dBc)')
plt.title(f"C: Ideal Spectrum | SNDR={m_ideal['SNDR']:.1f} dB, "
          f"ENOB={m_ideal['ENOB']:.1f} b")
plt.grid(True); plt.tight_layout()
print(f"[Ideal]  SNDR={m_ideal['SNDR']:.2f} dB   ENOB={m_ideal['ENOB']:.2f} b")

# E: Setup 1 transfer function vs ideal
vout_s1 = residue_amp(vin_sweep, **SETUPS[1])
plt.figure(figsize=(7,5))
plt.plot(vin_sweep/VFS, vout_ideal/VFS, 'k--', label='Ideal')
plt.plot(vin_sweep/VFS, vout_s1   /VFS, 'C1', lw=2, label='Setup 1')
plt.xlabel(r'$V_{in}/V_{FS}$'); plt.ylabel(r'$V_{out}/V_{FS}$')
plt.title(r'E: Setup 1: gain error $\epsilon$ + offset $V_{off}$')
plt.grid(True); plt.legend(); plt.tight_layout()

# F, G: Setup 2 & Setup 3 spectra with HD2/HD3 markers
for s_id in (2, 3):
    m = spectrum_metrics(residue_amp(vin_tone, **SETUPS[s_id]))
    plt.figure(figsize=(8,5))
    plt.plot(m['f'], m['psd'], lw=1)
    plt.plot(m['f'][m['hb2']], m['psd'][m['hb2']], 'r^', ms=10,
             label=f"HD2={m['HD2']:.1f} dBc")
    plt.plot(m['f'][m['hb3']], m['psd'][m['hb3']], 'gv', ms=10,
             label=f"HD3={m['HD3']:.1f} dBc")
    tag = 'f' if s_id == 2 else 'g'
    plt.xlabel(r'$f/f_s$'); plt.ylabel('PSD (dBc)')
    plt.title(f"({tag}) Setup {s_id} | SNDR={m['SNDR']:.1f} dB, "
              f"ENOB={m['ENOB']:.1f} b, SFDR={m['SFDR']:.1f} dBc")
    plt.grid(True); plt.legend(); plt.tight_layout()
    print(f"[Setup {s_id}] SNDR={m['SNDR']:6.2f} dB  ENOB={m['ENOB']:5.2f} b  "
          f"SFDR={m['SFDR']:6.2f} dBc  HD2={m['HD2']:6.2f}  HD3={m['HD3']:6.2f}")

# H: Parameter sweep: HD2 vs alpha_2
a2_sweep = np.linspace(0, 0.10/VFS**2, 21)
HD2_arr  = np.zeros_like(a2_sweep)
for i, a2 in enumerate(a2_sweep):
    m = spectrum_metrics(residue_amp(vin_tone, a2=a2))
    HD2_arr[i] = m['HD2']

# Analytical prediction (fundamental dominated by (1+eps)*G*A,
# HD2 amplitude from a2 term = a2*A^2/2 after removing DC component)
HD2_theory = 20*np.log10(np.maximum(a2_sweep*A_IN/(2*G), 1e-20))

plt.figure(figsize=(7,5))
plt.plot(a2_sweep*VFS**2, HD2_arr,   'o-', label='Simulated')
plt.plot(a2_sweep*VFS**2, HD2_theory,'k--',label=r'Theory: $20\log_{10}(\alpha_2 A/2G)$')
plt.xlabel(r'$\alpha_2 \cdot V_{FS}^2$'); plt.ylabel('HD2 (dBc)')
plt.title(r'H: HD2 vs $\alpha_2$ (linear growth in dB with $\log\alpha_2$)')
plt.grid(True); plt.legend(); plt.tight_layout()

plt.show()

# I: Parameter sweep: HD3 vs alpha_3  (all other params = 0)
a3_sweep = np.linspace(0, 0.10/VFS**2, 21)
HD3_arr  = np.zeros_like(a3_sweep)
for i, a3 in enumerate(a3_sweep):
    m = spectrum_metrics(residue_amp(vin_tone, a2=a2))
    HD3_arr[i] = m['HD3']

# Analytical prediction (fundamental dominated by (1+eps)*G*A,
# HD2 amplitude from a2 term = a2*A^2/2 after removing DC component)
HD3_theory = 20*np.log10(np.maximum(a3_sweep*A_IN/(2*G), 1e-20))

plt.figure(figsize=(7,5))
plt.plot(a3_sweep*VFS**2, HD2_arr,   'o-', label='Simulated')
plt.plot(a3_sweep*VFS**2, HD2_theory,'k--',label=r'Theory: $20\log_{10}(\alpha_3 A/2G)$')
plt.xlabel(r'$\alpha_3 \cdot V_{FS}^2$'); plt.ylabel('HD3 (dBc)')
plt.title(r'"I: HD3 vs $\alpha_3$ (linear growth in dB with $\log\alpha_3$)')
plt.grid(True); plt.legend(); plt.tight_layout()

plt.show()