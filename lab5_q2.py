import numpy as np
import matplotlib.pyplot as plt

# Part A: Design and Implement Ideal MDAC
class IdealMDAC_2_5Bit:
    def __init__(self, v_ref=1.0):
        self.v_ref = v_ref
        self.gain = 4.0
        
        # The 7 ideal DAC levels for a 2.5-bit stage (Codes 0 through 6)
        # Corresponding to -6/8, -4/8, -2/8, 0, 2/8, 4/8, 6/8 of Vref
        self.dac_levels = np.array([
            -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75
        ]) * self.v_ref

    def calculate_residue(self, v_in, code):
        """
        Calculates the MDAC residue output.
        V_res = Gain * (V_in - V_dac)
        """
        v_dac = self.dac_levels[code]
        v_res = self.gain * (v_in - v_dac)
        return v_res

# Part B: Plotting and Linearity Verification
def run_mdac_simulation():
    vfs = 1.0
    mdac = IdealMDAC_2_5Bit(v_ref=vfs)
    
    # 1. Verify Linearity Across All Codes (Independent of ADC thresholds)
    # We will plot the MDAC output for each code over a relevant small voltage range
    # to show that the gain (slope) is perfectly linear and constant.
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'magenta']
    
    print("--- MDAC Linearity Verification ---")
    print(f"{'Code':<6} | {'V_DAC (V)':<10} | {'Calculated Gain (Slope)':<25}")
    print("-" * 45)
    
    for code in range(7):
        # Create a small input sweep centered around the DAC level for this code
        v_dac = mdac.dac_levels[code]
        v_sweep_local = np.linspace(v_dac - 0.25, v_dac + 0.25, 100)
        
        # Calculate residue
        v_res_local = mdac.calculate_residue(v_sweep_local, code)
        
        # Verify linearity by calculating the slope (dV_out / dV_in)
        slope = (v_res_local[-1] - v_res_local[0]) / (v_sweep_local[-1] - v_sweep_local[0])
        print(f"{code:<6} | {v_dac:<10.3f} | {slope:<25.2f}")
        
        # Plot individual code transfer curves
        ax1.plot(v_sweep_local, v_res_local, color=colors[code], label=f'Code {code}')

    ax1.set_title('MDAC Linearity: Output per Digital Code', fontsize=12)
    ax1.set_xlabel('Analog Input Voltage (V)', fontsize=10)
    ax1.set_ylabel('MDAC Residue Output (V)', fontsize=10)
    ax1.axhline(vfs, color='black', linestyle='--')
    ax1.axhline(-vfs, color='black', linestyle='--')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # 2. Reconstruct the Full Stage Transfer Characteristic (The "Sawtooth")
    # This requires applying the ideal Sub-ADC thresholds to switch codes correctly.
    thresholds = np.array([-0.625, -0.375, -0.125, 0.125, 0.375, 0.625]) * vfs
    v_sweep_full = np.linspace(-vfs, vfs, 1000)
    full_residue = []
    
    for v in v_sweep_full:
        # Resolve ideal digital code (thermometer sum)
        code = np.sum(v >= thresholds)
        # Pass to MDAC
        res = mdac.calculate_residue(v, code)
        full_residue.append(res)
        
    ax2.plot(v_sweep_full, full_residue, color='black', lw=2)
    
    # Highlight the valid sub-ranges
    for th in thresholds:
        ax2.axvline(th, color='red', linestyle=':', alpha=0.5)
        
    ax2.set_title('Transfer Characteristic', fontsize=12)
    ax2.set_xlabel('Input Voltage (V)', fontsize=10)
    ax2.set_ylabel('Residue Output (V)', fontsize=10)
    ax2.axhline(vfs*0.5, color='blue', linestyle='--', label='+/- 0.5V (Ideal Peak)')
    ax2.axhline(-vfs*0.5, color='blue', linestyle='--')
    ax2.axhline(vfs, color='black', linestyle='-', alpha=0.3, label='+/- 1.0V (Saturation Limits)')
    ax2.axhline(-vfs, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_mdac_simulation()