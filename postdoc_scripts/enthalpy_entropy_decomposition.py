#!/usr/bin/env python
"""
enthalpy_entropy_decomposition.py

Compute ΔH‡ and TΔS‡ from AIMD trajectories across all windows.
Plots mean potential energy vs reaction coordinate and extracts
enthalpy/entropy contributions using IS/TS from PMF analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import PchipInterpolator

# Plot settings
LABEL_FONTSIZE = 18
TICK_LABELSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1


def read_energies_from_oszicar(file_path):
    """Extract E0 (potential energy) from OSZICAR."""
    energies = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'E0=' in line:
                    parts = line.split()
                    e0_index = next((i for i, part in enumerate(parts) if 'E0=' in part), None)
                    if e0_index is not None:
                        energy_str = parts[e0_index + 1]
                        try:
                            energy = float(energy_str)
                            energies.append(energy)
                        except ValueError:
                            pass
    except FileNotFoundError:
        pass
    return energies


def read_energies_from_window(window_dir):
    """Read and concatenate energies from all seg*/OSZICAR files in a window directory."""
    all_energies = []
    seg_folders = sorted(glob(f'{window_dir}/seg*/OSZICAR'))

    for oszicar_path in seg_folders:
        energies = read_energies_from_oszicar(oszicar_path)
        all_energies.extend(energies)

    return np.array(all_energies)


def get_cv_from_force_stats(window_dir):
    """Read the CV (constrained bond length) from force_stats_report.txt."""
    file_path = os.path.join(window_dir, 'force_stats_report.txt')
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'CV:' in line:
                    return float(line.split('CV:')[1].strip())
    except FileNotFoundError:
        pass
    return None


def read_pmf_results():
    """Read IS, TS, FS positions and ΔG from pmf_analysis_results.txt."""
    results = {}
    try:
        with open('pmf_analysis_results.txt', 'r') as f:
            for line in f:
                if 'Initial State:' in line:
                    results['IS'] = float(line.split(':')[1].strip().replace('Å', ''))
                elif 'Transition State:' in line:
                    results['TS'] = float(line.split(':')[1].strip().replace('Å', ''))
                elif 'Final State:' in line:
                    results['FS'] = float(line.split(':')[1].strip().replace('Å', ''))
                elif 'Forward Barrier:' in line:
                    parts = line.split(':')[1].strip().split('±')
                    results['delta_G'] = float(parts[0].strip().replace('eV', ''))
                    results['delta_G_err'] = float(parts[1].strip().replace('eV', ''))
    except FileNotFoundError:
        print("Warning: pmf_analysis_results.txt not found. Run pmf_analysis.py first.")
    return results


def compute_mean_energy(window_dir, discard_ps=2.0, timestep_fs=1.0):
    """Extract mean potential energy after discarding equilibration."""
    energies = read_energies_from_window(window_dir)
    if len(energies) == 0:
        return np.nan, np.nan, 0

    discard_steps = int(discard_ps * 1000 / timestep_fs)
    prod_energies = energies[discard_steps:]

    if len(prod_energies) == 0:
        return np.nan, np.nan, 0

    mean_E = np.mean(prod_energies)
    sem_E = np.std(prod_energies) / np.sqrt(len(prod_energies))
    return mean_E, sem_E, len(prod_energies)


def process_all_windows(discard_ps=2.0, timestep_fs=1.0):
    """Process all window directories and return data sorted by CV."""
    data = {
        'cv': [],
        'mean_E': [],
        'sem_E': [],
        'n_steps': [],
        'folder': []
    }

    folders = sorted(glob("[0-9].[0-9][0-9]_*"))
    if len(folders) == 0:
        print("No window directories found matching pattern [0-9].[0-9][0-9]_*")
        return None

    for folder in folders:
        cv = get_cv_from_force_stats(folder)
        if cv is None:
            # Try to extract CV from folder name
            try:
                cv = float(folder.split('_')[0])
            except ValueError:
                print(f"Could not determine CV for {folder}, skipping")
                continue

        mean_E, sem_E, n_steps = compute_mean_energy(folder, discard_ps, timestep_fs)

        if not np.isnan(mean_E):
            data['cv'].append(cv)
            data['mean_E'].append(mean_E)
            data['sem_E'].append(sem_E)
            data['n_steps'].append(n_steps)
            data['folder'].append(folder)

    # Sort by CV
    sorted_indices = np.argsort(data['cv'])
    for key in data:
        data[key] = [data[key][i] for i in sorted_indices]

    return data


def interpolate_energy_at_cv(data, cv_target):
    """Interpolate mean energy at a specific CV value using PCHIP."""
    cv = np.array(data['cv'])
    mean_E = np.array(data['mean_E'])
    sem_E = np.array(data['sem_E'])

    # Use PCHIP interpolation for smooth curve
    spline = PchipInterpolator(cv, mean_E)
    spline_sem = PchipInterpolator(cv, sem_E)

    E_interp = spline(cv_target)
    sem_interp = spline_sem(cv_target)

    return E_interp, sem_interp


def plot_enthalpy_profile(data, pmf_results, delta_H, delta_H_err):
    """Plot mean potential energy vs reaction coordinate."""
    cv = np.array(data['cv'])
    mean_E = np.array(data['mean_E'])
    sem_E = np.array(data['sem_E'])

    # Shift energies so IS is at zero
    E_IS, _ = interpolate_energy_at_cv(data, pmf_results['IS'])
    mean_E_shifted = mean_E - E_IS

    plt.figure(figsize=(10, 6))

    # Plot data points with error bars
    plt.errorbar(cv, mean_E_shifted, yerr=sem_E, fmt='o', color='black',
                 ecolor='black', capsize=3.5, label='⟨E_pot⟩')

    # Plot PCHIP interpolation
    spline = PchipInterpolator(cv, mean_E_shifted)
    cv_fine = np.linspace(cv.min(), cv.max(), 200)
    plt.plot(cv_fine, spline(cv_fine), 'k-', alpha=0.7)

    # Mark IS and TS
    if 'IS' in pmf_results:
        E_IS_shifted = 0.0  # By definition after shifting
        plt.axvline(pmf_results['IS'], color='blue', linestyle='--', alpha=0.5, label=f"IS ({pmf_results['IS']:.2f} Å)")
        plt.scatter([pmf_results['IS']], [E_IS_shifted], marker='s', s=100, color='blue', zorder=5)

    if 'TS' in pmf_results:
        E_TS_shifted = spline(pmf_results['TS'])
        plt.axvline(pmf_results['TS'], color='red', linestyle='--', alpha=0.5, label=f"TS ({pmf_results['TS']:.2f} Å)")
        plt.scatter([pmf_results['TS']], [E_TS_shifted], marker='s', s=100, color='red', zorder=5)

        # Annotate ΔH‡
        plt.annotate(f'ΔH‡ = {delta_H:.3f} ± {delta_H_err:.3f} eV',
                     xy=(pmf_results['TS'], E_TS_shifted),
                     xytext=(30, -20), textcoords='offset points',
                     fontsize=12, bbox=dict(boxstyle='round', fc='white', ec='gray'))

    plt.xlabel("Reaction Coordinate (Å)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("⟨E_pot⟩ - ⟨E_pot⟩_IS (eV)", fontsize=LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE,
                    length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('enthalpy_profile.png', dpi=300, bbox_inches='tight')
    print("Saved: enthalpy_profile.png")


def main():
    # === USER INPUT ===
    discard_ps = 9.0    # Equilibration time to discard (ps)
    timestep_fs = 1.0   # MD timestep (fs)
    delta_G_neb = 0.85  # eV, NEB+hTST barrier for comparison
    # === END USER INPUT ===

    print("Processing all window directories...")
    data = process_all_windows(discard_ps, timestep_fs)

    if data is None or len(data['cv']) == 0:
        print("No data found. Exiting.")
        return

    # Print data table
    print(f"\n{'CV (Å)':<10} {'⟨E_pot⟩ (eV)':<18} {'SEM (eV)':<12} {'Steps':<10} {'Folder'}")
    print("-" * 70)
    for i in range(len(data['cv'])):
        print(f"{data['cv'][i]:<10.2f} {data['mean_E'][i]:<18.4f} {data['sem_E'][i]:<12.4f} {data['n_steps'][i]:<10} {data['folder'][i]}")

    # Read PMF results
    pmf_results = read_pmf_results()

    if 'IS' not in pmf_results or 'TS' not in pmf_results:
        print("\nError: Could not find IS/TS from pmf_analysis_results.txt")
        print("Run pmf_analysis.py first to generate the PMF results.")
        return

    # Interpolate energies at IS and TS
    E_IS, err_IS = interpolate_energy_at_cv(data, pmf_results['IS'])
    E_TS, err_TS = interpolate_energy_at_cv(data, pmf_results['TS'])

    delta_H = E_TS - E_IS
    delta_H_err = np.sqrt(err_IS**2 + err_TS**2)

    delta_G = pmf_results.get('delta_G', np.nan)
    delta_G_err = pmf_results.get('delta_G_err', 0)

    T_delta_S = delta_H - delta_G
    T_delta_S_err = np.sqrt(delta_H_err**2 + delta_G_err**2)

    # Print results
    print("\n" + "=" * 50)
    print("ENTHALPY-ENTROPY DECOMPOSITION")
    print("=" * 50)
    print(f"IS position:   {pmf_results['IS']:.2f} Å")
    print(f"TS position:   {pmf_results['TS']:.2f} Å")
    print(f"\n⟨E_pot⟩_IS:    {E_IS:.4f} ± {err_IS:.4f} eV")
    print(f"⟨E_pot⟩_TS:    {E_TS:.4f} ± {err_TS:.4f} eV")
    print(f"\nΔH‡:           {delta_H:.3f} ± {delta_H_err:.3f} eV")
    print(f"ΔG‡ (AIMD):    {delta_G:.3f} ± {delta_G_err:.3f} eV")
    print(f"TΔS‡:          {T_delta_S:.3f} ± {T_delta_S_err:.3f} eV")

    print("\n" + "-" * 50)
    print("INTERPRETATION")
    print("-" * 50)
    if T_delta_S > 0:
        print(f"TΔS‡ > 0 → Entropy INCREASES at TS (more disordered)")
        print("         → Entropic contribution LOWERS the barrier")
    else:
        print(f"TΔS‡ < 0 → Entropy DECREASES at TS (more ordered)")
        print("         → Entropic contribution RAISES the barrier")

    print(f"\nComparison with NEB barrier ({delta_G_neb:.2f} eV):")
    if abs(delta_H - delta_G_neb) < 0.15:
        print(f"  ΔH‡ ≈ NEB barrier → Anharmonic correction is primarily ENTROPIC")
    else:
        diff = delta_H - delta_G_neb
        print(f"  ΔH‡ - NEB = {diff:.3f} eV → Mixed enthalpic/entropic contributions")

    # Plot
    plot_enthalpy_profile(data, pmf_results, delta_H, delta_H_err)

    # Save results to file
    with open('enthalpy_entropy_results.txt', 'w') as f:
        f.write("ENTHALPY-ENTROPY DECOMPOSITION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Equilibration discarded: {discard_ps} ps\n")
        f.write(f"MD timestep: {timestep_fs} fs\n\n")
        f.write(f"IS position:   {pmf_results['IS']:.2f} Å\n")
        f.write(f"TS position:   {pmf_results['TS']:.2f} Å\n\n")
        f.write(f"⟨E_pot⟩_IS:    {E_IS:.4f} ± {err_IS:.4f} eV\n")
        f.write(f"⟨E_pot⟩_TS:    {E_TS:.4f} ± {err_TS:.4f} eV\n\n")
        f.write(f"ΔH‡:           {delta_H:.3f} ± {delta_H_err:.3f} eV\n")
        f.write(f"ΔG‡ (AIMD):    {delta_G:.3f} ± {delta_G_err:.3f} eV\n")
        f.write(f"TΔS‡:          {T_delta_S:.3f} ± {T_delta_S_err:.3f} eV\n")
    print("\nSaved: enthalpy_entropy_results.txt")


if __name__ == "__main__":
    main()
