#!/usr/bin/env python
"""
enthalpy_entropy_decomposition.py

Compute ΔH‡ and TΔS‡ from AIMD trajectories.
"""

import numpy as np
from glob import glob


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
                            print(f"Warning: Non-numeric energy value in {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return energies


def read_energies_from_window(window_dir):
    """Read and concatenate energies from all seg*/OSZICAR files in a window directory."""
    all_energies = []
    seg_folders = sorted(glob(f'{window_dir}/seg*/OSZICAR'))

    if len(seg_folders) == 0:
        print(f"No seg*/OSZICAR files found in {window_dir}")
        return np.array([])

    print(f"  Found {len(seg_folders)} segments in {window_dir}")
    for oszicar_path in seg_folders:
        energies = read_energies_from_oszicar(oszicar_path)
        all_energies.extend(energies)

    return np.array(all_energies)


def compute_mean_energy(window_dir, discard_ps=2.0, timestep_fs=1.0):
    """Extract mean potential energy after discarding equilibration."""
    energies = read_energies_from_window(window_dir)
    if len(energies) == 0:
        return np.nan, np.nan

    discard_steps = int(discard_ps * 1000 / timestep_fs)
    prod_energies = energies[discard_steps:]

    if len(prod_energies) == 0:
        print(f"Warning: No production data after discarding {discard_ps} ps")
        return np.nan, np.nan

    print(f"  Total steps: {len(energies)}, Production steps: {len(prod_energies)}")
    mean_E = np.mean(prod_energies)
    sem_E = np.std(prod_energies) / np.sqrt(len(prod_energies))
    return mean_E, sem_E


def main():
    # === USER INPUT ===
    # Window directories (adjust names to match your structure)
    is_window = "1.11_708"  # Initial state (1.10 Å)
    ts_window = "1.61_917"  # Transition state (1.59 Å)

    discard_ps = 9.0    # Equilibration time to discard
    timestep_fs = 1.0   # MD timestep

    # Known values
    delta_G_aimd = 1.07  # eV, AIMD water barrier
    delta_G_neb = 0.85   # eV, NEB+hTST barrier
    # === END USER INPUT ===

    print(f"Reading IS: {is_window}")
    E_IS, err_IS = compute_mean_energy(is_window, discard_ps, timestep_fs)

    print(f"\nReading TS: {ts_window}")
    E_TS, err_TS = compute_mean_energy(ts_window, discard_ps, timestep_fs)

    delta_H = E_TS - E_IS
    delta_H_err = np.sqrt(err_IS**2 + err_TS**2)

    T_delta_S = delta_H - delta_G_aimd

    print("\n=== Results ===")
    print(f"⟨E_pot⟩_IS:  {E_IS:.4f} ± {err_IS:.4f} eV")
    print(f"⟨E_pot⟩_TS:  {E_TS:.4f} ± {err_TS:.4f} eV")
    print(f"ΔH‡:         {delta_H:.3f} ± {delta_H_err:.3f} eV")
    print(f"ΔG‡ (AIMD):  {delta_G_aimd:.2f} eV")
    print(f"TΔS‡:        {T_delta_S:.3f} eV")

    print("\n=== Interpretation ===")
    if abs(delta_H - delta_G_neb) < 0.1:
        print(f"ΔH‡ ≈ NEB barrier ({delta_G_neb} eV)")
        print("→ Anharmonic correction is ENTROPIC")
        print("  (frustrated translations/hindered rotations)")
    else:
        print(f"ΔH‡ ≠ NEB barrier ({delta_G_neb} eV)")
        print("→ Mixed enthalpic/entropic contribution")


if __name__ == "__main__":
    main()
