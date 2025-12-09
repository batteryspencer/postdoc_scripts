#!/usr/bin/env python3
"""
Force-Distance Correlation Analysis for VASP AIMD Simulations

Analyzes correlations between constrained forces and specific bond distances
across simulation trajectories. Useful for understanding which geometric
parameters drive force variations in PMF calculations.

Usage:
    1. Configure ATOM_PAIRS below with your atom indices and labels
    2. Run: python force_distance_correlation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import pearsonr, spearmanr
from ase.io import read as ase_read

# Import force reading function from existing script
from constrained_force_stats import read_simulation_data, get_file_line_count

# =============================================================================
# CONFIGURATION - Edit these for your system
# =============================================================================

# Atom pairs to analyze: (atom_index_1, atom_index_2, label)
# Indices are 0-based (matching ASE/Python convention)
ATOM_PAIRS = [
    (0, 116, "C2-Pt1"),
    (30, 116, "H-Pt1"),
    (30, 101, "H-Pt2"),
]

# Constraint index (which constraint to analyze if multiple in ICONST)
CONSTRAINT_INDEX = 0

# Maximum MD steps to analyze (None = all steps)
MAX_STEPS = None

# Output settings
OUTPUT_DIR = "correlation_analysis"
DPI = 300

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def read_trajectory_from_segments(folders, file_type='XDATCAR'):
    """
    Read and concatenate trajectories from multiple segment folders.

    Parameters:
        folders: List of folder paths containing trajectory files
        file_type: 'XDATCAR' or 'OUTCAR'

    Returns:
        List of ASE Atoms objects (trajectory frames)
    """
    all_frames = []

    for folder in folders:
        traj_file = os.path.join(folder, file_type)
        if os.path.exists(traj_file):
            try:
                frames = ase_read(traj_file, index=':')
                all_frames.extend(frames)
                print(f"  Read {len(frames)} frames from {folder}/{file_type}")
            except Exception as e:
                print(f"  Warning: Could not read {traj_file}: {e}")

    return all_frames


def calculate_distance(atoms, idx1, idx2, mic=True):
    """
    Calculate distance between two atoms with minimum image convention.

    Parameters:
        atoms: ASE Atoms object
        idx1, idx2: Atom indices (0-based)
        mic: Use minimum image convention for periodic boundaries

    Returns:
        Distance in Angstroms
    """
    return atoms.get_distance(idx1, idx2, mic=mic)


def calculate_distances_trajectory(trajectory, atom_pairs):
    """
    Calculate distances for all atom pairs across entire trajectory.

    Parameters:
        trajectory: List of ASE Atoms objects
        atom_pairs: List of (idx1, idx2, label) tuples

    Returns:
        Dictionary mapping labels to arrays of distances
    """
    distances = {label: [] for _, _, label in atom_pairs}

    for atoms in trajectory:
        for idx1, idx2, label in atom_pairs:
            dist = calculate_distance(atoms, idx1, idx2)
            distances[label].append(dist)

    # Convert to numpy arrays
    for label in distances:
        distances[label] = np.array(distances[label])

    return distances


def get_timestep_from_incar(incar_path='INCAR'):
    """Read POTIM (timestep in fs) from INCAR file."""
    try:
        with open(incar_path) as f:
            for line in f:
                if 'POTIM' in line:
                    return float(line.split('=')[-1].strip())
    except FileNotFoundError:
        pass
    return 1.0  # Default fallback


def calculate_correlation(forces, distances):
    """
    Calculate Pearson and Spearman correlation coefficients.

    Returns:
        Dictionary with correlation statistics
    """
    # Ensure same length
    min_len = min(len(forces), len(distances))
    f = forces[:min_len]
    d = distances[:min_len]

    pearson_r, pearson_p = pearsonr(f, d)
    spearman_r, spearman_p = spearmanr(f, d)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_points': min_len
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_distance_vs_time(distances, timestep_fs, output_dir):
    """Plot distance evolution over simulation time for all atom pairs."""
    fig, axes = plt.subplots(len(distances), 1, figsize=(10, 3*len(distances)),
                             sharex=True)
    if len(distances) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(distances)))

    for ax, (label, dist), color in zip(axes, distances.items(), colors):
        time_ps = np.arange(len(dist)) * timestep_fs / 1000  # Convert to ps
        ax.plot(time_ps, dist, color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f'{label}\nDistance (Å)')
        ax.axhline(np.mean(dist), color='red', linestyle='--',
                   label=f'Mean: {np.mean(dist):.3f} Å')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Simulation Time (ps)')
    fig.suptitle('Bond Distances Along Trajectory', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_vs_time.png'), dpi=DPI)
    plt.close()


def plot_force_vs_distance(forces, distances, output_dir):
    """Create scatter plots of force vs distance for each atom pair."""
    n_pairs = len(distances)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 4))
    if n_pairs == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))

    for ax, (label, dist), color in zip(axes, distances.items(), colors):
        min_len = min(len(forces), len(dist))
        f = forces[:min_len]
        d = dist[:min_len]

        # Calculate correlation
        corr = calculate_correlation(f, d)

        # Scatter plot with transparency for overlapping points
        ax.scatter(d, f, alpha=0.3, s=10, color=color, edgecolors='none')

        # Add linear fit line
        z = np.polyfit(d, f, 1)
        p = np.poly1d(z)
        d_range = np.linspace(d.min(), d.max(), 100)
        ax.plot(d_range, p(d_range), 'r-', linewidth=2,
                label=f'Linear fit')

        ax.set_xlabel(f'{label} Distance (Å)')
        ax.set_ylabel('Force (eV/Å)')
        ax.set_title(f'{label}\nPearson r = {corr["pearson_r"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('Force vs Distance Correlation', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'force_vs_distance.png'), dpi=DPI)
    plt.close()


def plot_combined_analysis(forces, distances, timestep_fs, output_dir):
    """Create a combined figure with force, distances, and correlations."""
    n_pairs = len(distances)
    fig = plt.figure(figsize=(14, 3 + 3*n_pairs))

    # Create grid: top row for force, middle rows for distances, bottom for correlations
    gs = fig.add_gridspec(2 + n_pairs, n_pairs, height_ratios=[1] + [1]*n_pairs + [1.2])

    time_ps = np.arange(len(forces)) * timestep_fs / 1000

    # Top row: Force vs time (spans all columns)
    ax_force = fig.add_subplot(gs[0, :])
    ax_force.plot(time_ps, forces, 'b-', linewidth=0.5, alpha=0.8)
    ax_force.set_ylabel('Force (eV/Å)')
    ax_force.set_title('Constrained Force Along Trajectory')
    ax_force.axhline(np.mean(forces), color='red', linestyle='--',
                     label=f'Mean: {np.mean(forces):.2f}')
    ax_force.legend(loc='upper right')
    ax_force.grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))

    # Middle rows: Distance vs time
    for i, ((label, dist), color) in enumerate(zip(distances.items(), colors)):
        ax = fig.add_subplot(gs[1 + i, :])
        dist_time = np.arange(len(dist)) * timestep_fs / 1000
        ax.plot(dist_time, dist, color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f'{label} (Å)')
        ax.axhline(np.mean(dist), color='red', linestyle='--')
        ax.grid(True, alpha=0.3)
        if i == n_pairs - 1:
            ax.set_xlabel('Simulation Time (ps)')

    # Bottom row: Force vs Distance scatter plots
    for i, ((label, dist), color) in enumerate(zip(distances.items(), colors)):
        ax = fig.add_subplot(gs[-1, i])
        min_len = min(len(forces), len(dist))
        f = forces[:min_len]
        d = dist[:min_len]

        corr = calculate_correlation(f, d)
        ax.scatter(d, f, alpha=0.2, s=5, color=color, edgecolors='none')

        # Linear fit
        z = np.polyfit(d, f, 1)
        p = np.poly1d(z)
        d_range = np.linspace(d.min(), d.max(), 100)
        ax.plot(d_range, p(d_range), 'r-', linewidth=2)

        ax.set_xlabel(f'{label} Distance (Å)')
        ax.set_ylabel('Force (eV/Å)')
        ax.set_title(f'r = {corr["pearson_r"]:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_analysis.png'), dpi=DPI)
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("Force-Distance Correlation Analysis")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get simulation parameters
    num_constraints = get_file_line_count('ICONST')
    timestep_fs = get_timestep_from_incar('INCAR')
    print(f"\nSimulation parameters:")
    print(f"  Timestep: {timestep_fs} fs")
    print(f"  Number of constraints: {num_constraints}")
    print(f"  Analyzing constraint index: {CONSTRAINT_INDEX}")

    # Find and sort segment folders
    folders = sorted(glob('seg*'))
    if not folders:
        print("ERROR: No segment folders found!")
        return
    print(f"\nFound {len(folders)} segment folders")

    # Read force data from all segments
    print("\nReading force data...")
    all_lambda_values = []
    for folder in folders:
        report_path = os.path.join(folder, 'REPORT')
        if os.path.exists(report_path):
            lambda_vals, _, _ = read_simulation_data(folder, max_steps=MAX_STEPS)
            all_lambda_values.extend(lambda_vals)

    # Extract forces for the specific constraint
    forces = np.array(all_lambda_values[CONSTRAINT_INDEX::num_constraints])
    print(f"  Total force values: {len(forces)}")

    # Read trajectory data
    print("\nReading trajectory data...")
    trajectory = read_trajectory_from_segments(folders, file_type='XDATCAR')
    print(f"  Total trajectory frames: {len(trajectory)}")

    # Verify atom pair indices are valid
    n_atoms = len(trajectory[0]) if trajectory else 0
    print(f"  Atoms per frame: {n_atoms}")

    for idx1, idx2, label in ATOM_PAIRS:
        if idx1 >= n_atoms or idx2 >= n_atoms:
            print(f"ERROR: Atom indices ({idx1}, {idx2}) for {label} exceed "
                  f"number of atoms ({n_atoms})")
            return

    # Calculate distances
    print("\nCalculating distances...")
    distances = calculate_distances_trajectory(trajectory, ATOM_PAIRS)
    for label, dist in distances.items():
        print(f"  {label}: {len(dist)} values, mean = {np.mean(dist):.3f} Å")

    # Check data alignment
    min_len = min(len(forces), min(len(d) for d in distances.values()))
    if len(forces) != len(trajectory):
        print(f"\nNote: Force data ({len(forces)}) and trajectory ({len(trajectory)}) "
              f"have different lengths. Using {min_len} points.")

    # Calculate correlations
    print("\n" + "=" * 60)
    print("Correlation Analysis Results")
    print("=" * 60)

    correlation_results = {}
    for label, dist in distances.items():
        corr = calculate_correlation(forces, dist)
        correlation_results[label] = corr
        print(f"\n{label}:")
        print(f"  Pearson r:  {corr['pearson_r']:+.4f} (p = {corr['pearson_p']:.2e})")
        print(f"  Spearman r: {corr['spearman_r']:+.4f} (p = {corr['spearman_p']:.2e})")

    # Generate plots
    print("\nGenerating plots...")
    plot_distance_vs_time(distances, timestep_fs, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/distance_vs_time.png")

    plot_force_vs_distance(forces, distances, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/force_vs_distance.png")

    plot_combined_analysis(forces, distances, timestep_fs, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/combined_analysis.png")

    # Write summary report
    report_path = os.path.join(OUTPUT_DIR, 'correlation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Force-Distance Correlation Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestep: {timestep_fs} fs\n")
        f.write(f"Total frames analyzed: {min_len}\n")
        f.write(f"Constraint index: {CONSTRAINT_INDEX}\n\n")

        f.write("Atom Pairs Analyzed:\n")
        for idx1, idx2, label in ATOM_PAIRS:
            f.write(f"  {label}: atoms {idx1} - {idx2}\n")

        f.write("\n" + "-" * 50 + "\n")
        f.write("Correlation Results\n")
        f.write("-" * 50 + "\n\n")

        for label, corr in correlation_results.items():
            dist = distances[label]
            f.write(f"{label}:\n")
            f.write(f"  Distance range: {np.min(dist):.3f} - {np.max(dist):.3f} Å\n")
            f.write(f"  Distance mean:  {np.mean(dist):.3f} ± {np.std(dist):.3f} Å\n")
            f.write(f"  Pearson r:      {corr['pearson_r']:+.4f} (p = {corr['pearson_p']:.2e})\n")
            f.write(f"  Spearman r:     {corr['spearman_r']:+.4f} (p = {corr['spearman_p']:.2e})\n\n")

        f.write("-" * 50 + "\n")
        f.write("Interpretation:\n")
        f.write("  |r| > 0.7: Strong correlation\n")
        f.write("  |r| 0.4-0.7: Moderate correlation\n")
        f.write("  |r| < 0.4: Weak correlation\n")

    print(f"  Saved: {report_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
