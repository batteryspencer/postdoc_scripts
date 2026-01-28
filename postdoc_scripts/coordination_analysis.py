#!/usr/bin/env python3
"""
Coordination-Based Force Analysis for VASP AIMD Simulations

Analyzes constraint forces by categorizing frames based on C and H 
coordination numbers to surface Pt atoms. Useful for understanding 
bimodality in PMF calculations.

Key insight: Bimodality may arise from different binding modes 
(top/bridge/hollow for C, bound/unbound for H) rather than simple
distance variations.

Usage:
    1. Configure atom indices and cutoffs below
    2. Run: python coordination_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from ase.io import read as ase_read
from ase.data import covalent_radii, chemical_symbols

# Import from existing script
from constrained_force_stats import (read_simulation_data, get_file_line_count,
                                      find_histogram_peaks, find_closest_frames)


def get_ase_bond_cutoff(element1, element2, mult=1.3):
    """
    Calculate bond cutoff using ASE's covalent radii.

    This mimics how ASE GUI determines bonds for visualization.
    The formula is: cutoff = (covalent_radius_1 + covalent_radius_2) * mult

    ASE GUI typically uses mult ≈ 1.3 for bond visualization.

    Parameters:
        element1: Chemical symbol (e.g., 'C', 'H', 'Pt')
        element2: Chemical symbol
        mult: Multiplier for the sum of covalent radii (default 1.3)

    Returns:
        cutoff: Bond distance cutoff in Angstroms
    """
    idx1 = chemical_symbols.index(element1)
    idx2 = chemical_symbols.index(element2)
    base_cutoff = covalent_radii[idx1] + covalent_radii[idx2]
    return base_cutoff * mult

# =============================================================================
# CONFIGURATION
# =============================================================================

# Atom indices (0-based)
C_INDEX = 1          # Carbon atom
H_INDEX = 5          # Departing hydrogen atom

# Pt atom indices - all surface Pt that could potentially bond
# Adjust this list based on your slab structure
# For a 3x3 surface, top layer typically has 9 atoms
PT_INDICES = list(range(11, 56))  # Pt atoms at indices 11-55 (after 11 propane atoms)

# Bonding cutoff multiplier
# ASE GUI uses approximately 1.3 for bond visualization
# Formula: cutoff = (covalent_radius_A + covalent_radius_B) * BOND_MULT
BOND_MULT = 1.3

# Calculate cutoffs using ASE covalent radii (matching ASE GUI behavior)
# These are computed dynamically from ASE's covalent_radii database
C_PT_CUTOFF = get_ase_bond_cutoff('C', 'Pt', mult=BOND_MULT)
H_PT_CUTOFF = get_ase_bond_cutoff('H', 'Pt', mult=BOND_MULT)
C_H_CUTOFF = get_ase_bond_cutoff('C', 'H', mult=BOND_MULT)

# Print derived cutoffs for reference
print(f"ASE-derived bond cutoffs (mult={BOND_MULT}):")
print(f"  C-Pt: {C_PT_CUTOFF:.3f} Å (covalent radii: C={covalent_radii[chemical_symbols.index('C')]:.3f}, Pt={covalent_radii[chemical_symbols.index('Pt')]:.3f})")
print(f"  H-Pt: {H_PT_CUTOFF:.3f} Å (covalent radii: H={covalent_radii[chemical_symbols.index('H')]:.3f}, Pt={covalent_radii[chemical_symbols.index('Pt')]:.3f})")
print(f"  C-H:  {C_H_CUTOFF:.3f} Å")

# Constraint index
CONSTRAINT_INDEX = 0

# Output settings
OUTPUT_DIR = "coordination_analysis"
DPI = 300

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def read_trajectory_from_segments(folders, file_type='XDATCAR'):
    """Read and concatenate trajectories from multiple segment folders."""
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


def get_coordination_number(atoms, center_idx, neighbor_indices, cutoff):
    """
    Calculate coordination number for an atom.
    
    Parameters:
        atoms: ASE Atoms object
        center_idx: Index of central atom
        neighbor_indices: List of potential neighbor indices (e.g., Pt atoms)
        cutoff: Distance cutoff for bonding
    
    Returns:
        cn: Coordination number
        bonded_indices: List of indices within cutoff
    """
    bonded = []
    for pt_idx in neighbor_indices:
        dist = atoms.get_distance(center_idx, pt_idx, mic=True)
        if dist < cutoff:
            bonded.append(pt_idx)
    return len(bonded), bonded


def get_binding_mode_label(cn):
    """Convert coordination number to binding mode label."""
    if cn == 0:
        return "unbound"
    elif cn == 1:
        return "top"
    elif cn == 2:
        return "bridge"
    elif cn >= 3:
        return "hollow"
    return f"cn{cn}"


def analyze_trajectory_coordination(trajectory, c_idx, h_idx, pt_indices, 
                                     c_cutoff, h_cutoff):
    """
    Analyze coordination numbers for C and H across trajectory.
    
    Returns:
        results: dict with arrays of CN values, labels, and bonded Pt indices
    """
    results = {
        'c_cn': [],
        'h_cn': [],
        'c_mode': [],
        'h_mode': [],
        'combined_label': [],
        'c_bonded_pt': [],
        'h_bonded_pt': [],
    }
    
    for atoms in trajectory:
        # Get coordination numbers
        c_cn, c_bonded = get_coordination_number(atoms, c_idx, pt_indices, c_cutoff)
        h_cn, h_bonded = get_coordination_number(atoms, h_idx, pt_indices, h_cutoff)
        
        # Get mode labels
        c_mode = get_binding_mode_label(c_cn)
        h_mode = get_binding_mode_label(h_cn)
        
        # Combined label
        combined = f"C-{c_mode}_H-{h_mode}"
        
        results['c_cn'].append(c_cn)
        results['h_cn'].append(h_cn)
        results['c_mode'].append(c_mode)
        results['h_mode'].append(h_mode)
        results['combined_label'].append(combined)
        results['c_bonded_pt'].append(tuple(sorted(c_bonded)))
        results['h_bonded_pt'].append(tuple(sorted(h_bonded)))
    
    # Convert to numpy arrays where appropriate
    results['c_cn'] = np.array(results['c_cn'])
    results['h_cn'] = np.array(results['h_cn'])
    
    return results


def compute_category_statistics(forces, labels):
    """
    Compute force statistics for each category.

    Returns:
        stats: dict mapping label to {mean, std, sem, count, fraction}
    """
    unique_labels = sorted(set(labels))
    stats = {}

    total = len(forces)
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        f = forces[mask]
        stats[label] = {
            'mean': np.mean(f),
            'std': np.std(f),
            'sem': np.std(f) / np.sqrt(len(f)) if len(f) > 1 else 0,
            'count': len(f),
            'fraction': len(f) / total,
        }

    return stats


def analyze_peaks_by_category(forces, labels, frame_indices=None):
    """
    Detect histogram peaks for each coordination category.

    Parameters:
        forces: Array of force values
        labels: List of category labels for each frame
        frame_indices: Optional array of original frame indices (if None, uses 0-based)

    Returns:
        peak_analysis: dict mapping category to peak information
    """
    if frame_indices is None:
        frame_indices = np.arange(len(forces))

    unique_labels = sorted(set(labels))
    peak_analysis = {}

    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        cat_forces = forces[mask]
        cat_frame_indices = frame_indices[mask]

        # Need enough data points for meaningful peak detection
        if len(cat_forces) < 50:
            peak_analysis[label] = {
                'peaks': [],
                'peak_frames': {},
                'note': 'Insufficient data for peak detection'
            }
            continue

        # Find peaks in this category's force distribution
        peaks = find_histogram_peaks(cat_forces, bins=50)

        # Find frames corresponding to each peak
        peak_forces = [p[0] for p in peaks]

        # Map back to original frame indices
        peak_frames = {}
        for peak_force in peak_forces:
            tolerance = 0.1
            close_mask = np.abs(cat_forces - peak_force) <= tolerance
            close_frames = cat_frame_indices[close_mask]
            # Store up to 5 representative frames
            peak_frames[peak_force] = list(close_frames[:5])

        peak_analysis[label] = {
            'peaks': peaks,
            'peak_frames': peak_frames,
            'note': None
        }

    return peak_analysis


def analyze_overall_peaks(forces):
    """
    Detect histogram peaks for the overall force distribution.

    Returns:
        peaks: list of (force, frequency) tuples
        peak_frames: dict mapping peak force to list of frame indices
    """
    peaks = find_histogram_peaks(forces, bins=100)
    peak_forces = [p[0] for p in peaks]
    peak_frames = find_closest_frames(list(forces), peak_forces, tolerance=0.1)
    return peaks, peak_frames


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_force_histogram_by_category(forces, labels, output_dir):
    """Plot force histograms colored by binding mode category (two separate figures)."""
    unique_labels = sorted(set(labels))
    n_categories = len(unique_labels)

    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_categories, 10)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    force_by_cat = {label: forces[np.array([l == label for l in labels])]
                    for label in unique_labels}

    # Figure 1: Stacked histogram
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist([force_by_cat[l] for l in unique_labels],
             bins=50, stacked=True,
             label=unique_labels,
             color=[color_map[l] for l in unique_labels],
             edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Force (eV/Å)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Force Distribution by Binding Mode')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'force_histogram_stacked.png'), dpi=DPI)
    plt.close(fig1)

    # Figure 2: Overlaid normalized histograms
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for label in unique_labels:
        f = force_by_cat[label]
        if len(f) > 10:
            ax2.hist(f, bins=30, alpha=0.5, density=True,
                    label=f"{label} (n={len(f)})",
                    color=color_map[label])
    ax2.set_xlabel('Force (eV/Å)')
    ax2.set_ylabel('Density')
    ax2.set_title('Normalized Force Distributions')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'force_histogram_normalized.png'), dpi=DPI)
    plt.close(fig2)


def plot_coordination_time_series(results, forces, timestep_fs, output_dir):
    """Plot coordination numbers and forces over time."""
    n_frames = len(results['c_cn'])
    time_ps = np.arange(n_frames) * timestep_fs / 1000
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Force
    ax1 = axes[0]
    ax1.plot(time_ps[:len(forces)], forces[:n_frames], 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Force (eV/Å)')
    ax1.set_title('Constrained Force')
    ax1.grid(True, alpha=0.3)
    
    # C coordination
    ax2 = axes[1]
    ax2.plot(time_ps, results['c_cn'], 'g-', linewidth=0.5, alpha=0.7)
    ax2.set_ylabel('C-Pt CN')
    ax2.set_title('Carbon Coordination Number')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax2.grid(True, alpha=0.3)
    
    # H coordination
    ax3 = axes[2]
    ax3.plot(time_ps, results['h_cn'], 'r-', linewidth=0.5, alpha=0.7)
    ax3.set_ylabel('H-Pt CN')
    ax3.set_xlabel('Time (ps)')
    ax3.set_title('Hydrogen Coordination Number')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coordination_time_series.png'), dpi=DPI)
    plt.close()


def plot_force_vs_coordination(forces, results, output_dir):
    """Scatter plot of force vs coordination numbers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n = min(len(forces), len(results['c_cn']))
    f = forces[:n]
    c_cn = results['c_cn'][:n]
    h_cn = results['h_cn'][:n]
    
    # Force vs C-CN with jitter
    ax1 = axes[0]
    jitter_c = c_cn + np.random.normal(0, 0.1, len(c_cn))
    ax1.scatter(jitter_c, f, alpha=0.2, s=5, c='green')
    
    # Add box plot overlay
    for cn in sorted(set(c_cn)):
        mask = c_cn == cn
        if np.sum(mask) > 0:
            mean_f = np.mean(f[mask])
            std_f = np.std(f[mask])
            ax1.errorbar(cn, mean_f, yerr=std_f, fmt='ro', markersize=10, 
                        capsize=5, capthick=2, elinewidth=2)
    
    ax1.set_xlabel('C-Pt Coordination Number')
    ax1.set_ylabel('Force (eV/Å)')
    ax1.set_title('Force vs Carbon Coordination')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax1.grid(True, alpha=0.3)
    
    # Force vs H-CN with jitter
    ax2 = axes[1]
    jitter_h = h_cn + np.random.normal(0, 0.1, len(h_cn))
    ax2.scatter(jitter_h, f, alpha=0.2, s=5, c='red')
    
    for cn in sorted(set(h_cn)):
        mask = h_cn == cn
        if np.sum(mask) > 0:
            mean_f = np.mean(f[mask])
            std_f = np.std(f[mask])
            ax2.errorbar(cn, mean_f, yerr=std_f, fmt='bo', markersize=10,
                        capsize=5, capthick=2, elinewidth=2)
    
    ax2.set_xlabel('H-Pt Coordination Number')
    ax2.set_ylabel('Force (eV/Å)')
    ax2.set_title('Force vs Hydrogen Coordination')
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'force_vs_coordination.png'), dpi=DPI)
    plt.close()


def plot_2d_coordination_heatmap(forces, results, output_dir):
    """2D heatmap of mean force as function of C-CN and H-CN."""
    n = min(len(forces), len(results['c_cn']))
    f = forces[:n]
    c_cn = results['c_cn'][:n]
    h_cn = results['h_cn'][:n]
    
    # Create 2D binned statistics
    c_bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
    h_bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
    
    mean_force = np.zeros((4, 4))
    count = np.zeros((4, 4))
    
    for i in range(n):
        c = int(min(c_cn[i], 3))
        h = int(min(h_cn[i], 3))
        mean_force[c, h] += f[i]
        count[c, h] += 1
    
    # Avoid division by zero
    with np.errstate(invalid='ignore'):
        mean_force = np.where(count > 0, mean_force / count, np.nan)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean force heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(mean_force, origin='lower', aspect='auto', cmap='RdYlBu_r')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax1.set_xlabel('H-Pt Coordination')
    ax1.set_ylabel('C-Pt Coordination')
    ax1.set_title('Mean Force (eV/Å)')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            if count[i, j] > 0:
                ax1.text(j, i, f'{mean_force[i,j]:.2f}\n(n={int(count[i,j])})',
                        ha='center', va='center', fontsize=8)
    
    # Count heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(count, origin='lower', aspect='auto', cmap='Blues')
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['unbound', 'top', 'bridge', 'hollow'])
    ax2.set_xlabel('H-Pt Coordination')
    ax2.set_ylabel('C-Pt Coordination')
    ax2.set_title('Frame Count')
    plt.colorbar(im2, ax=ax2)
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            if count[i, j] > 0:
                pct = 100 * count[i, j] / n
                ax2.text(j, i, f'{int(count[i,j])}\n({pct:.1f}%)',
                        ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coordination_heatmap.png'), dpi=DPI)
    plt.close()


def plot_pt_site_analysis(results, forces, output_dir):
    """Analyze which specific Pt atoms are involved in bonding."""
    n = min(len(forces), len(results['c_bonded_pt']))
    
    # Count frequency of each Pt site combination
    c_site_counts = defaultdict(list)
    h_site_counts = defaultdict(list)
    
    for i in range(n):
        c_sites = results['c_bonded_pt'][i]
        h_sites = results['h_bonded_pt'][i]
        c_site_counts[c_sites].append(forces[i])
        h_site_counts[h_sites].append(forces[i])
    
    # Print summary
    print("\n" + "=" * 60)
    print("Pt Site Analysis")
    print("=" * 60)
    
    print("\nC bonding sites (sorted by frequency):")
    for sites, force_list in sorted(c_site_counts.items(), 
                                     key=lambda x: -len(x[1]))[:10]:
        mean_f = np.mean(force_list)
        print(f"  Pt{list(sites)}: n={len(force_list)}, "
              f"mean force={mean_f:.3f} eV/Å")
    
    print("\nH bonding sites (sorted by frequency):")
    for sites, force_list in sorted(h_site_counts.items(), 
                                     key=lambda x: -len(x[1]))[:10]:
        mean_f = np.mean(force_list)
        print(f"  Pt{list(sites)}: n={len(force_list)}, "
              f"mean force={mean_f:.3f} eV/Å")
    
    return c_site_counts, h_site_counts


def get_timestep_from_incar(incar_path='INCAR'):
    """Read POTIM from INCAR file."""
    try:
        with open(incar_path) as f:
            for line in f:
                if 'POTIM' in line:
                    return float(line.split('=')[-1].strip())
    except FileNotFoundError:
        pass
    return 1.0


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Coordination-Based Force Analysis")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  C atom index: {C_INDEX}")
    print(f"  H atom index: {H_INDEX}")
    print(f"  Bond cutoff multiplier: {BOND_MULT} (ASE GUI-like)")
    print(f"  C-Pt cutoff: {C_PT_CUTOFF:.3f} Å (ASE covalent radii)")
    print(f"  H-Pt cutoff: {H_PT_CUTOFF:.3f} Å (ASE covalent radii)")
    print(f"  Pt atoms: {len(PT_INDICES)} atoms")
    
    # Get simulation parameters
    num_constraints = get_file_line_count('ICONST')
    timestep_fs = get_timestep_from_incar('INCAR')
    print(f"  Timestep: {timestep_fs} fs")
    
    # Find segment folders
    folders = sorted(glob('seg*'))
    if not folders:
        print("ERROR: No segment folders found!")
        return
    print(f"\nFound {len(folders)} segment folders")
    
    # Read forces
    print("\nReading force data...")
    all_lambda_values = []
    for folder in folders:
        report_path = os.path.join(folder, 'REPORT')
        if os.path.exists(report_path):
            lambda_vals, _, _ = read_simulation_data(folder, max_steps=None)
            all_lambda_values.extend(lambda_vals)
    
    forces = np.array(all_lambda_values[CONSTRAINT_INDEX::num_constraints])
    print(f"  Total force values: {len(forces)}")
    
    # Read trajectory
    print("\nReading trajectory...")
    trajectory = read_trajectory_from_segments(folders)
    print(f"  Total frames: {len(trajectory)}")

    # Analyze coordination
    print("\nAnalyzing coordination numbers...")
    results = analyze_trajectory_coordination(
        trajectory, C_INDEX, H_INDEX, PT_INDICES, C_PT_CUTOFF, H_PT_CUTOFF
    )
    
    # Summary statistics
    n = min(len(forces), len(results['c_cn']))
    print(f"\nCoordination Summary (n={n} frames):")
    print(f"  C-Pt CN: mean={np.mean(results['c_cn'][:n]):.2f}, "
          f"range=[{np.min(results['c_cn'][:n])}, {np.max(results['c_cn'][:n])}]")
    print(f"  H-Pt CN: mean={np.mean(results['h_cn'][:n]):.2f}, "
          f"range=[{np.min(results['h_cn'][:n])}, {np.max(results['h_cn'][:n])}]")
    
    # Category statistics
    print("\n" + "=" * 60)
    print("Force Statistics by Category")
    print("=" * 60)
    
    labels = results['combined_label'][:n]
    stats = compute_category_statistics(forces[:n], labels)
    
    print(f"\n{'Category':<25} {'Count':>8} {'Fraction':>10} {'Mean Force':>12} {'Std':>10}")
    print("-" * 70)
    for label in sorted(stats.keys(), key=lambda x: -stats[x]['count']):
        s = stats[label]
        print(f"{label:<25} {s['count']:>8} {s['fraction']:>10.1%} "
              f"{s['mean']:>12.3f} {s['std']:>10.3f}")
    
    # Pt site analysis
    c_sites, h_sites = plot_pt_site_analysis(results, forces[:n], OUTPUT_DIR)

    # Peak analysis
    print("\n" + "=" * 60)
    print("Histogram Peak Analysis")
    print("=" * 60)

    # Overall peaks
    overall_peaks, overall_peak_frames = analyze_overall_peaks(forces[:n])
    print("\nOverall Force Distribution Peaks:")
    for i, (force, freq) in enumerate(overall_peaks, 1):
        frames = overall_peak_frames[force]
        frame_str = ", ".join(str(f) for f in frames)
        print(f"  Peak {i}: Force = {force:.3f} eV/Å, Frequency = {freq:.0f}")
        print(f"           Reference Frames: {frame_str}")

    # Per-category peaks
    frame_indices = np.arange(n)
    category_peaks = analyze_peaks_by_category(forces[:n], labels, frame_indices)

    print("\nPeaks by Coordination Category:")
    for label in sorted(category_peaks.keys(), key=lambda x: -stats[x]['count']):
        cat_info = category_peaks[label]
        print(f"\n  {label}:")
        if cat_info['note']:
            print(f"    {cat_info['note']}")
        else:
            for i, (force, freq) in enumerate(cat_info['peaks'], 1):
                frames = cat_info['peak_frames'].get(force, [])
                frame_str = ", ".join(str(f) for f in frames) if frames else "N/A"
                print(f"    Peak {i}: Force = {force:.3f} eV/Å, Freq = {freq:.0f}, Frames: {frame_str}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_force_histogram_by_category(forces[:n], labels, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/force_histogram_stacked.png")
    print(f"  Saved: {OUTPUT_DIR}/force_histogram_normalized.png")
    
    plot_coordination_time_series(results, forces, timestep_fs, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/coordination_time_series.png")
    
    plot_force_vs_coordination(forces[:n], results, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/force_vs_coordination.png")
    
    plot_2d_coordination_heatmap(forces[:n], results, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/coordination_heatmap.png")
    
    # Write report
    report_path = os.path.join(OUTPUT_DIR, 'coordination_report.txt')
    with open(report_path, 'w') as f:
        f.write("Coordination-Based Force Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  C atom index: {C_INDEX}\n")
        f.write(f"  H atom index: {H_INDEX}\n")
        f.write(f"  Bond cutoff multiplier: {BOND_MULT} (ASE GUI-like)\n")
        f.write(f"  C-Pt cutoff: {C_PT_CUTOFF:.3f} Å (ASE covalent radii)\n")
        f.write(f"  H-Pt cutoff: {H_PT_CUTOFF:.3f} Å (ASE covalent radii)\n")
        f.write(f"  C-H cutoff:  {C_H_CUTOFF:.3f} Å (ASE covalent radii)\n")
        f.write(f"  Frames analyzed: {n}\n\n")
        
        f.write("Category Statistics:\n")
        f.write("-" * 50 + "\n")
        for label in sorted(stats.keys(), key=lambda x: -stats[x]['count']):
            s = stats[label]
            f.write(f"\n{label}:\n")
            f.write(f"  Count: {s['count']} ({s['fraction']:.1%})\n")
            f.write(f"  Mean force: {s['mean']:.4f} ± {s['sem']:.4f} eV/Å\n")
            f.write(f"  Std dev: {s['std']:.4f} eV/Å\n")
        
        f.write("\n" + "-" * 50 + "\n")
        f.write("Top C bonding site combinations:\n")
        for sites, force_list in sorted(c_sites.items(), 
                                         key=lambda x: -len(x[1]))[:10]:
            f.write(f"  Pt{list(sites)}: n={len(force_list)}, "
                   f"mean={np.mean(force_list):.3f} eV/Å\n")
        
        f.write("\nTop H bonding site combinations:\n")
        for sites, force_list in sorted(h_sites.items(),
                                         key=lambda x: -len(x[1]))[:10]:
            f.write(f"  Pt{list(sites)}: n={len(force_list)}, "
                   f"mean={np.mean(force_list):.3f} eV/Å\n")

        # Write peak analysis
        f.write("\n" + "=" * 50 + "\n")
        f.write("Histogram Peak Analysis\n")
        f.write("=" * 50 + "\n")

        f.write("\nOverall Force Distribution Peaks:\n")
        f.write("-" * 50 + "\n")
        for i, (force, freq) in enumerate(overall_peaks, 1):
            frames = overall_peak_frames[force]
            frame_str = ", ".join(str(fr) for fr in frames)
            f.write(f"Peak {i}: Force = {force:.4f} eV/Å, Frequency = {freq:.0f}\n")
            f.write(f"   Reference Frames: {frame_str}\n")

        f.write("\nPeaks by Coordination Category:\n")
        f.write("-" * 50 + "\n")
        for label in sorted(category_peaks.keys(), key=lambda x: -stats[x]['count']):
            cat_info = category_peaks[label]
            f.write(f"\n{label}:\n")
            if cat_info['note']:
                f.write(f"  {cat_info['note']}\n")
            else:
                for i, (force, freq) in enumerate(cat_info['peaks'], 1):
                    frames = cat_info['peak_frames'].get(force, [])
                    frame_str = ", ".join(str(fr) for fr in frames) if frames else "N/A"
                    f.write(f"  Peak {i}: Force = {force:.4f} eV/Å, Frequency = {freq:.0f}\n")
                    f.write(f"     Reference Frames: {frame_str}\n")

    print(f"  Saved: {report_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
