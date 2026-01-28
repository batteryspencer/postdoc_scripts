#!/usr/bin/env python3
"""
Coordination Evolution Along Reaction Coordinate

Aggregates coordination analysis data from multiple simulation directories
and visualizes how C-Pt and H-Pt coordination evolves along the reaction
coordinate (C-H bond distance).

Usage:
    python coordination_along_reaction_coordinate.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output settings
OUTPUT_DIR = "coordination_evolution"
DPI = 300

# Plot styling (publication style)
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
TICK_LABELSIZE = 12
LEGEND_FONTSIZE = 10
TICK_LENGTH_MAJOR = 6
TICK_WIDTH_MAJOR = 1

# Binding mode colors (publication-friendly color scheme)
MODE_COLORS = {
    'unbound': '#4575b4',   # Blue
    'top': '#91cf60',       # Green
    'bridge': '#fc8d59',    # Orange
    'hollow': '#d73027',    # Red
}

# Binding mode markers
MODE_MARKERS = {
    'unbound': 'o',
    'top': 's',
    'bridge': '^',
    'hollow': 'D',
}

# Reaction coordinate markers (IS, TS, FS)
RC_MARKERS = {
    'IS': 1.10,   # Initial State (C-H equilibrium)
    'TS': 1.61,   # Transition State
    'FS': 3.01,   # Final State (dissociated)
}
RC_MARKER_COLORS = {
    'IS': '#2ca02c',  # Green
    'TS': '#d62728',  # Red
    'FS': '#1f77b4',  # Blue
}


def add_rc_markers(ax):
    """Add vertical dashed lines for IS, TS, FS positions."""
    for label, x_pos in RC_MARKERS.items():
        ax.axvline(x_pos, color=RC_MARKER_COLORS[label], linestyle='--',
                   linewidth=1.5, alpha=0.7, label=label)

# =============================================================================
# DATA COLLECTION
# =============================================================================

def parse_coordination_report(report_path):
    """
    Parse a coordination_report.txt file and extract category statistics.

    Returns:
        dict with:
            - 'frames_analyzed': int
            - 'categories': dict mapping category name to {count, fraction, mean_force, std_dev}
    """
    result = {
        'frames_analyzed': 0,
        'categories': {}
    }

    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r') as f:
        content = f.read()

    # Extract frames analyzed
    frames_match = re.search(r'Frames analyzed:\s*(\d+)', content)
    if frames_match:
        result['frames_analyzed'] = int(frames_match.group(1))

    # Parse category statistics
    # Pattern: "C-mode_H-mode:\n  Count: N (X%)\n  Mean force: Y +/- Z eV/A\n  Std dev: W eV/A"
    category_pattern = re.compile(
        r'(C-\w+_H-\w+):\s*\n'
        r'\s*Count:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n'
        r'\s*Mean force:\s*([+-]?[0-9.]+)\s*.*?eV/.*?\n'
        r'\s*Std dev:\s*([0-9.]+)',
        re.MULTILINE
    )

    for match in category_pattern.finditer(content):
        category = match.group(1)
        count = int(match.group(2))
        fraction = float(match.group(3)) / 100.0
        mean_force = float(match.group(4))
        std_dev = float(match.group(5))

        result['categories'][category] = {
            'count': count,
            'fraction': fraction,
            'mean_force': mean_force,
            'std_dev': std_dev,
        }

    return result


def parse_force_stats_report(report_path):
    """
    Parse force_stats_report.txt to extract histogram peak information.

    Returns:
        dict with:
            - 'n_peaks': int (number of detected peaks)
            - 'peaks': list of dicts with 'force', 'frequency'
            - 'mean_force': float
            - 'std_err': float
    """
    result = {
        'n_peaks': 0,
        'peaks': [],
        'mean_force': None,
        'std_err': None,
    }

    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r') as f:
        content = f.read()

    # Extract mean force and standard error
    mean_match = re.search(r'Mean Force:\s*([+-]?[0-9.]+)', content)
    if mean_match:
        result['mean_force'] = float(mean_match.group(1))

    std_match = re.search(r'Standard Error of Mean:\s*([0-9.]+)', content)
    if std_match:
        result['std_err'] = float(std_match.group(1))

    # Extract peaks
    # Format: "Peak N: Force = X.XX, Frequency = Y"
    peak_pattern = re.compile(r'Peak\s+\d+:\s*Force\s*=\s*([+-]?[0-9.]+),\s*Frequency\s*=\s*(\d+)')
    for match in peak_pattern.finditer(content):
        result['peaks'].append({
            'force': float(match.group(1)),
            'frequency': int(match.group(2)),
        })

    result['n_peaks'] = len(result['peaks'])
    return result


def collect_coordination_data():
    """
    Scan all simulation directories and collect coordination data.

    Returns:
        list of dicts, each with:
            - 'ch_distance': float (C-H bond length)
            - 'dir_name': str
            - 'data': parsed coordination report
    """
    all_data = []

    # Find directories matching pattern
    for folder in sorted(glob('[0-9].[0-9][0-9]_*')):
        if not os.path.isdir(folder):
            continue

        # Extract C-H distance from folder name
        match = re.match(r'([0-9.]+)_', folder)
        if not match:
            continue

        ch_distance = float(match.group(1))
        report_path = os.path.join(folder, 'coordination_analysis', 'coordination_report.txt')

        data = parse_coordination_report(report_path)
        if data is not None and data['categories']:
            # Also parse force stats report for peak information
            force_stats_path = os.path.join(folder, 'force_stats_report.txt')
            force_stats = parse_force_stats_report(force_stats_path)

            all_data.append({
                'ch_distance': ch_distance,
                'dir_name': folder,
                'data': data,
                'force_stats': force_stats,
            })
            peaks_info = f", {force_stats['n_peaks']} peaks" if force_stats else ""
            print(f"  Loaded {folder}: {data['frames_analyzed']} frames, "
                  f"{len(data['categories'])} categories{peaks_info}")
        else:
            print(f"  Skipped {folder}: no coordination analysis found")

    # Sort by C-H distance
    all_data.sort(key=lambda x: x['ch_distance'])
    return all_data


def process_coordination_data(all_data):
    """
    Process collected data to compute binding mode fractions.

    Returns:
        dict with:
            - 'ch_distances': array of C-H distances
            - 'c_modes': dict mapping mode -> array of fractions
            - 'h_modes': dict mapping mode -> array of fractions
            - 'c_mode_errors': dict mapping mode -> array of standard errors
            - 'h_mode_errors': dict mapping mode -> array of standard errors
            - 'combined_modes': dict mapping combined mode -> array of fractions
            - 'mean_forces': array of mean forces
            - 'force_errors': array of force standard errors
    """
    modes = ['unbound', 'top', 'bridge', 'hollow']

    ch_distances = []
    c_modes = {m: [] for m in modes}
    h_modes = {m: [] for m in modes}
    c_mode_errors = {m: [] for m in modes}
    h_mode_errors = {m: [] for m in modes}
    combined_modes = defaultdict(list)
    mean_forces = []
    force_errors = []

    for entry in all_data:
        ch_distances.append(entry['ch_distance'])
        categories = entry['data']['categories']
        n_frames = entry['data']['frames_analyzed']

        # Aggregate C and H mode fractions
        c_fracs = {m: 0.0 for m in modes}
        h_fracs = {m: 0.0 for m in modes}

        for cat_name, cat_data in categories.items():
            # Parse category name (e.g., "C-top_H-bridge")
            # Use [a-z]+ instead of \w+ to avoid matching underscore
            c_match = re.search(r'C-([a-z]+)', cat_name)
            h_match = re.search(r'H-([a-z]+)', cat_name)

            if c_match and h_match:
                c_mode = c_match.group(1)
                h_mode = h_match.group(1)
                frac = cat_data['fraction']

                if c_mode in c_fracs:
                    c_fracs[c_mode] += frac
                if h_mode in h_fracs:
                    h_fracs[h_mode] += frac

                combined_modes[cat_name].append(frac)

        # Store fractions and compute binomial standard errors
        for mode in modes:
            c_modes[mode].append(c_fracs[mode])
            h_modes[mode].append(h_fracs[mode])

            # Standard error for proportion: sqrt(p*(1-p)/n)
            c_se = np.sqrt(c_fracs[mode] * (1 - c_fracs[mode]) / n_frames) if n_frames > 0 else 0
            h_se = np.sqrt(h_fracs[mode] * (1 - h_fracs[mode]) / n_frames) if n_frames > 0 else 0
            c_mode_errors[mode].append(c_se)
            h_mode_errors[mode].append(h_se)

        # Compute weighted mean force and error
        total_force = 0.0
        total_var = 0.0
        for cat_data in categories.values():
            total_force += cat_data['fraction'] * cat_data['mean_force']
            # Weighted variance contribution
            total_var += cat_data['fraction'] * cat_data['std_dev']**2

        mean_forces.append(total_force)
        force_errors.append(np.sqrt(total_var / n_frames) if n_frames > 0 else 0)

    # Pad combined modes for missing entries
    n_points = len(ch_distances)
    for key in combined_modes:
        while len(combined_modes[key]) < n_points:
            combined_modes[key].append(0.0)

    return {
        'ch_distances': np.array(ch_distances),
        'c_modes': {k: np.array(v) for k, v in c_modes.items()},
        'h_modes': {k: np.array(v) for k, v in h_modes.items()},
        'c_mode_errors': {k: np.array(v) for k, v in c_mode_errors.items()},
        'h_mode_errors': {k: np.array(v) for k, v in h_mode_errors.items()},
        'combined_modes': dict(combined_modes),
        'mean_forces': np.array(mean_forces),
        'force_errors': np.array(force_errors),
    }


def load_pmf_data():
    """
    Load PMF data from pmf_analysis_results.txt if available.

    Returns:
        dict with 'ch_distances', 'mean_forces', 'force_errors' or None
    """
    pmf_file = 'pmf_analysis_results.txt'
    if not os.path.exists(pmf_file):
        return None

    ch_distances = []
    mean_forces = []
    force_errors = []

    with open(pmf_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    ch_dist = float(parts[0])
                    mean_f = float(parts[1])
                    se = float(parts[2])
                    ch_distances.append(ch_dist)
                    mean_forces.append(mean_f)
                    force_errors.append(se)
                except ValueError:
                    continue

    if ch_distances:
        return {
            'ch_distances': np.array(ch_distances),
            'mean_forces': np.array(mean_forces),
            'force_errors': np.array(force_errors),
        }
    return None


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_coordination_vs_reaction_coordinate(processed_data, pmf_data, output_dir):
    """
    Create the key 3-panel figure:
    - Panel 1: PMF plot (mean force vs C-H distance)
    - Panel 2: C binding mode fractions
    - Panel 3: H binding mode fractions
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    modes = ['unbound', 'top', 'bridge', 'hollow']
    x = processed_data['ch_distances']

    # Panel 1: PMF plot
    ax1 = axes[0]
    if pmf_data is not None:
        pmf_x = pmf_data['ch_distances']
        pmf_y = pmf_data['mean_forces']
        pmf_err = pmf_data['force_errors']
        ax1.errorbar(pmf_x, pmf_y, yerr=pmf_err, fmt='ko-', capsize=3,
                     markersize=6, linewidth=1.5, capthick=1)
    else:
        # Use coordination-derived force data
        ax1.errorbar(x, processed_data['mean_forces'],
                     yerr=processed_data['force_errors'],
                     fmt='ko-', capsize=3, markersize=6, linewidth=1.5, capthick=1)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax1.set_ylabel('Mean Force (eV/A)', fontsize=LABEL_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE,
                    length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    ax1.set_title('Potential of Mean Force', fontsize=TITLE_FONTSIZE)
    add_rc_markers(ax1)

    # Panel 2: C coordination
    ax2 = axes[1]
    for mode in modes:
        y = processed_data['c_modes'][mode]
        yerr = processed_data['c_mode_errors'][mode]
        ax2.errorbar(x, y, yerr=yerr, fmt=f'{MODE_MARKERS[mode]}-',
                     color=MODE_COLORS[mode], label=mode,
                     capsize=2, markersize=5, linewidth=1.2, capthick=1)

    ax2.set_ylabel('Fraction', fontsize=LABEL_FONTSIZE)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE,
                    length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=LEGEND_FONTSIZE)
    ax2.set_title('Carbon Coordination to Pt', fontsize=TITLE_FONTSIZE)
    add_rc_markers(ax2)

    # Panel 3: H coordination
    ax3 = axes[2]
    for mode in modes:
        y = processed_data['h_modes'][mode]
        yerr = processed_data['h_mode_errors'][mode]
        ax3.errorbar(x, y, yerr=yerr, fmt=f'{MODE_MARKERS[mode]}-',
                     color=MODE_COLORS[mode], label=mode,
                     capsize=2, markersize=5, linewidth=1.2, capthick=1)

    ax3.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel('Fraction', fontsize=LABEL_FONTSIZE)
    ax3.set_ylim(-0.05, 1.05)
    ax3.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE,
                    length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=LEGEND_FONTSIZE)
    ax3.set_title('Hydrogen Coordination to Pt', fontsize=TITLE_FONTSIZE)
    add_rc_markers(ax3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coordination_vs_reaction_coordinate.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_mean_coordination(processed_data, output_dir):
    """Plot mean coordination numbers vs reaction coordinate."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = processed_data['ch_distances']
    modes = ['unbound', 'top', 'bridge', 'hollow']
    cn_values = [0, 1, 2, 3]

    # Calculate mean CN for C and H
    c_mean_cn = np.zeros_like(x)
    h_mean_cn = np.zeros_like(x)

    for mode, cn in zip(modes, cn_values):
        c_mean_cn += cn * processed_data['c_modes'][mode]
        h_mean_cn += cn * processed_data['h_modes'][mode]

    ax.plot(x, c_mean_cn, 'o-', color='#1f77b4', label='C-Pt CN', markersize=6, linewidth=1.5)
    ax.plot(x, h_mean_cn, 's--', color='#d62728', label='H-Pt CN', markersize=6, linewidth=1.5)

    ax.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Mean Coordination Number', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE,
                   length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['0 (unbound)', '1 (top)', '2 (bridge)', '3 (hollow)'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_coordination_vs_rc.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_binding_mode_fractions(processed_data, output_dir):
    """Create stacked area plots for C and H binding mode fractions."""
    modes = ['unbound', 'top', 'bridge', 'hollow']
    x = processed_data['ch_distances']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # C binding modes
    ax1 = axes[0]
    c_stack = np.vstack([processed_data['c_modes'][m] for m in modes])
    ax1.stackplot(x, c_stack, labels=modes,
                  colors=[MODE_COLORS[m] for m in modes], alpha=0.8)
    ax1.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Fraction', fontsize=LABEL_FONTSIZE)
    ax1.set_title('Carbon Binding Modes', fontsize=TITLE_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    ax1.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
    ax1.set_ylim(0, 1)

    # H binding modes
    ax2 = axes[1]
    h_stack = np.vstack([processed_data['h_modes'][m] for m in modes])
    ax2.stackplot(x, h_stack, labels=modes,
                  colors=[MODE_COLORS[m] for m in modes], alpha=0.8)
    ax2.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('Fraction', fontsize=LABEL_FONTSIZE)
    ax2.set_title('Hydrogen Binding Modes', fontsize=TITLE_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    ax2.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binding_mode_fractions.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()

    # Also save individual plots
    for name, data_key, title in [('c_binding_mode_fractions', 'c_modes', 'Carbon'),
                                   ('h_binding_mode_fractions', 'h_modes', 'Hydrogen')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        stack = np.vstack([processed_data[data_key][m] for m in modes])
        ax.stackplot(x, stack, labels=modes,
                     colors=[MODE_COLORS[m] for m in modes], alpha=0.8)
        ax.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Fraction', fontsize=LABEL_FONTSIZE)
        ax.set_title(f'{title} Binding Modes', fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
        ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=DPI, bbox_inches='tight')
        plt.close()


def plot_combined_mode_heatmap(processed_data, all_data, output_dir):
    """Create heatmap showing combined C-H binding mode populations."""
    x = processed_data['ch_distances']

    # Get all unique combined modes
    all_modes = set()
    for entry in all_data:
        all_modes.update(entry['data']['categories'].keys())
    all_modes = sorted(all_modes)

    # Build matrix
    matrix = np.zeros((len(all_modes), len(x)))
    for i, entry in enumerate(all_data):
        for j, mode in enumerate(all_modes):
            if mode in entry['data']['categories']:
                matrix[j, i] = entry['data']['categories'][mode]['fraction']

    fig, ax = plt.subplots(figsize=(12, max(6, len(all_modes) * 0.4)))

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([f'{d:.2f}' for d in x], rotation=45, ha='right', fontsize=TICK_LABELSIZE-2)
    ax.set_yticks(range(len(all_modes)))
    ax.set_yticklabels(all_modes, fontsize=TICK_LABELSIZE-2)

    ax.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Binding Mode', fontsize=LABEL_FONTSIZE)
    ax.set_title('Combined C-H Binding Mode Populations', fontsize=TITLE_FONTSIZE)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction', fontsize=LABEL_FONTSIZE)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_modes_heatmap.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()


def analyze_bimodality(all_data, output_dir):
    """
    Analyze bimodality using peak detection from force histograms.

    Primary metric: Number of peaks detected in force histogram (from force_stats_report.txt)
    Secondary metric: Force spread between binding modes

    Returns dict with bimodality metrics for each RC point.
    """
    bimodality_data = []

    for entry in all_data:
        ch_dist = entry['ch_distance']
        categories = entry['data']['categories']
        n_frames = entry['data']['frames_analyzed']
        force_stats = entry.get('force_stats')

        # PRIMARY: Peak-based classification from histogram analysis
        n_peaks = 0
        peaks = []
        peak_forces = []
        if force_stats and force_stats['n_peaks'] > 0:
            n_peaks = force_stats['n_peaks']
            peaks = force_stats['peaks']
            peak_forces = [p['force'] for p in peaks]

        # SECONDARY: Force spread between binding modes (for reference)
        significant_cats = {k: v for k, v in categories.items() if v['fraction'] > 0.05}
        if len(significant_cats) >= 2:
            forces = [v['mean_force'] for v in significant_cats.values()]
            force_spread = max(forces) - min(forces)
        else:
            force_spread = 0.0

        # Classification based on NUMBER OF PEAKS (true multimodality)
        if n_peaks >= 3:
            classification = "Multimodal"
        elif n_peaks == 2:
            classification = "Bimodal"
        elif n_peaks == 1:
            classification = "Unimodal"
        else:
            classification = "Unknown"

        bimodality_data.append({
            'ch_distance': ch_dist,
            'n_peaks': n_peaks,
            'peaks': peaks,
            'peak_forces': peak_forces,
            'classification': classification,
            'force_spread': force_spread,  # Secondary metric
            'n_binding_modes': len(significant_cats),
            'modes': significant_cats,
        })

    return bimodality_data


def plot_bimodality_analysis(bimodality_data, all_data, output_dir):
    """Plot bimodality analysis: peak count and force by binding mode at each RC point."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ch_dists = [d['ch_distance'] for d in bimodality_data]
    n_peaks_list = [d['n_peaks'] for d in bimodality_data]

    # Panel 1: Number of peaks (true multimodality indicator)
    ax1 = axes[0]
    # Color bars by classification
    bar_colors = ['#d62728' if n >= 2 else '#1f77b4' for n in n_peaks_list]
    ax1.bar(ch_dists, n_peaks_list, width=0.08, color=bar_colors, edgecolor='black')
    ax1.set_ylabel('Number of Peaks', fontsize=LABEL_FONTSIZE)
    ax1.set_title('Multimodality: Number of Peaks in Force Histogram', fontsize=TITLE_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    ax1.axhline(1.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['0', '1 (Unimodal)', '2 (Bimodal)', '3+ (Multimodal)'])
    add_rc_markers(ax1)

    # Panel 2: Forces by binding mode at each RC
    ax2 = axes[1]

    # Collect all unique modes
    all_modes = set()
    for entry in all_data:
        all_modes.update(entry['data']['categories'].keys())

    # Create color map for modes
    mode_list = sorted(all_modes)
    colors = plt.cm.tab10(np.linspace(0, 1, len(mode_list)))
    mode_color_map = {m: colors[i] for i, m in enumerate(mode_list)}

    # Plot forces for each mode at each RC
    for entry in all_data:
        ch_dist = entry['ch_distance']
        for cat_name, cat_data in entry['data']['categories'].items():
            if cat_data['fraction'] > 0.05:  # Only plot significant modes
                ax2.scatter(ch_dist, cat_data['mean_force'],
                           s=cat_data['fraction'] * 500,  # Size proportional to population
                           c=[mode_color_map[cat_name]],
                           alpha=0.7, edgecolors='black', linewidth=0.5)

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('Mean Force (eV/A)', fontsize=LABEL_FONTSIZE)
    ax2.set_title('Mean Force by Binding Mode (size = population)', fontsize=TITLE_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    add_rc_markers(ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bimodality_analysis.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()

    # Also create a detailed force-by-mode plot
    fig, ax = plt.subplots(figsize=(14, 6))

    for entry in all_data:
        ch_dist = entry['ch_distance']
        cats = entry['data']['categories']

        # Sort by fraction
        sorted_cats = sorted(cats.items(), key=lambda x: -x[1]['fraction'])

        for i, (cat_name, cat_data) in enumerate(sorted_cats):
            if cat_data['fraction'] > 0.01:  # Show modes with >1% population
                offset = (i - len(sorted_cats)/2) * 0.02
                ax.errorbar(ch_dist + offset, cat_data['mean_force'],
                           yerr=cat_data['std_dev'],
                           fmt='o', markersize=max(3, cat_data['fraction'] * 20),
                           color=mode_color_map[cat_name],
                           alpha=0.7, capsize=2)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('C-H Distance (A)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Mean Force (eV/A)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Force Distribution by Binding Mode Along Reaction Coordinate', fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)

    # Create legend with mode names
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=mode_color_map[m], markersize=8, label=m)
                       for m in mode_list if any(m in entry['data']['categories']
                                                  and entry['data']['categories'][m]['fraction'] > 0.05
                                                  for entry in all_data)]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=LEGEND_FONTSIZE-2, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'force_by_mode_detailed.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()


def write_summary(processed_data, all_data, bimodality_data, output_dir):
    """Write text summary of coordination evolution."""
    summary_path = os.path.join(output_dir, 'coordination_evolution_summary.txt')

    modes = ['unbound', 'top', 'bridge', 'hollow']
    x = processed_data['ch_distances']

    with open(summary_path, 'w') as f:
        f.write("Coordination Evolution Along Reaction Coordinate\n")
        f.write("=" * 60 + "\n\n")

        # Summary table
        f.write("C-H Dist (A)  | C-Pt Modes (%)                    | H-Pt Modes (%)\n")
        f.write("              | unb    top    brg    hol          | unb    top    brg    hol\n")
        f.write("-" * 80 + "\n")

        for i, dist in enumerate(x):
            c_vals = [processed_data['c_modes'][m][i] * 100 for m in modes]
            h_vals = [processed_data['h_modes'][m][i] * 100 for m in modes]

            f.write(f"{dist:12.2f}  | {c_vals[0]:5.1f}  {c_vals[1]:5.1f}  {c_vals[2]:5.1f}  {c_vals[3]:5.1f}  "
                    f"        | {h_vals[0]:5.1f}  {h_vals[1]:5.1f}  {h_vals[2]:5.1f}  {h_vals[3]:5.1f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Key Observations:\n\n")

        # Identify key transitions
        h_top_frac = processed_data['h_modes']['top']
        h_bridge_frac = processed_data['h_modes']['bridge']

        # Find where H becomes mostly coordinated
        h_coordinated = 1 - processed_data['h_modes']['unbound']
        for i, (dist, frac) in enumerate(zip(x, h_coordinated)):
            if frac > 0.5:
                f.write(f"- H becomes predominantly coordinated to Pt at C-H ~ {dist:.2f} A\n")
                break

        # Find dominant mode at each key point
        f.write(f"\n- Initial state (C-H = {x[0]:.2f} A):\n")
        f.write(f"    C: {max(modes, key=lambda m: processed_data['c_modes'][m][0])} dominant\n")
        f.write(f"    H: {max(modes, key=lambda m: processed_data['h_modes'][m][0])} dominant\n")

        f.write(f"\n- Final state (C-H = {x[-1]:.2f} A):\n")
        f.write(f"    C: {max(modes, key=lambda m: processed_data['c_modes'][m][-1])} dominant\n")
        f.write(f"    H: {max(modes, key=lambda m: processed_data['h_modes'][m][-1])} dominant\n")

        # Bimodality analysis
        f.write("\n" + "=" * 60 + "\n")
        f.write("MULTIMODALITY ANALYSIS (Peak-Based Classification)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Classification based on number of peaks detected in force histogram\n")
        f.write("(from scipy.signal.find_peaks on smoothed histogram)\n\n")

        f.write(f"{'C-H Dist (A)':<14} {'# Peaks':<10} {'Peak Forces (eV/A)':<30} {'Classification'}\n")
        f.write("-" * 80 + "\n")

        for bd in bimodality_data:
            n_peaks = bd['n_peaks']
            classification = bd['classification']
            peak_forces_str = ", ".join(f"{f:.2f}" for f in bd['peak_forces']) if bd['peak_forces'] else "N/A"

            f.write(f"{bd['ch_distance']:<14.2f} {n_peaks:<10} {peak_forces_str:<30} {classification}\n")

        # Find bimodal points
        bimodal_points = [bd for bd in bimodality_data if bd['n_peaks'] >= 2]
        if bimodal_points:
            f.write(f"\nBimodal/Multimodal points detected: {len(bimodal_points)}\n")
            for bd in bimodal_points:
                f.write(f"  C-H = {bd['ch_distance']:.2f} A: {bd['n_peaks']} peaks at {', '.join(f'{f:.2f}' for f in bd['peak_forces'])} eV/A\n")

        # Also show force spread as secondary metric
        f.write("\n" + "-" * 60 + "\n")
        f.write("Secondary Metric: Force Spread Between Binding Modes\n")
        f.write("-" * 60 + "\n\n")

        f.write(f"{'C-H Dist (A)':<14} {'# Bind Modes':<14} {'Force Spread (eV/A)'}\n")
        f.write("-" * 50 + "\n")

        for bd in bimodality_data:
            f.write(f"{bd['ch_distance']:<14.2f} {bd['n_binding_modes']:<14} {bd['force_spread']:.3f}\n")

    print(f"  Saved: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Coordination Evolution Along Reaction Coordinate")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect data
    print("\nCollecting coordination data from simulation directories...")
    all_data = collect_coordination_data()

    if not all_data:
        print("ERROR: No coordination data found!")
        return

    print(f"\nFound {len(all_data)} reaction coordinate points")

    # Process data
    print("\nProcessing coordination data...")
    processed_data = process_coordination_data(all_data)

    # Load PMF data if available
    print("\nLoading PMF data...")
    pmf_data = load_pmf_data()
    if pmf_data:
        print(f"  Loaded {len(pmf_data['ch_distances'])} PMF data points")
    else:
        print("  PMF data not found, using coordination-derived forces")

    # Generate plots
    print("\nGenerating plots...")

    plot_coordination_vs_reaction_coordinate(processed_data, pmf_data, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/coordination_vs_reaction_coordinate.png")

    plot_mean_coordination(processed_data, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/mean_coordination_vs_rc.png")

    plot_binding_mode_fractions(processed_data, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/binding_mode_fractions.png")
    print(f"  Saved: {OUTPUT_DIR}/c_binding_mode_fractions.png")
    print(f"  Saved: {OUTPUT_DIR}/h_binding_mode_fractions.png")

    plot_combined_mode_heatmap(processed_data, all_data, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/combined_modes_heatmap.png")

    # Bimodality analysis
    print("\nAnalyzing bimodality...")
    bimodality_data = analyze_bimodality(all_data, OUTPUT_DIR)

    plot_bimodality_analysis(bimodality_data, all_data, OUTPUT_DIR)
    print(f"  Saved: {OUTPUT_DIR}/bimodality_analysis.png")
    print(f"  Saved: {OUTPUT_DIR}/force_by_mode_detailed.png")

    # Write summary
    print("\nWriting summary...")
    write_summary(processed_data, all_data, bimodality_data, OUTPUT_DIR)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
