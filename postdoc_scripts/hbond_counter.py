#!/usr/bin/env python3
"""
Count H-bonds frame-by-frame in a VASP XDATCAR trajectory and plot results.

Usage
-----
python hbond_counter.py POSCAR XDATCAR  \
    --dcut 3.5  --acut 35  \
    --outfile hbonds_per_frame.csv

Requirements
------------
MDAnalysis >= 2.6  (has native VASP/XDATCAR reader)
matplotlib, pandas
"""

from ase.io import read
import numpy as np
from ase.neighborlist import NeighborList
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

# Workaround for cross-environment pickle compatibility (numpy._core alias)
import numpy as _np
import sys as _sys
_sys.modules['numpy._core'] = _np.core

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1
PS_TO_FS = 1000  # Conversion factor from picoseconds to femtoseconds
LONG_MA_WINDOW = 500    # frames for long moving average

def main():
    # Hardcoded inputs and parameters
    poscar = "POSCAR"
    xdatcar = "XDATCAR"
    dcut = 3.5         # O...O distance cutoff (Å)
    acut = 40         # angle cutoff (degrees)
    oh_bond_cut = 1.2  # O-H bond cutoff to identify donor hydrogens (Å)

    # Output log file
    log_file = open("hbond_output_log.txt", "w")

    # Cache settings for H-bond analysis (specific to angle cutoff)
    cache_file = f"hbond_cache_acut_{acut}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        df = data["df"]
        max_angles_all = data["max_angles_all"]
        print("Loaded cached H-bond data", file=log_file)
    else:
        # Prepare to compute H-bond data
        max_angles_all = []  # collect max O–H⋯O angle per donor H (distance-based only)

    if not (os.path.exists(cache_file)):
        # Load initial structure to get atom types
        initial = read(poscar)
        symbols = initial.get_chemical_symbols()
        # Identify indices for oxygens and hydrogens
        oxy_idx = [i for i, el in enumerate(symbols) if el == "O"]
        hyd_idx = [i for i, el in enumerate(symbols) if el == "H"]
        # Number of water molecules (one donor O per water)
        n_water = len(oxy_idx)

        # Prepare cutoffs for ASE neighbor lists (PBC-aware)
        cutoffs_oh = [oh_bond_cut if el == "O" else 0.0 for el in symbols]
        cutoffs_acc = [dcut if el == "O" else 0.0 for el in symbols]

        # Read VASP XDATCAR trajectory using ASE VASP reader
        traj = read(xdatcar, index=":", format="vasp-xdatcar")
        hb_counts = []

        for atoms in traj:
            atoms.set_pbc([True, True, True])
            cell = atoms.get_cell()
            # Build neighbor lists for OH donors and O acceptors
            nl_oh = NeighborList(cutoffs_oh, bothways=True, self_interaction=False)
            nl_oh.update(atoms)
            nl_acc = NeighborList(cutoffs_acc, bothways=True, self_interaction=False)
            nl_acc.update(atoms)

            count = 0
            # For each frame, we will track max angle per donor H
            for idxO in oxy_idx:
                h_nbrs, h_shifts = nl_oh.get_neighbors(idxO)
                for h_idx, h_shift in zip(h_nbrs, h_shifts):
                    if h_idx not in hyd_idx:
                        continue
                    pos_O = atoms.positions[idxO]
                    pos_H = atoms.positions[h_idx] + np.dot(h_shift, cell)
                    # Gather all angles for this donor H
                    h_angles = []
                    o_nbrs, o_shifts = nl_acc.get_neighbors(h_idx)
                    for o_idx, o_shift in zip(o_nbrs, o_shifts):
                        if o_idx not in oxy_idx or o_idx == idxO:
                            continue
                        pos_O2 = atoms.positions[o_idx] + np.dot(o_shift, cell)
                        v1 = pos_O - pos_H
                        v2 = pos_O2 - pos_H
                        angle = np.degrees(
                            np.arccos(
                                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            )
                        )
                        h_angles.append(angle)
                    # After checking all neighbors, record the maximum angle found (if any)
                    if h_angles:
                        max_angle = max(h_angles)
                        max_angles_all.append(max_angle)
                        # Count an H-bond if the most linear angle passes the cutoff
                        if max_angle > (180 - acut):
                            count += 1
            hb_counts.append(count)

        # Save results
        df = pd.DataFrame({"frame": range(len(hb_counts)), "n_hbonds": hb_counts})
        # Convert frame index to time in ps and compute moving average
        df['time_ps'] = df['frame'] / PS_TO_FS

    # Compute only long moving average
    df['hbonds_ma_long'] = df['n_hbonds'].rolling(window=LONG_MA_WINDOW).mean()

    # Import energy MA data and compute H-bond MA for correlation
    energy_df = pd.read_csv("moving_average_3000_steps.csv")
    # Compute H-bond moving average with window = 3000 frames
    df['hbonds_ma_3000'] = df['n_hbonds'].rolling(window=3000).mean()
    # Keep only rows where H-bond MA actually changed
    df_hb = df.loc[df['hbonds_ma_3000'].diff().abs() > 1e-6]

    # Merge the two MA series on nearest time_ps
    ma_merge = pd.merge_asof(
        energy_df.sort_values('time_ps'),
        df_hb[['time_ps', 'hbonds_ma_3000']].sort_values('time_ps'),
        on='time_ps', direction='nearest', tolerance=0.0005  # 0.5 fs guard
    )
    # Drop any rows where either MA is NaN
    ma_merge = ma_merge.dropna(subset=['hbonds_ma_3000', 'moving_average_energy_eV'])
    # Perform linear regression fit
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        ma_merge['hbonds_ma_3000'], ma_merge['moving_average_energy_eV']
    )
    # Correlation scatter with styled plot
    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()
    ax.grid(True, ls='--', lw=0.5, alpha=0.6)

    # Scatter plot of down-sampled data
    ax.scatter(
        ma_merge['hbonds_ma_3000'][::10],
        ma_merge['moving_average_energy_eV'][::10],
        s=6,
        alpha=0.6,
        edgecolors='none',
        label='Data'
    )

    # Regression line
    x_line = np.array([ma_merge['hbonds_ma_3000'].min(),
                    ma_merge['hbonds_ma_3000'].max()])
    ax.plot(x_line, intercept + slope * x_line,
            color='C3', lw=2,
            label=f"Slope = {slope:.3f} eV/bond\n$R^2$ = {r_value**2:.2f}")

    # Labels
    ax.set_xlabel("H-bonds MA (3000 frames)", fontsize=14)
    ax.set_ylabel("Energy MA (3000 frames) / eV", fontsize=14)
    ax.set_title("Energy vs H-bond Correlation", fontsize=16)
    ax.tick_params(axis='both', labelsize=11)

    # Legend outside
    ax.legend(fontsize=11, frameon=False,
            loc='lower left')

    plt.tight_layout()
    plt.savefig("energy_vs_hbonds_correlation.png", dpi=300, bbox_inches='tight')

    # Save computed H-bond data to cache
    with open(cache_file, "wb") as f:
        pickle.dump({"df": df, "max_angles_all": max_angles_all}, f)
    print("Saved cached H-bond data", file=log_file)

    # Debug: print ranges
    print(f"Frames: {len(df)}", file=log_file)
    print(f"Frames: {len(df)}")
    print(f"time_ps range: {df['time_ps'].min()} to {df['time_ps'].max()}", file=log_file)
    print(f"time_ps range: {df['time_ps'].min()} to {df['time_ps'].max()}")
    print(f"n_hbonds range: {df['n_hbonds'].min()} to {df['n_hbonds'].max()}", file=log_file)
    print(f"n_hbonds range: {df['n_hbonds'].min()} to {df['n_hbonds'].max()}")

    # Compare average H-bonds in first and last 1 ps
    first_chunk = df[(df['time_ps'] >= 0) & (df['time_ps'] < 1)]
    t_end = df['time_ps'].max()
    last_chunk = df[(df['time_ps'] >= t_end - 1) & (df['time_ps'] <= t_end)]

    first_avg = first_chunk['n_hbonds'].mean()
    last_avg = last_chunk['n_hbonds'].mean()
    delta = last_avg - first_avg

    print(f"Average H-bonds in first 1 ps: {first_avg:.1f}", file=log_file)
    print(f"Average H-bonds in first 1 ps: {first_avg:.1f}")
    print(f"Average H-bonds in last 1 ps: {last_avg:.1f}", file=log_file)
    print(f"Average H-bonds in last 1 ps: {last_avg:.1f}")
    print(f"Net change (last - first): {delta:.1f} H-bonds", file=log_file)
    print(f"Net change (last - first): {delta:.1f} H-bonds")

    plt.figure()

    # Plot: raw counts and long moving average vs time
    plt.plot(df['time_ps'], df['n_hbonds'], lw=0.8, alpha=0.2, color='black', label='Raw')
    if acut < 180:
        plt.plot(df['time_ps'], df['hbonds_ma_long'], lw=3, color='#852d37', label=f"{LONG_MA_WINDOW}-frame MA")
    else:
        plt.plot(df['time_ps'], df['hbonds_ma_long'], lw=3, color='#284366', label=f"{LONG_MA_WINDOW}-frame MA")
    plt.xlabel("Time (ps)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Total number of H-bonds", fontsize=LABEL_FONTSIZE)
    if acut < 180:
        plt.title(f"Near-linear H-bonds vs. time ($\\theta_{{\\mathrm{{bend}}}}^{{c}}$ = {acut}°)", fontsize=TITLE_FONTSIZE-3)
    else:
        plt.title(f"Distance-only H-bonds vs. time ($\\theta_{{\\mathrm{{bend}}}}^{{c}}$ = {acut}°)", fontsize=TITLE_FONTSIZE-3)
    plt.tick_params(
        axis='both',
        which='major',
        labelsize=TICK_LABELSIZE,
        length=TICK_LENGTH_MAJOR,
        width=TICK_WIDTH_MAJOR
    )
    plt.annotate(f"Start: {first_avg:.1f}", xy=(0.5, first_avg), xytext=(1, first_avg + 5),
                 arrowprops=dict(arrowstyle="->"), fontsize=12)
    plt.annotate(f"End: {last_avg:.1f}", xy=(t_end - 0.5, last_avg), xytext=(t_end - 1.5, last_avg + 5),
                arrowprops=dict(arrowstyle="->"), fontsize=12)
    if acut < 180:
        delta_y = (first_avg + last_avg) / 2 + 10
    else:
        delta_y = (first_avg + last_avg) / 2 + 2  # lower placement for acut=180
    delta_x = t_end / 2
    plt.annotate(f"Δ = {delta:.1f}", xy=(delta_x, (first_avg + last_avg) / 2), xytext=(delta_x, delta_y),
                 fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    plt.tight_layout()
    plt.savefig(f"hbonds_acut_{acut}.png", dpi=300)

    plt.figure()
    bins = np.arange(0, 181, 1)
    acut_rec = 40
    # Compute and print percentage of max angles above cutoff
    threshold = 180 - acut_rec
    above_count = sum(1 for angle in max_angles_all if angle > threshold)
    total = len(max_angles_all)
    pct = above_count / total * 100 if total > 0 else 0

    print(f"Percentage of max O–H⋯O angles above {threshold}°: {pct:.1f}%", file=log_file)
    print(f"Percentage of max O–H⋯O angles above {threshold}°: {pct:.1f}%")
    # Normalize counts by number of frames for average per-frame frequency
    n_frames = len(df)
    weights = np.ones(len(max_angles_all)) / n_frames
    plt.hist(max_angles_all, bins=bins, weights=weights, color='C0', edgecolor='k', alpha=0.7)
    plt.axvline(threshold, color='C3', linestyle='--', linewidth=2,
                label=f"Cutoff ({threshold}°)")
    plt.xlabel("Max ∠(O–H···O) (°)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Average count per frame", fontsize=LABEL_FONTSIZE)
    plt.title("Max O–H⋯O angle distribution (O…O ≤ 3.5 Å)", fontsize=TITLE_FONTSIZE-2)
    plt.legend(fontsize=LEGEND_FONTSIZE)

    # Annotate percentages below and above the cutoff line (no arrows)
    plt.text(threshold - 25, 1.3, f"{100 - pct:.1f}%", fontsize=12,
             ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="#f4cccc", alpha=0.85))
    plt.text(threshold + 25, 1.0, f"{pct:.1f}%", fontsize=12,
             ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="#d9ead3", alpha=0.85))

    plt.tight_layout()
    plt.savefig("angle_hist_max_angles.png", dpi=300)

    log_file.close()

if __name__ == "__main__":
    main()
