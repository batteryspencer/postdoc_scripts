#!/usr/bin/env python3
"""
compile_multi_node_results.py

Aggregate and analyze VASP benchmark results across different node counts.

Outputs:
  - multi_node_results.csv       : all configs + runtimes
  - multi_node_summary.txt       : best config per node count
  - multi_node_plot.png          : bar chart of best runtime vs nodes
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    base = Path('.')
    all_dfs = []

    # Gather each node directory
    for node_dir in sorted(base.glob('nodes=*')):
        n = node_dir.name.split('=', 1)[1]
        csv_path = node_dir / 'benchmark_results.csv'
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found; skipping.")
            continue
        df = pd.read_csv(csv_path)
        df['nodes'] = int(n)
        all_dfs.append(df)

    if not all_dfs:
        sys.exit("No benchmark_results.csv files found under nodes=*/")

    # Combine
    big_df = pd.concat(all_dfs, ignore_index=True)
    big_df.to_csv('multi_node_results.csv', index=False)
    print("Wrote multi_node_results.csv")

    # Pick best per node count
    idx = big_df.groupby('nodes')['runtime_per_step_s'].idxmin()
    best_df = big_df.loc[idx].sort_values('nodes')
    
    # Write summary text
    with open('multi_node_summary.txt', 'w') as f:
        f.write("Best configuration per node count:\n")
        for _, row in best_df.iterrows():
            f.write(
                f"Nodes={int(row['nodes'])}: "
                f"NCORE={int(row['NCORE'])}, "
                f"NPAR={int(row['NPAR'])}, "
                f"KPAR={int(row['KPAR'])} -> "
                f"{row['runtime_per_step_s']:.2f} s/step\n"
            )
    print("Wrote multi_node_summary.txt")

    # Plot: runtime vs nodes and compute cost per step
    CORES_PER_NODE = 192

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # Top panel: runtime per step
    bars1 = ax1.bar(best_df['nodes'].astype(str), best_df['runtime_per_step_s'])
    # ax1.set_xlabel("Number of nodes")  # Remove x-axis label from top plot
    ax1.set_ylabel("Seconds per step")
    ax1.set_title("Runtime per step")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * h,
                 f"{h:.2f}s", ha='center', va='bottom', fontsize=8)

    # Bottom panel: compute cost per step (core·seconds)
    cost = best_df['runtime_per_step_s'] * best_df['nodes'] * CORES_PER_NODE
    bars2 = ax2.bar(best_df['nodes'].astype(str), cost)
    ax2.set_xlabel("Number of nodes")
    ax2.set_ylabel("Core·seconds per step")
    ax2.set_title("Compute cost per step")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * h,
                 f"{h:.0f}", ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig('multi_node_plot.png', dpi=300)
    print("Wrote multi_node_plot.png")

if __name__ == '__main__':
    main()