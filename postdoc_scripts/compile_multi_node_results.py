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

def main():
    base = Path('.')
    all_dfs = []

    # Gather each node directory
    for node_dir in sorted(base.glob('nodes=*')):
        n = node_dir.name.split('=', 1)[1]
        csv_path = node_dir / 'benchmark_folders' / 'benchmark_results.csv'
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found; skipping.")
            continue
        df = pd.read_csv(csv_path)
        df['nodes'] = int(n)
        all_dfs.append(df)

    if not all_dfs:
        sys.exit("No benchmark_results.csv files found under nodes=*/benchmark_folders/")

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

    # Plot best runtime vs nodes with config annotations
    plt.figure(figsize=(6, 4))
    bars = plt.bar(best_df['nodes'].astype(str), best_df['runtime_per_step_s'])
    plt.xlabel("Number of nodes")
    plt.ylabel("Seconds per step (best config)")
    plt.title("Best VASP Benchmark Runtime vs Nodes")

    # Annotate each bar with runtime only
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.2f}s"
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01*height,
                 label, ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('multi_node_plot.png', dpi=300)
    print("Wrote multi_node_plot.png")

if __name__ == '__main__':
    main()