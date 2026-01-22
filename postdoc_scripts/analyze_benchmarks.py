#!/usr/bin/env python3
"""analyze_benchmarks.py - fast VASP benchmark analyzer using bash one-liners"""

import argparse
import statistics
import sys

import pandas as pd
import matplotlib.pyplot as plt

import re
import pathlib

def extract_runtimes(folder):
    """
    Extract all '(xxx seconds)' entries from job*.out files using grep.
    Discard the first 20% of entries or first 20 entries, whichever is smaller.
    Returns a list of floats.
    """
    import subprocess
    cmd = f'grep -h -oP "(?<=\\()\\d+(?:\\.\\d+)?(?= seconds\\))" "{folder}"/job*.out'
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout:
        return []
    # parse all matched times
    times = [float(x) for x in proc.stdout.split()]
    # discard warm-up entries
    skip = min(int(len(times) * 0.2), 20)
    return times[skip:]

def check_ionic_steps_completed(folder):
    """
    Check if OUTCAR contains ionic step data (LOOP+ lines).
    Returns the number of ionic steps completed.
    """
    import subprocess
    outcar_path = f"{folder}/OUTCAR"
    cmd = f'grep -c "LOOP+" "{outcar_path}" 2>/dev/null || echo "0"'
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return 0

def parse_combo_from_folder(name):
    """Extract (NCORE, NPAR, KPAR) from a folder name if present."""
    import re
    m = re.search(r"NCORE=(\d+)_NPAR=(\d+)_KPAR=(\d+)", name)
    if m:
        return tuple(map(int, m.groups()))
    return (None, None, None)

def main():
    parser = argparse.ArgumentParser(
        description="Fast analyzer for VASP benchmark runtimes"
    )
    parser.add_argument("--base", default="benchmark_folders",
                        help="Directory containing benchmark subfolders")
    parser.add_argument("--top", type=float, default=0.1,
                        help="Fractional window around best to select top configs")
    args = parser.parse_args()

    # Automatically detect NSW from INCAR in the first folder
    first_folder = sorted(pathlib.Path(args.base).iterdir())[0]
    incar_path = first_folder / "INCAR"
    nsw = None
    try:
        with open(incar_path) as f:
            for line in f:
                if line.strip().startswith("NSW"):
                    # handle formats like 'NSW = 40'
                    parts = line.split("=")
                    if len(parts) >= 2:
                        nsw = int(parts[1].split()[0])
                        break
    except Exception:
        pass
    if nsw is None:
        sys.exit(f"Error: Could not detect NSW from {incar_path}")

    base = pathlib.Path(args.base)
    if not base.is_dir():
        sys.exit(f"Error: Base directory '{args.base}' not found")

    rows = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue

        # Check if ionic steps were actually executed
        num_steps = check_ionic_steps_completed(folder.as_posix())
        if num_steps == 0:
            print(f"Warning: no ionic steps completed in {folder.name}")
            continue

        times = extract_runtimes(folder.as_posix())
        if not times:
            print(f"Warning: no runtimes found in {folder.name}")
            continue

        avg_time = statistics.mean(times)
        ncore, npar, kpar = parse_combo_from_folder(folder.name)
        rows.append({
            "folder": folder.name,
            "NCORE": ncore,
            "NPAR": npar,
            "KPAR": kpar,
            "runtime_total_s": avg_time,
            "runtime_per_step_s": avg_time / nsw
        })

    if not rows:
        sys.exit("Error: no data collected (check folder names and job*.out files)")

    df = pd.DataFrame(rows).sort_values("runtime_per_step_s")
    df.to_csv("benchmark_results.csv", index=False)

    # Plot horizontal bar chart
    df_plot = df.sort_values("runtime_per_step_s")
    plt.figure(figsize=(8, max(4, 0.3 * len(df_plot))))
    plt.barh(df_plot["folder"], df_plot["runtime_per_step_s"])
    plt.gca().invert_yaxis()
    plt.xlabel(f"Seconds per step (NSW={nsw})")
    plt.title("VASP Benchmark Runtimes")
    # annotate bars
    for i, v in enumerate(df_plot["runtime_per_step_s"]):
        plt.text(v + 0.1, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    plt.savefig("benchmark_plot.png", dpi=300)

    # Summarize and save to file
    summary_lines = []
    summary_lines.append("===== Benchmark summary =====")
    summary_lines.append(df.to_string(
        index=False,
        formatters={
            "NCORE": "{:.0f}".format,
            "NPAR": "{:.0f}".format,
            "KPAR": "{:.0f}".format,
            "runtime_per_step_s": "{:.2f}".format
        }
    ))

    best = df.iloc[0]
    best_time = best["runtime_per_step_s"]
    window = best_time * (1 + args.top)
    top_df = df[df["runtime_per_step_s"] <= window]
    top_df.to_csv("top_configs.csv", index=False)

    summary_lines.append(f"")
    summary_lines.append(f"Best performance: {best_time:.2f} s/step in folder {best['folder']}")
    summary_lines.append(f"Configs within {args.top*100:.0f}% of best:")
    summary_lines.append(top_df[["folder", "runtime_per_step_s"]]
                         .to_string(index=False,
                                    formatters={"runtime_per_step_s": "{:.2f}".format})
    )
    outputs_msg = "Outputs written to: benchmark_results.csv, benchmark_plot.png, top_configs.csv, benchmark_summary.txt"
    summary_lines.append(outputs_msg)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open("benchmark_summary.txt", "w") as f:
        f.write(summary_text)

if __name__ == "__main__":
    main()
