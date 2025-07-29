import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy.interpolate import PchipInterpolator

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1
MIN_POINTS_FOR_EXTRAPOLATION = 3
MAX_POINTS_FOR_FIT = 5
POLY_ORDER = 3

# This function reads the force_stats_report.txt and extracts the values
def read_force_stats(file_path):
    """
    Reads the force stats report text file and extracts the CV value as well as cumulative analysis results.
    Returns a dictionary with:
      'CV': the CV value
      'Mean Force': the mean force value
      'Standard Deviation': the standard deviation value
      'MD steps': the number of MD steps
      'cumulative_data': a list of tuples (interval, cumulative mean, cumulative std)
    """
    cumulative_data = []
    cv_value = None
    mean_force = None
    std_dev = None
    md_steps = None
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'CV:' in line:
            cv_value = float(line.split('CV:')[1].strip())
        elif 'Mean Force:' in line:
            mean_force = -1 * float(line.split('Mean Force:')[1].strip())
        elif 'Standard Deviation:' in line:
            std_dev = float(line.split('Standard Deviation:')[1].strip())
        elif 'MD steps:' in line:
            md_steps = int(line.split('MD steps:')[1].strip())

    # Find the section containing "Cumulative Analysis Results:"
    start_index = None
    for i, line in enumerate(lines):
        if "Cumulative Analysis Results:" in line:
            start_index = i
            break

    if start_index is not None:
        # Process lines following the header
        for line in lines[start_index+1:]:
            parts = line.split()
            if not parts:
                continue
            try:
                interval = int(parts[0])
                cum_mean = float(parts[1])
                cum_std = float(parts[2])
                cumulative_data.append((interval, cum_mean, cum_std))
            except (ValueError, IndexError):
                continue

    return {
        "CV": cv_value,
        "Mean Force": mean_force,
        "Standard Deviation": std_dev,
        "MD steps": md_steps,
        "cumulative_data": cumulative_data
    }

def process_data():
    data = {
        'Constrained_Bond_Length (Å)': [],
        'Mean_Force (eV/Å)': [],
        'Standard_Deviation (eV/Å)': [],
        'MD_steps': [],
    }
    for folder in glob.glob("[0-9].[0-9][0-9]_*"):
        file_path = os.path.join(folder, 'force_stats_report.txt')
        if os.path.isfile(file_path):
            file_stats = read_force_stats(file_path)
            data['Constrained_Bond_Length (Å)'].append(file_stats.get("CV"))
            data['Mean_Force (eV/Å)'].append(file_stats.get("Mean Force"))
            data['Standard_Deviation (eV/Å)'].append(file_stats.get("Standard Deviation"))
            data['MD_steps'].append(file_stats.get("MD steps"))
    df = pd.DataFrame(data).sort_values(by=['Constrained_Bond_Length (Å)'])
    return df

# Updated calculate_pmf to identify three x-intercepts (IS, TS, FS) and extrapolate missing ones
# TS is defined as the intercept where the force changes from negative to positive.
# IS is the intercept to the left of TS (force changes from positive to negative) and FS is to the right.
# If either IS or FS is missing from raw data, a cubic polynomial extrapolation is performed using the nearest 7 points.
# Finally, the function inserts these intercepts into the sorted data and integrates between IS-TS and TS-FS to obtain
# the forward and backward barriers, respectively.

def calculate_pmf(x, y, std_dev):
    # Compute raw x-intercepts with sign information
    raw_intercepts = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            # Linear interpolation to find intercept
            x_cross = x[i] - (y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
            # Determine type: if sign goes from negative to positive, mark as TS candidate
            if y[i] < 0 and y[i + 1] > 0:
                raw_intercepts.append((x_cross, "TS"))
            else:
                raw_intercepts.append((x_cross, "other"))

    # Identify TS from raw intercepts (should always exist since data spans TS)
    ts_candidates = [xi for xi, typ in raw_intercepts if typ == "TS"]
    if ts_candidates:
        ts_val = ts_candidates[0]
    else:
        raise ValueError("TS intercept (negative-to-positive crossing) not found in raw data.")

    # Classify remaining intercepts as IS and FS based on their relation to TS
    is_val = None
    fs_val = None
    for xi, typ in raw_intercepts:
        if typ != "TS":
            if xi < ts_val:
                # For IS, choose the intercept closest to TS
                if is_val is None or (ts_val - xi) < (ts_val - is_val):
                    is_val = xi
            elif xi > ts_val:
                if fs_val is None or (xi - ts_val) < (fs_val - ts_val):
                    fs_val = xi

    # Extrapolate if IS or FS is missing
    x_sorted_raw = np.sort(x)
    y_sorted_raw = np.array(y)[np.argsort(x)]

    if is_val is None:
        # Extrapolate on the left side using a polynomial fit
        left_mask = x_sorted_raw < ts_val
        if np.sum(left_mask) >= MIN_POINTS_FOR_EXTRAPOLATION:
            x_left = x_sorted_raw[left_mask]
            y_left = y_sorted_raw[left_mask]
            n_points = min(MAX_POINTS_FOR_FIT, len(x_left))
            x_fit = x_left[:n_points]
            y_fit = y_left[:n_points]
            coeffs = np.polyfit(x_fit, y_fit, POLY_ORDER)
            roots = np.roots(coeffs)
            # Consider only roots less than ts_val and less than the minimum fitted x value
            real_roots = roots[np.isreal(roots)].real
            candidates = real_roots[(real_roots < ts_val) & (real_roots < x_fit[0])]
            if len(candidates) > 0:
                is_val = candidates[np.argmin(np.abs(ts_val - candidates))]

    if fs_val is None:
        # Extrapolate on the right side using a polynomial fit
        right_mask = x_sorted_raw > ts_val
        if np.sum(right_mask) >= MIN_POINTS_FOR_EXTRAPOLATION:
            x_right = x_sorted_raw[right_mask]
            y_right = y_sorted_raw[right_mask]
            n_points = min(MAX_POINTS_FOR_FIT, len(x_right))
            x_fit = x_right[-n_points:]
            y_fit = y_right[-n_points:]
            coeffs = np.polyfit(x_fit, y_fit, POLY_ORDER)
            roots = np.roots(coeffs)
            real_roots = roots[np.isreal(roots)].real
            # Consider only roots greater than ts_val and greater than the maximum fitted x value
            candidates = real_roots[(real_roots > ts_val) & (real_roots > x_fit[-1])]
            if len(candidates) > 0:
                fs_val = candidates[np.argmin(np.abs(candidates - ts_val))]

    if is_val is None:
        is_val = x_sorted_raw[0]
    if fs_val is None:
        fs_val = x_sorted_raw[-1]

    # Insert the intercepts into the data for integration
    # Only extend with points that are not already present in x
    new_points = []
    for pt in [is_val, ts_val, fs_val]:
        if not np.any(np.isclose(x, pt)):
            new_points.append(pt)
    x_extended = np.concatenate((x, np.array(new_points)))
    y_extended = np.concatenate((y, np.zeros(len(new_points))))
    sorted_indices = np.argsort(x_extended)
    x_sorted = x_extended[sorted_indices]
    y_sorted = y_extended[sorted_indices]

    # Create an extended std_dev array by interpolating at the intercept points
    # First, sort the original x and std_dev data
    x_sorted_raw = np.sort(x)
    std_sorted_raw = np.array(std_dev)[np.argsort(x)]
    
    # Interpolate std_dev at the intercepts and concatenate with the original std_dev
    std_extended = np.concatenate((std_dev, np.array([
        np.interp(is_val, x_sorted_raw, std_sorted_raw),
        np.interp(ts_val, x_sorted_raw, std_sorted_raw),
        np.interp(fs_val, x_sorted_raw, std_sorted_raw)
    ])))
    
    # Sort the extended std_dev array using the same sorted indices
    std_sorted = std_extended[sorted_indices]

    # Find nearest indices for IS, TS, FS
    def find_nearest_index(arr, value):
        return np.argmin(np.abs(arr - value))

    idx_is = find_nearest_index(x_sorted, is_val)
    idx_ts = find_nearest_index(x_sorted, ts_val)
    idx_fs = find_nearest_index(x_sorted, fs_val)

    # Compute barriers using PCHIP spline integration
    spline = PchipInterpolator(x_sorted, y_sorted)
    anti = spline.antiderivative()

    # Barrier and reaction energies via spline antiderivative
    forward_barrier = abs(-(anti(ts_val) - anti(is_val)))
    backward_barrier = abs(-(anti(fs_val) - anti(ts_val)))
    delta_G = -(anti(fs_val) - anti(is_val))

    # Error propagation via non-parametric bootstrap sampling
    N_boot = 5000
    rng = np.random.default_rng()
    areas_fwd = []
    areas_bwd = []
    areas_dG = []
    n_points = len(x_sorted)
    for _ in range(N_boot):
        # sample with replacement from the original data points
        idx = rng.integers(0, n_points, size=n_points)
        x_bs = x_sorted[idx]
        y_bs = y_sorted[idx]
        # combine duplicates by averaging to ensure strictly increasing x
        bs_df = pd.DataFrame({'x': x_bs, 'y': y_bs})
        bs_df = bs_df.groupby('x', sort=True, as_index=False)['y'].mean()
        x_bs_sorted = bs_df['x'].to_numpy()
        y_bs_sorted = bs_df['y'].to_numpy()
        # build spline and integrate using the original intercepts
        spline_bs = PchipInterpolator(x_bs_sorted, y_bs_sorted)
        anti_bs = spline_bs.antiderivative()
        areas_fwd.append(abs(anti_bs(ts_val) - anti_bs(is_val)))
        areas_bwd.append(abs(anti_bs(fs_val) - anti_bs(ts_val)))
        areas_dG.append(anti_bs(fs_val) - anti_bs(is_val))

    forward_std = np.std(areas_fwd, ddof=1)
    backward_std = np.std(areas_bwd, ddof=1)
    delta_G_std = np.std(areas_dG, ddof=1)

    return {
        "x_sorted": x_sorted,
        "y_sorted": y_sorted,
        "std_sorted": std_sorted,
        "IS": is_val,
        "TS": ts_val,
        "FS": fs_val,
        "Forward Barrier": forward_barrier,
        "Backward Barrier": backward_barrier,
        "Forward Barrier Std": forward_std,
        "Backward Barrier Std": backward_std,
        "Delta G": delta_G,
        "Delta G Std": delta_G_std
    }

def format_results(results):
    results_string = 'Activation Barriers (Area under the curve):\n'
    if "Forward Barrier" in results:
        results_string += f"Forward Barrier: {results['Forward Barrier']:.2f} ± {results['Forward Barrier Std']:.2f} eV\n"
    if "Backward Barrier" in results:
        results_string += f"Reverse Barrier: {results['Backward Barrier']:.2f} ± {results['Backward Barrier Std']:.2f}  eV\n"
    results_string += f"Free Energy of Reaction (ΔG): {results['Delta G']:.2f} ± {results['Delta G Std']:.2f} eV\n"

    results_string += "\nEquilibrium Bond Distances: \n"
    results_string += f"Initial State: {results['IS']:.2f} Å\n"
    results_string += f"Transition State: {results['TS']:.2f} Å\n"
    results_string += f"Final State: {results['FS']:.2f} Å\n"
    return results_string

def plot_pmf(results, x, y, std_dev, annotate=True, color_scheme="presentation", plot_spline=False):
    # Plot the PMF curve
    plt.figure(figsize=(10, 6))
    x_sorted = results["x_sorted"]
    y_sorted = results["y_sorted"]
    if plot_spline:
        # evaluate PCHIP spline on a fine grid
        spline = PchipInterpolator(x_sorted, y_sorted)
        xx = np.linspace(x_sorted[0], x_sorted[-1], 200)
        yy = spline(xx)
        plt.plot(xx, yy, color="black")
        mask_forward = (xx >= results["IS"]) & (xx <= results["TS"])
        mask_reverse = (xx >= results["TS"]) & (xx <= results["FS"])
        if color_scheme == "publication":
            plt.fill_between(xx, yy, where=mask_forward, facecolor='0.9', interpolate=True)
            plt.fill_between(xx, yy, where=mask_reverse, facecolor='0.9', interpolate=True)
        else:
            plt.fill_between(xx, yy, where=mask_forward, color="red", alpha=0.3, interpolate=True)
            plt.fill_between(xx, yy, where=mask_reverse, color="green", alpha=0.3, interpolate=True)
    else:
        plt.plot(x_sorted, y_sorted, color="black")  # PMF curve
        mask_forward = (x_sorted >= results["IS"]) & (x_sorted <= results["TS"])
        mask_reverse = (x_sorted >= results["TS"]) & (x_sorted <= results["FS"])
        if color_scheme == "publication":
            plt.fill_between(x_sorted, y_sorted, where=mask_forward, facecolor='0.9', interpolate=True)
            plt.fill_between(x_sorted, y_sorted, where=mask_reverse, facecolor='0.9', interpolate=True)
        else:
            plt.fill_between(x_sorted, y_sorted, where=mask_forward, color="red", alpha=0.3, interpolate=True)
            plt.fill_between(x_sorted, y_sorted, where=mask_reverse, color="green", alpha=0.3, interpolate=True)

    plt.errorbar(x, y, yerr=std_dev, fmt='o', color="black", ecolor='black', capsize=3.5)

    if annotate:
        # Plot markers for IS, TS, FS and annotate them with arrow and rectangular text box
        plt.scatter(results["IS"], 0, marker="o", s=50, color="red", edgecolors="black", zorder=5)
        plt.annotate("IS", xy=(results["IS"], 0), xytext=(25, 25), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="black"),
                     bbox=dict(boxstyle="square", fc="white", ec="black", lw=0.5),
                     fontsize=LABEL_FONTSIZE - 4, color="black")

        plt.scatter(results["TS"], 0, marker="o", s=50, color="red", edgecolors="black", zorder=5)
        plt.annotate("TS", xy=(results["TS"], 0), xytext=(-25, 25), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="black"),
                     bbox=dict(boxstyle="square", fc="white", ec="black", lw=0.5),
                     fontsize=LABEL_FONTSIZE - 4, color="black")

        plt.scatter(results["FS"], 0, marker="o", s=50, color="red", edgecolors="black", zorder=5)
        plt.annotate("FS", xy=(results["FS"], 0), xytext=(-25, -40), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="black"),
                     bbox=dict(boxstyle="square", fc="white", ec="black", lw=0.5),
                     fontsize=LABEL_FONTSIZE - 4, color="black")

    # Draw x-axis
    plt.axhline(0, color='black', linewidth=1)

    # Set axis labels and tick parameters
    plt.xlabel("Constrained Bond Length (Å)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Mean Force (eV/Å)", fontsize=LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)

    # Set full box frame
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)

    # Save the plot
    plt.savefig('pmf_curve.png', dpi=300, bbox_inches='tight')

def main():
    df = process_data()
    x = df['Constrained_Bond_Length (Å)'].to_numpy()
    y = df['Mean_Force (eV/Å)'].to_numpy()
    std_dev = df['Standard_Deviation (eV/Å)'].to_numpy()

    results = calculate_pmf(x, y, std_dev)
    results_string = format_results(results)

    # Print data in a table format and save it to a text file
    table_string = df.to_string(index=False)
    print(table_string + '\n')
    print(results_string)
    with open("pmf_analysis_results.txt", "w") as text_file:
        text_file.write(table_string + '\n\n')
        text_file.write(results_string)

    plot_pmf(results, x, y, std_dev, annotate=False, color_scheme="publication", plot_spline=True)

if __name__ == "__main__":
    main()
