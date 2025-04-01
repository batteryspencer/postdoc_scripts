import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy.integrate import simps

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1
MIN_POINTS_FOR_EXTRAPOLATION = 3
MAX_POINTS_FOR_FIT = 6
POLY_ORDER = 3

# This function reads the force_stats_report.txt and extracts the values
def read_force_stats(file_path, target_steps=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        stats = {}
        
        # Parse initial values
        if len(lines) > 2:
            stats['CV'] = float(lines[1].split(':')[1].strip())
            if target_steps is None:
                stats['Mean Force'] = -1 * np.around(float(lines[2].split(':')[1].strip()), 2)
                stats['Standard Deviation'] = float(lines[3].split(':')[1].strip())
                stats['MD steps'] = int(lines[4].split(':')[1].strip())

            # Parse cumulative analysis results
            for line in lines[6:]:  # Assuming the Cumulative Analysis starts at line 7
                parts = line.split()
                if len(parts) == 3 and int(parts[0]) == target_steps:
                    stats['Mean Force'] = -1 * float(parts[1])
                    stats['Standard Deviation'] = float(parts[2])
                    stats['MD steps'] = target_steps

        return stats

def process_data(target_steps=None):
    data = {'Constrained_Bond_Length (Å)': [], 'Mean_Force (eV/Å)': [], 'Standard_Deviation (eV/Å)': [], 'MD_Steps': []}
    for folder in glob.glob("[0-9].[0-9][0-9]_*"):
        file_path = os.path.join(folder, 'force_stats_report.txt')
        if os.path.isfile(file_path):
            stats = read_force_stats(file_path, target_steps=target_steps)
            data['Constrained_Bond_Length (Å)'].append(stats['CV'])
            data['Mean_Force (eV/Å)'].append(stats['Mean Force'])
            data['Standard_Deviation (eV/Å)'].append(stats['Standard Deviation'])
            data['MD_Steps'].append(stats['MD steps'])
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
        # Extrapolate on the left side using a polynomial fit with fallback
        left_mask = x_sorted_raw < ts_val
        if np.sum(left_mask) >= MIN_POINTS_FOR_EXTRAPOLATION:
            x_left = x_sorted_raw[left_mask]
            y_left = y_sorted_raw[left_mask]
            n_points_initial = min(MAX_POINTS_FOR_FIT, len(x_left))
            is_found = False
            # Loop from n_points_initial down to 2
            for n_points in range(n_points_initial, 1, -1):
                # Adjust polynomial order based on available points
                current_poly_order = POLY_ORDER if n_points > POLY_ORDER else n_points - 1
                if current_poly_order < 1:
                    continue
                x_fit = x_left[:n_points]
                y_fit = y_left[:n_points]
                coeffs = np.polyfit(x_fit, y_fit, current_poly_order)
                roots = np.roots(coeffs)
                real_roots = roots[np.isreal(roots)].real
                candidates = real_roots[real_roots < ts_val]
                if len(candidates) > 0:
                    is_val = candidates[np.argmin(np.abs(ts_val - candidates))]
                    is_found = True
                    break
            if not is_found:
                is_val = None

    if fs_val is None:
        # Extrapolate on the right side using a polynomial fit with fallback
        right_mask = x_sorted_raw > ts_val
        if np.sum(right_mask) >= MIN_POINTS_FOR_EXTRAPOLATION:
            x_right = x_sorted_raw[right_mask]
            y_right = y_sorted_raw[right_mask]
            n_points_initial = min(MAX_POINTS_FOR_FIT, len(x_right))
            fs_found = False
            # Loop from n_points_initial down to 2
            for n_points in range(n_points_initial, 1, -1):
                # Adjust polynomial order based on available points
                current_poly_order = POLY_ORDER if n_points > POLY_ORDER else n_points - 1
                if current_poly_order < 1:
                    continue
                x_fit = x_right[-n_points:]
                y_fit = y_right[-n_points:]
                coeffs = np.polyfit(x_fit, y_fit, current_poly_order)
                roots = np.roots(coeffs)
                real_roots = roots[np.isreal(roots)].real
                candidates = real_roots[real_roots > ts_val]
                if len(candidates) > 0:
                    fs_val = candidates[np.argmin(np.abs(candidates - ts_val))]
                    fs_found = True
                    break
            if not fs_found:
                fs_val = None

    # Insert the intercepts into the data for integration
    x_extended = np.concatenate((x, np.array([is_val, ts_val, fs_val])))
    y_extended = np.concatenate((y, np.array([0, 0, 0])))
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

    # Integrate force (using Simpson's rule) to estimate free energy barriers
    # Forward barrier: from IS to TS
    forward_barrier = abs(simps(y_sorted[idx_is:idx_ts + 1], x_sorted[idx_is:idx_ts + 1]))
    forward_variance = simps((std_sorted[idx_is:idx_ts + 1])**2, x_sorted[idx_is:idx_ts + 1])
    forward_std = np.sqrt(forward_variance)
    
    # Backward barrier: from TS to FS
    backward_barrier = abs(simps(y_sorted[idx_ts:idx_fs + 1], x_sorted[idx_ts:idx_fs + 1]))
    backward_variance = simps((std_sorted[idx_ts:idx_fs + 1])**2, x_sorted[idx_ts:idx_fs + 1])
    backward_std = np.sqrt(backward_variance)

    # Compute free energy of reaction (ΔG) and its uncertainty from IS to FS
    delta_G = abs(simps(y_sorted[idx_is:idx_fs + 1], x_sorted[idx_is:idx_fs + 1]))
    delta_G_var = simps((std_sorted[idx_is:idx_fs + 1])**2, x_sorted[idx_is:idx_fs + 1])
    delta_G_std = np.sqrt(delta_G_var)

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

def plot_pmf(results, x, y, std_dev, annotate=True, color_scheme="presentation"):
    # Plot the PMF curve
    plt.figure(figsize=(10, 6))
    x_sorted = results["x_sorted"]
    y_sorted = results["y_sorted"]
    plt.plot(x_sorted, y_sorted, color="black")  # PMF curve
    plt.errorbar(x, y, yerr=std_dev, fmt='o', color="black", ecolor='black', capsize=3.5)

    # Fill areas under the curve between IS-TS and TS-FS
    mask_forward = (x_sorted >= results["IS"]) & (x_sorted <= results["TS"])
    mask_reverse = (x_sorted >= results["TS"]) & (x_sorted <= results["FS"])
    if color_scheme == "publication":
        plt.fill_between(x_sorted, y_sorted, where=mask_forward, facecolor='0.9', interpolate=True)
        plt.fill_between(x_sorted, y_sorted, where=mask_reverse, facecolor='0.9', interpolate=True)
    else:  # presentation style
        plt.fill_between(x_sorted, y_sorted, where=mask_forward, color="red", alpha=0.3, interpolate=True)
        plt.fill_between(x_sorted, y_sorted, where=mask_reverse, color="green", alpha=0.3, interpolate=True)

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
    # target_steps=None will use the data until the last step of the simulation.
    # target_steps=np.arange(500, 10500, 500) will use the specified steps.
    df = process_data(target_steps=None)
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

    plot_pmf(results, x, y, std_dev, annotate=False, color_scheme="publication")

if __name__ == "__main__":
    main()
