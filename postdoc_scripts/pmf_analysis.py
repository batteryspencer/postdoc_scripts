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
MAX_POINTS_FOR_FIT = 5
POLY_ORDER = 3

# This function reads the force_stats_report.txt and extracts the values
def read_force_stats(file_path):
    """
    Reads the force stats report text file and extracts the CV value as well as cumulative analysis results.
    Returns a dictionary with:
      'CV': the CV value
      'cumulative_data': a list of tuples (interval, cumulative mean, cumulative std)
    """
    cumulative_data = []
    cv_value = None
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read the CV value from the file (search for the line containing 'CV:')
    for line in lines:
        if 'CV:' in line:
            cv_value = float(line.split('CV:')[1].strip())
            break
    if cv_value is None:
        raise ValueError("CV value not found in file, but it is expected to always be available.")

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

    return {"CV": cv_value, "cumulative_data": cumulative_data}

def process_data():
    data = {
        'Constrained_Bond_Length (Å)': [],
        'Mean_Force (eV/Å)': [],
        'Standard_Deviation (eV/Å)': []
    }
    for folder in glob.glob("[0-9].[0-9][0-9]_*"):
        file_path = os.path.join(folder, 'force_stats_report.txt')
        if os.path.isfile(file_path):
            file_stats = read_force_stats(file_path)
            cv_val = file_stats.get("CV")
            cumulative_data = file_stats.get("cumulative_data", [])
            # For each cumulative analysis row, use the CV value for the bond length and invert the sign of the mean force
            for _, cum_mean, cum_std in cumulative_data:
                data['Constrained_Bond_Length (Å)'].append(cv_val)
                data['Mean_Force (eV/Å)'].append(-1 * np.around(cum_mean, 2))
                data['Standard_Deviation (eV/Å)'].append(cum_std)
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

    # Integrate force (using Simpson's rule) to estimate free energy barriers
    # Forward barrier: from IS to TS
    forward_barrier = abs(-simps(y_sorted[idx_is:idx_ts + 1], x_sorted[idx_is:idx_ts + 1]))
    forward_variance = simps((std_sorted[idx_is:idx_ts + 1])**2, x_sorted[idx_is:idx_ts + 1])
    forward_std = np.sqrt(forward_variance)
    
    # Backward barrier: from TS to FS
    backward_barrier = abs(-simps(y_sorted[idx_ts:idx_fs + 1], x_sorted[idx_ts:idx_fs + 1]))
    backward_variance = simps((std_sorted[idx_ts:idx_fs + 1])**2, x_sorted[idx_ts:idx_fs + 1])
    backward_std = np.sqrt(backward_variance)

    # Compute free energy of reaction (ΔG) and its uncertainty from IS to FS
    # ΔG = -∫F(x)dx from IS to FS
    # Note: The negative sign is used to convert from force to free energy
    # The integral of the force gives the change in free energy
    # between the initial and final states.
    delta_G = -simps(y_sorted[idx_is:idx_fs + 1], x_sorted[idx_is:idx_fs + 1])
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

    plot_pmf(results, x, y, std_dev, annotate=False, color_scheme="publication")

if __name__ == "__main__":
    main()
