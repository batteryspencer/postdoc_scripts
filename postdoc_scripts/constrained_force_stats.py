import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import savgol_filter, find_peaks

def get_file_line_count(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Modified function to include max_steps parameter
def read_simulation_data(folder_name, max_steps=None):
    lambda_values = []
    force_values_on_constrained_bond = []
    md_steps = 0
    
    with open(f'./{folder_name}/REPORT') as file:
        for line in file:
            if "b_m" in line:
                lambda_values.append(float(line.split()[1]))
            if "cc>" in line:
                try:
                    force_values_on_constrained_bond.append(float(line.split()[2]))
                except ValueError:
                    print('Error parsing collective variable value')
            if "MD step No." in line:
                md_steps += 1
                # Break the loop if the number of steps reaches max_steps
                if max_steps is not None and md_steps >= max_steps:
                    break
    
    return lambda_values, force_values_on_constrained_bond, md_steps

def autocorrelation(x):
    x = np.array(x)
    n = len(x)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode='full')[-n:]
    r /= np.arange(n, 0, -1)
    return r / r[0]

def integrated_autocorrelation_time(x):
    r = autocorrelation(x)
    # Sum over positive correlations (ignoring the zero lag)
    positive_r = r[1:][r[1:] > 0]
    if len(positive_r) == 0:
        return 1.0
    else:
        return 1 + 2 * np.sum(positive_r)

def calculate_statistics(lambda_values_per_cv):
    mean_force = np.mean(lambda_values_per_cv)
    std_uncorr = np.std(lambda_values_per_cv)
    tau = integrated_autocorrelation_time(lambda_values_per_cv)
    # Effective standard deviation accounts for the correlation time
    std_eff = std_uncorr * np.sqrt(tau / len(lambda_values_per_cv))
    return mean_force, std_eff

def cumulative_force_analysis(force_values):
    cumulative_means = []
    cumulative_stds = []
    cumulative_intervals = []

    total_number = len(force_values)
    if total_number >= 500:
        for i in range(500, total_number + 1, 500):
            cumulative_intervals.append(i)
            subset = force_values[:i]
            mean_val = np.mean(subset)
            std_uncorr = np.std(subset)
            tau = integrated_autocorrelation_time(subset)
            effective_std = std_uncorr * np.sqrt(tau / i)
            cumulative_means.append(mean_val)
            cumulative_stds.append(effective_std)

    return cumulative_intervals, cumulative_means, cumulative_stds

def find_histogram_peaks(data, bins=100):
    """Find significant peaks in histogram using smoothing and peak detection."""
    # Calculate histogram
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth the histogram using Savitzky-Golay filter
    window_length = 11  # Must be odd number
    hist_smooth = savgol_filter(hist, window_length, 3)  # polynomial order 3
    
    # Find peaks with minimum requirements
    peaks, _ = find_peaks(hist_smooth, 
                         height=np.max(hist_smooth) * 0.1,  # Minimum height threshold
                         distance=10,  # Minimum distance between peaks
                         prominence=np.max(hist_smooth) * 0.05)  # Minimum prominence
    
    # Create list of (force, frequency) tuples for significant peaks
    peak_list = [(bin_centers[peak], hist[peak]) for peak in peaks]
    peak_list.sort(key=lambda x: x[1], reverse=True)  # Sort by height
    
    return peak_list

def find_closest_frames(force_values, peak_forces, tolerance=0.1):
    """Find frame indices where force values are closest to peak forces."""
    peak_frames = {}
    for peak_force in peak_forces:
        # Find all forces within tolerance of the peak
        close_indices = [i for i, force in enumerate(force_values) 
                        if abs(force - peak_force) <= tolerance]
        peak_frames[peak_force] = close_indices[:5]  # Store up to 5 closest frames
    return peak_frames

def main():
    constraint_index = 0  # Specify the index of the constraint of interest
    num_constraints = get_file_line_count('ICONST')
    total_md_steps = 0  # Initialize total MD steps accumulator
    max_steps = None  # Specify the maximum number of steps to consider for analysis

    with open("INCAR", "r") as file:
        time_step = float(next((line.split('=')[1].strip() for line in file if "POTIM" in line), None))

    folders = sorted(glob('seg*'))
    if len(folders) == 0:  # Check if there are any folders to analyze
        print('No folders found for analysis')
        exit()
    
    all_lambda_values = []
    all_cv_values = []
    
    for folder in folders:
        if os.path.exists(f'./{folder}/REPORT'):
            lambda_values, cv_values, md_steps = read_simulation_data(folder, max_steps=max_steps)
            all_lambda_values.extend(lambda_values)
            all_cv_values.extend(cv_values)
            total_md_steps += md_steps  # Accumulate total MD steps here
    
    lambda_values_per_cv = all_lambda_values[constraint_index::num_constraints]
    # Generate a histogram of the lambda values
    plt.hist(lambda_values_per_cv, bins=100, edgecolor='black')
    plt.xlabel('Force (eV/Å)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Force Values')
    plt.savefig('force_histogram.png', dpi=300)

    mean_force, std_dev = calculate_statistics(lambda_values_per_cv)
    cumulative_intervals, cumulative_means, cumulative_stds = cumulative_force_analysis(lambda_values_per_cv)
    histogram_peaks = find_histogram_peaks(lambda_values_per_cv)

    # Find frames corresponding to peaks
    peak_forces = [peak[0] for peak in histogram_peaks]
    peak_frames = find_closest_frames(lambda_values_per_cv, peak_forces)

    with open('force_stats_report.txt', 'w') as output_file:
        output_file.write(f'Integrating over Reaction Coordinate Index: {constraint_index}, with a total of {num_constraints} constraints\n')
        output_file.write(f'CV: {all_cv_values[0]:.2f}\n')
        output_file.write(f'Mean Force: {mean_force:.2f}\n')
        output_file.write(f'Standard Deviation: {std_dev:.2f}\n')
        output_file.write(f'MD steps: {total_md_steps}\n')

        # Write cumulative analysis results to the file with aligned formatting
        output_file.write("Cumulative Analysis Results:\n")
        output_file.write(f"{'Interval':>10}{'Cumulative Mean':>20}{'Cumulative Std':>20}\n")
        for interval, mean, std in zip(cumulative_intervals, cumulative_means, cumulative_stds):
            output_file.write(f"{interval:>10}{mean:>20.2f}{std:>20.2f}\n")

        # Write histogram peak information with corresponding frames
        output_file.write("\nHistogram Peaks (Force, Frequency, Reference Frames):\n")
        for i, (force, frequency) in enumerate(histogram_peaks, 1):
            frames = peak_frames[force]
            frame_str = ", ".join(f"{frame}" for frame in frames)
            output_file.write(f"Peak {i}: Force = {force:.2f}, Frequency = {frequency:.0f}\n")
            output_file.write(f"   Reference Frames: {frame_str}\n")

    plt.figure()
    plt.errorbar(cumulative_intervals, cumulative_means, yerr=cumulative_stds, capsize=4, fmt='o-', label='Cumulative Mean Force')
    plt.xlabel('Simulation Interval (Number of Data Points)')
    plt.ylabel('Force (Arbitrary Units)')
    plt.title('Cumulative Analysis of Mean Force Over Simulation Intervals')
    plt.legend()
    plt.savefig('cumulative_force_stats.png', dpi=100)  # Updated file name for clarity

if __name__ == "__main__":
    main()
