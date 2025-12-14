from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
import json
from alive_progress import alive_bar

# Analysis mode configurations
# Each mode defines which analyses are enabled
MODE_CONFIG = {
    'heating': {
        'temperature_trend': True,
        'temperature_blocks': True,
        'energy_trend': False,
        'energy_stability': False,
        'energy_blocks': False,
        'temperature_autocorr': False,
        'energy_autocorr': False,
        'fourier_temperature': False,
        'fourier_vacf': False,
        'vacf': False,
    },
    'equilibration': {
        'temperature_trend': True,
        'temperature_blocks': True,
        'energy_trend': True,
        'energy_stability': True,
        'energy_blocks': True,
        'temperature_autocorr': False,
        'energy_autocorr': False,
        'fourier_temperature': False,
        'fourier_vacf': False,
        'vacf': False,
    },
    'production': {
        'temperature_trend': True,
        'temperature_blocks': True,
        'energy_trend': True,
        'energy_stability': True,
        'energy_blocks': True,
        'temperature_autocorr': True,
        'energy_autocorr': True,
        'fourier_temperature': True,
        'fourier_vacf': True,
        'vacf': True,
    },
}

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1
PS_TO_FS = 1000  # Conversion factor from picoseconds to femtoseconds

def detect_equilibration(values_series, stability_threshold, timestep_fs):
    """
    Determine whether the system equilibrated via block-mean overlap.
    Returns (eq_detected: bool, equil_time_ps: float or None).
    """
    default_block_size = len(values_series) // 10
    num_blocks = len(values_series) // default_block_size
    block_means = []
    for i in range(num_blocks):
        blk = values_series[i*default_block_size:(i+1)*default_block_size]
        block_means.append(blk.mean())
    eq_block = next(
        (i for i in range(1, len(block_means))
         if all(abs(block_means[j] - block_means[j-1]) <= stability_threshold
                for j in range(i, len(block_means)))),
        None
    )
    if eq_block is None:
        return False, None
    equil_time_ps = eq_block * default_block_size * timestep_fs / PS_TO_FS
    return True, equil_time_ps

def compute_stability_metrics(production_series, window_sizes, analysis_window_ps, stability_threshold, timestep_fs):
    """
    Compute moving-average fluctuations and drift slope for the production series.
    Returns a dict with keys:
      'sigma': {window_ps: (mean, fluctuation, is_stable)},
      'drift_slope': float,
      'drift_r2': float,
      'drift_stderr': float
    """
    metrics = {'sigma': {}}
    # convert to ps
    steps_per_ps = PS_TO_FS / timestep_fs
    # Determine dynamic window length
    production_len_ps = len(production_series) * (timestep_fs / PS_TO_FS)
    max_window_ps = min(analysis_window_ps, production_len_ps)
    dynamic_window_steps = int(max_window_ps * PS_TO_FS / timestep_fs)
    # Compute σ for each moving-average window
    for w in window_sizes:
        window_ps = w / steps_per_ps
        rolling_mean = production_series.rolling(window=w).mean().dropna()
        analysis_data = rolling_mean.iloc[-dynamic_window_steps:]
        fluctuation = analysis_data.std()
        mean_energy = analysis_data.mean()
        is_stable = fluctuation <= stability_threshold
        metrics['sigma'][window_ps] = (mean_energy, fluctuation, is_stable)
    # Compute drift slope
    times_prod = np.arange(len(production_series)) * (timestep_fs / PS_TO_FS)
    drift_result = linregress(times_prod, production_series.values)
    metrics['drift_slope'] = drift_result.slope
    metrics['drift_r2'] = drift_result.rvalue**2
    metrics['drift_stderr'] = drift_result.stderr
    return metrics

def read_energies_from_oszicar(file_path):
    energies = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'E0=' in line:
                    parts = line.split()
                    e0_index = next((i for i, part in enumerate(parts) if 'E0=' in part), None)
                    if e0_index is not None:
                        energy_str = parts[e0_index+1]
                        try:
                            energy = float(energy_str)
                            energies.append(energy)
                        except ValueError:
                            print(f"Warning: Non-numeric energy value encountered in {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return energies

def extract_total_energies(file_path):
    total_energies = []
    e0_pending = None

    with open(file_path, 'r') as file:
        for line in file:
            # Capture the "energy without entropy" (E₀)
            if 'energy without entropy' in line:
                # Split on '=' and take the first token after it, handle unicode dash
                raw = line.split('=', 1)[1].strip().split()[0]
                raw = raw.replace('–', '-')
                e0_pending = float(raw)
            # When EKIN appears and we have an E₀ pending, pair and reset
            elif 'kinetic energy EKIN' in line and e0_pending is not None:
                raw = line.split('=', 1)[1].strip().split()[0]
                ekin = float(raw)
                total_energies.append(e0_pending + ekin)
                e0_pending = None

    return total_energies

def read_temperatures_from_outcar(file_path):
    temperatures = []
    temp_pattern = re.compile(r"temperature\s+(\d+\.\d+)\s+K")
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'kin. lattice' in line:
                    match = temp_pattern.search(line)
                    if match:
                        temperature = float(match.group(1))
                        temperatures.append(temperature)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return temperatures

def get_num_atoms_from_outcar(outcar_path):
    num_atoms = 0
    with open(outcar_path, 'r') as file:
        for line in file:
            if "NIONS" in line:
                num_atoms = int(line.split('=')[-1].strip())
                break
    if num_atoms == 0:
        raise ValueError("Unable to find NIONS in the OUTCAR file.")
    return num_atoms

def read_velocities_from_vdatcar(file_path, num_atoms):
    velocities = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    reading_velocities = False
    for line in lines:
        if 'Direct configuration=' in line:  # Start of a new set of velocities
            reading_velocities = True
            continue
        if reading_velocities:
            if line.strip() and 'KINETIC ENERGY' not in line and 'TEMP EFF' not in line:
                # Read velocities only if the line is not empty and does not contain these keywords
                try:
                    velocity = [float(val) for val in line.split()]
                    if len(velocity) == 3:  # Ensure there are exactly three components
                        velocities.append(velocity)
                except ValueError:
                    # Handle the case where conversion to float fails
                    continue

    velocities = np.array(velocities)
    num_timesteps = len(velocities) // num_atoms

    if len(velocities) != num_atoms * num_timesteps:
        raise ValueError("Mismatch in the expected number of velocity entries and the actual count.")

    return velocities.reshape((num_timesteps, num_atoms, 3))

def autocorrelation(x):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')[-n:]
    result /= np.max(result)
    return result

def compute_vacf(velocities):
    num_steps, num_atoms, _ = velocities.shape
    vacf = np.zeros(num_steps)
    with alive_bar(num_atoms, title='Computing VACF') as bar:
        for i in range(num_atoms):
            for j in range(3):  # x, y, z components
                vacf += autocorrelation(velocities[:, i, j])
            bar()
    vacf /= (num_atoms * 3)
    return vacf

def plot_values(values, target_value, window_size, ylabel, title, file_name):
    plt.figure(figsize=(10, 6))
    steps = range(len(values))
    
    # Convert values to a Pandas Series to use rolling function
    values_series = pd.Series(values)
    rolling_mean = values_series.rolling(window=window_size).mean()  # Adjust the window size as needed

    # Plot the raw data in gray
    plt.plot(steps, values, label=ylabel, color='gray', alpha=0.5)

    # Plot the rolling mean in black
    plt.plot(steps, rolling_mean, label='Rolling Mean', color='black', linewidth=2)

    mean_value = np.mean(values)
    std_dev = np.std(values)

    # Compute the rolling mean of the final 2500 steps
    if len(values) >= 2500:
        final_2500_mean = np.mean(values[-2500:])
    else:
        final_2500_mean = np.mean(values)  # If less than 2500 steps, take the mean of the entire array

    # Print the mean, standard deviation, and final 2500 steps rolling mean to the report file
    with open(f'equilibrium_analysis_report.txt', 'a') as file:
        file.write(f"Mean {ylabel}: {mean_value:.2f}\n")
        file.write(f"Standard Deviation: {std_dev:.2f}\n")
        file.write(f"Based on Final 2500 Steps, Equilibrated {ylabel}: {final_2500_mean:.2f}\n\n")

    # Highlight the overall mean with a red dashed line
    plt.axhline(mean_value, color='r', linestyle='dashed', linewidth=1, label=f"Mean {ylabel}: {mean_value:.2f}")
    
    if target_value is not None:
        plt.axhline(target_value, color='g', linestyle='dashed', linewidth=1, label=f'Target {ylabel}: {target_value:.2f}')

    plt.xlabel('Ionic Step', fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.savefig(file_name)
    plt.close()

def test_energy_stability(values, window_size, analysis_window_ps=5, stability_threshold=0.1, timestep_fs=1, ylabel='Average Energy (eV)', file_name='stability_plot.png'):
    # Detect equilibration
    values_series = pd.Series(values)
    eq_detected, equil_time_ps = detect_equilibration(values_series, stability_threshold, timestep_fs)

    # Estimate energy autocorrelation time in steps
    acf = autocorrelation(values_series.values)
    tau_steps = estimate_autocorrelation_time(acf, timestep=1)  # timestep in fs
    # Write tau to report
    with open('equilibrium_analysis_report.txt', 'a') as file:
        file.write(f"Energy autocorrelation time: {tau_steps * timestep_fs / PS_TO_FS:.2f} ps ({tau_steps} steps)\n")

    # Handle missing equilibration block: log, but continue to plotting with limited analysis
    if not eq_detected:
        print("No equilibration block found: system not equilibrated per block-overlap test.")
        with open('equilibrium_analysis_report.txt', 'a') as file:
            file.write("\n=== Equilibration Status ===\n")
            file.write("Equilibration NOT detected via block-overlap test.\n")
            file.write("Recommendation: extend simulation or review equilibration criteria.\n")
            file.write("============================\n\n")

    # Guard against empty energy data
    if values_series.empty:
        raise ValueError("No energy data available for stability analysis.")

    # Define production series: limit to last 10 ps after equilibration or full post length
    if eq_detected:
        # Full post-equilibration series
        start_step = int(equil_time_ps * PS_TO_FS / timestep_fs)
        full_post = values_series.iloc[start_step:]
        # Limit production window to last 10 ps or full post length, whichever is smaller
        max_prod_ps = min(len(full_post) * (timestep_fs / PS_TO_FS), 10.0)
        prod_steps = int(max_prod_ps * PS_TO_FS / timestep_fs)
        production_series = full_post.iloc[-prod_steps:]
        # Determine actual production start time in ps
        prod_start_step = start_step + (len(full_post) - prod_steps)
        prod_start_ps = prod_start_step * (timestep_fs / PS_TO_FS)
    else:
        production_series = values_series
        max_prod_ps = len(production_series) * (timestep_fs / PS_TO_FS)

    # Use the single window size provided, auto-adjust if too large
    if window_size > len(production_series):
        adjusted_window = len(production_series) // 2  # Use half the available data
        print(f"Warning: window_size ({window_size}) exceeds production data length ({len(production_series)}). "
              f"Automatically reducing to {adjusted_window} steps.")
        with open('equilibrium_analysis_report.txt', 'a') as file:
            file.write(f"Warning: window_size adjusted from {window_size} to {adjusted_window} steps "
                       f"(production data has only {len(production_series)} steps).\n\n")
        window_size = adjusted_window
    window_sizes = [window_size]

    # Compute stability metrics on production_series
    metrics = compute_stability_metrics(production_series, window_sizes, analysis_window_ps, stability_threshold, timestep_fs)

    # Create a figure with three subplots in a 3x1 grid: raw, moving average, cumulative mean
    fig, (ax_raw, ax_moving, ax_cum) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # === 1: Raw energy panel ===
    ax_raw.plot(np.arange(len(values)) * timestep_fs / PS_TO_FS, values, color='gray', alpha=0.7)
    ax_raw.set_ylabel('Internal Energy (eV)', fontsize=LABEL_FONTSIZE)
    ax_raw.set_title('Raw Energy Data', fontsize=TITLE_FONTSIZE)
    ax_raw.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)

    # === 2: Moving-average panel ===
    colors = plt.cm.Dark2(np.linspace(0, 1, len(window_sizes)))
    steps_per_ps = 1000 / timestep_fs
    for idx, window_size in enumerate(window_sizes):
        rolling_mean = values_series.rolling(window=window_size).mean()
        filtered_rolling_mean = rolling_mean.dropna()
        # Use the Series index to preserve absolute timestamps
        times = filtered_rolling_mean.index * timestep_fs / PS_TO_FS
        window_ps = window_size / steps_per_ps
        ax_moving.plot(times, filtered_rolling_mean,
                       label=f'{window_ps:.1f} ps', linewidth=2, color=colors[idx])
        # Fill ±σ band and annotate σ only if eq_detected
        if eq_detected:
            mean_energy, fluctuation, stability_status = metrics['sigma'][window_ps]
            # Shaded uncertainty band ± standard deviation
            ax_moving.fill_between(
                times,
                filtered_rolling_mean - fluctuation,
                filtered_rolling_mean + fluctuation,
                color=colors[idx],
                alpha=0.2
            )
    # Mark equilibration cutoff and annotate equil time if detected
    if eq_detected:
        ax_moving.axvline(equil_time_ps, color='black', linestyle='--', linewidth=1)
        ax_moving.text(
            equil_time_ps + 0.5,
            ax_moving.get_ylim()[1],
            f'Equil = {equil_time_ps:.1f} ps',
            color='black',
            fontsize=LEGEND_FONTSIZE,
            rotation=90,
            va='top'
        )
        # Compute window size in ps for shading offset
        steps_per_ps = PS_TO_FS / timestep_fs
        window_ps = window_size / steps_per_ps
        # Use the MA curve to determine actual end of production window
        # Use the same times variable as above (from the last window, which is the only one anyway)
        # MA curve ends at the last available time stamp
        ma_end_ps = times[-1]
        # Shade the production window using the actual MA range
        ax_moving.axvspan(
            prod_start_ps + window_ps,
            ma_end_ps,
            color='gray', alpha=0.2,
            label='Production Window'
        )

        # Annotate σ inside the production window (use the first window size for annotation)
        window_ps = window_sizes[0] / steps_per_ps
        mean_energy, fluctuation, stability_status = metrics['sigma'][window_ps]
        # Center the σ annotation within the shaded region
        x_sig = (prod_start_ps + window_ps + ma_end_ps) / 2
        y_min, y_max = ax_moving.get_ylim()
        y_sig = y_min + 0.6 * (y_max - y_min)
        ax_moving.text(
            x_sig,
            y_sig,
            f"σ = {fluctuation:.2f} eV",
            fontsize=LEGEND_FONTSIZE,
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
        # Annotate absolute drift slope inside the production window
        # Recompute y_min, y_max in case they have changed
        y_min, y_max = ax_moving.get_ylim()
        slope_abs = abs(metrics['drift_slope'])
        # Place slope text just below the σ annotation
        y_slope = y_sig - 0.05 * (y_max - y_min)
        ax_moving.text(
            x_sig,
            y_slope,
            f"|slope| = {slope_abs:.4f} eV/ps",
            fontsize=LEGEND_FONTSIZE,
            ha='center',
            va='top',
            color='#222222',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

    ax_moving.set_xlabel('Time (ps)', fontsize=LABEL_FONTSIZE)
    ax_moving.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax_moving.set_title('Moving Average Data', fontsize=TITLE_FONTSIZE)
    ax_moving.legend(fontsize=LEGEND_FONTSIZE, loc='lower left')
    ax_moving.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)

    # === 3: Cumulative-mean panel ===
    cumu_mean = values_series.expanding().mean()
    times_full = np.arange(len(values_series)) * timestep_fs / PS_TO_FS
    ax_cum.plot(times_full, cumu_mean, color='blue', linewidth=2, label='Cumulative Mean')
    ax_cum.set_ylabel('Cumulative Mean (eV)', fontsize=LABEL_FONTSIZE)
    ax_cum.set_xlabel('Time (ps)', fontsize=LABEL_FONTSIZE)
    ax_cum.set_title('Cumulative Mean Energy', fontsize=TITLE_FONTSIZE)
    ax_cum.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    ax_cum.legend(fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    # --------- Report writing after plotting ---------
    if eq_detected:
        # Calculate the actual window size in ps for reporting
        steps_per_ps_report = PS_TO_FS / timestep_fs
        window_ps_report = window_sizes[0] / steps_per_ps_report
        with open('equilibrium_analysis_report.txt', 'a') as file:
            file.write("\n=== Equilibration Status ===\n")
            file.write(f"Equilibration detected at {equil_time_ps:.2f} ps.\n")
            file.write(f"Production window: {max_prod_ps:.2f} ps\n")
            file.write(f"  • Drift slope: {metrics['drift_slope']:.4g} eV/ps\n")
            file.write(f"  • σ ({window_ps_report:.1f} ps MA window): {metrics['sigma'][window_ps_report][1]:.2f} eV\n")
            file.write("============================\n\n")

    # Write machine-readable JSON report with rounded values
    json_metrics = {
        "energy_autocorrelation_time": {
            "value": round(tau_steps * timestep_fs / PS_TO_FS, 3),
            "unit": "ps"
        },
    }
    if eq_detected:
        json_metrics["equilibration_time"] = {
            "value": round(equil_time_ps, 3),
            "unit": "ps"
        }
        json_metrics["drift_slope"] = {
            "value": round(metrics["drift_slope"], 6),
            "unit": "eV/ps"
        }
        json_metrics["sigma"] = {}
        for w, (_, fluctuation, _) in metrics["sigma"].items():
            key = f"{w:.1f}_ps"
            json_metrics["sigma"][key] = {
                "value": round(fluctuation, 6),
                "unit": "eV"
            }
    else:
        json_metrics["equilibration_time"] = None
        json_metrics["drift_slope"] = None
        json_metrics["sigma"] = None

    with open('equilibrium_analysis_report.json', 'w') as jf:
        json.dump(json_metrics, jf, indent=2)
    return None

def print_top_frequencies(frequencies, amplitudes, data_type, top_n):
    with open(f'equilibrium_analysis_report.txt', 'a') as file:
        file.write(f"\n{data_type} - Top {top_n} Frequencies:\n")
        file.write("-------------------------------------------\n")
        file.write("Rank | Frequency (THz) | Amplitude\n")
        file.write("-----|-----------------|-----------\n")
        for i in range(top_n):
            file.write(f"{i+1:<4} | {frequencies[i]:<15.2f} | {amplitudes[i]:<10.2f}\n")

def plot_fourier_transform(values, timestep_fs, ylabel, title, file_name, data_type="Data"):
    plt.figure(figsize=(10, 6)) 
    
    # Convert time step to seconds
    timestep_s = timestep_fs * 1e-15  # Convert fs to s

    # Compute the FFT and corresponding frequencies in Hz
    fft_values = np.fft.fft(values)
    fft_freq_hz = np.fft.fftfreq(len(values), d=timestep_s)

    # Convert frequencies to THz from Hz
    fft_freq_thz = fft_freq_hz * 1e-12  # Convert Hz to THz
    
    # Only take the positive half of the spectrum and corresponding frequencies
    half_n = len(fft_values) // 2
    fft_values = np.abs(fft_values[:half_n])
    fft_freq_thz = fft_freq_thz[:half_n]

    # Plot the FFT spectrum
    plt.plot(fft_freq_thz, fft_values, label=ylabel)

    # Exclude zero frequency
    nonzero_indices = np.where(fft_freq_thz > 0)[0]
    fft_values_nonzero = fft_values[nonzero_indices]
    fft_freq_thz_nonzero = fft_freq_thz[nonzero_indices]

    # Define the number of top frequencies you are interested in
    top_n = 20

    # Sort amplitudes and frequencies in descending order based on amplitudes
    sorted_indices = np.argsort(fft_values_nonzero)[::-1]  # Get indices for sorted amplitudes in descending order

    # Select the top N frequencies and their amplitudes
    top_indices = sorted_indices[:top_n]
    top_frequencies = fft_freq_thz_nonzero[top_indices]
    top_amplitudes = fft_values_nonzero[top_indices]

    # Print the top frequencies for temperature fluctuations
    print_top_frequencies(top_frequencies, top_amplitudes, data_type, top_n)

    plt.xlabel('Frequency (THz)', fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.savefig(file_name)
    plt.close()

def compute_and_plot_block_averages(data_series, num_blocks=10, target_value=None, x_label='Block Number', y_label='Value', title='Block Averages and Std Dev', filename='block_averages_std_dev.png'):
    # Ensure there's enough data to form the requested number of blocks
    if len(data_series) < num_blocks:
        print(f"Warning: Not enough data points ({len(data_series)}) for the requested number of blocks ({num_blocks}).")
        return

    # Divide the data into blocks
    block_size = len(data_series) // num_blocks
    block_means = []
    block_stds = []

    for i in range(num_blocks):
        block = data_series[i * block_size:(i + 1) * block_size]
        block_means.append(np.mean(block))
        block_stds.append(np.std(block))

    # Calculate the overall mean and standard deviation
    overall_mean = np.mean(data_series)
    overall_std = np.std(data_series)

    # Define the y-axis range dynamically based on the overall standard deviation
    y_min = overall_mean - 3 * overall_std
    y_max = overall_mean + 3 * overall_std

    # Plotting
    plt.figure(figsize=(10, 6))
    blocks = range(1, num_blocks + 1)
    plt.errorbar(blocks, block_means, yerr=block_stds, fmt='-o', capsize=5, label='Block Mean ± Std Dev')

    if target_value is not None:
        plt.axhline(target_value, color='g', linestyle='dashed', linewidth=1, label=f'Target {y_label}: {target_value:.2f}')

    plt.xlabel(x_label, fontsize=LABEL_FONTSIZE)
    plt.ylabel(y_label, fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    plt.ylim(y_min, y_max)  # Set the y-axis limits
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.savefig(filename)
    plt.close()

def estimate_autocorrelation_time(acf, timestep=1):
    """
    Estimate the intrinsic fluctuation timescale as the lag (in time units) 
    at which the autocorrelation decays to 1/e.
    """
    threshold = 1 / np.e
    indices_below = np.where(acf < threshold)[0]
    if len(indices_below) == 0:
        return len(acf) * timestep
    return indices_below[0] * timestep

def plot_autocorrelation(values, ylabel):
    plt.figure(figsize=(10, 6))
    acf = autocorrelation(values)
    steps = range(len(acf))
    plt.plot(steps, acf, label='Autocorrelation')
    plt.axhline(0.0, color='r', linestyle='dashed', linewidth=1)

    plt.xlabel('Lag', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Autocorrelation', fontsize=LABEL_FONTSIZE)
    plt.title(f'{ylabel} Autocorrelation Function', fontsize=TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.savefig(f'{ylabel.lower().replace(" ", "_")}_autocorrelation.png')
    plt.close()

def plot_vacf(vacf, timestep_fs, ylabel='Velocity Autocorrelation', filename='vacf.png'):
    plt.figure(figsize=(10, 6))
    time = np.arange(len(vacf)) * timestep_fs  # Convert step number to actual time
    plt.plot(time, vacf, label=ylabel)
    plt.axhline(0.0, color='r', linestyle='dashed', linewidth=1)

    plt.xlabel('Time (fs)', fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.title(f'{ylabel} Function', fontsize=TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.savefig(filename)
    plt.close()

def run_analysis(mode, config, num_steps, window_size, energy_window_size):
    """Run all enabled analyses based on the provided configuration."""
    # Find all directories that start with 'seg' and are present in the current directory
    seg_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('seg')])
    current_directory = os.getcwd()
    seg_dirs.insert(0, current_directory)

    total_temperatures = []
    total_energies = []
    total_velocities = None

    # Read timestep from INCAR
    timestep_fs = float([line.split('=')[-1].strip() for line in open(f'{seg_dirs[0]}/INCAR') if 'POTIM' in line][0])

    # Load VDATCAR data if VACF analysis is enabled
    if config['vacf']:
        print("Loading VDATCAR data...")
        vdatcar_path = 'VDATCAR'
        num_atoms = get_num_atoms_from_outcar(f'{seg_dirs[0]}/OUTCAR')
        total_velocities = read_velocities_from_vdatcar(vdatcar_path, num_atoms)

    # Read temperature and energy data from all segment directories
    # Calculate target time for progress display
    target_time_ps = num_steps * timestep_fs / PS_TO_FS

    with alive_bar(manual=True, title='Reading OUTCAR files') as bar:
        for seg_dir in seg_dirs:
            outcar_path = os.path.join(seg_dir, 'OUTCAR')
            if os.path.exists(outcar_path):
                total_temperatures.extend(read_temperatures_from_outcar(outcar_path))
                total_energies.extend(extract_total_energies(outcar_path))
            # Calculate current time based on steps read so far
            current_steps = min(len(total_temperatures), num_steps)
            current_time_ps = current_steps * timestep_fs / PS_TO_FS
            progress_fraction = min(current_time_ps / target_time_ps, 1.0)
            bar(progress_fraction)
            bar.text(f'{current_time_ps:.1f} / {target_time_ps:.1f} ps')

    # Truncate data
    total_temperatures = total_temperatures[:num_steps]
    total_energies = total_energies[:num_steps]
    target_temperature = total_temperatures[0]
    target_energy = None

    # Initialize report file
    with open('equilibrium_analysis_report.txt', 'w') as file:
        file.write(f"=== AIMD Analysis Report (Mode: {mode}) ===\n\n")

    # Count enabled analyses for progress bar
    analysis_steps = []
    if config['temperature_trend']:
        analysis_steps.append(('Temperature trend', lambda: plot_values(total_temperatures, target_temperature, window_size, 'Temperature (K)', 'Temperature per Ionic Step Across Simulation', 'temperature_trend.png')))
    if config['energy_trend']:
        analysis_steps.append(('Energy trend', lambda: plot_values(total_energies, target_energy, window_size, 'Total Energy (eV)', 'Total Energy per Ionic Step Across Simulation', 'total_energy_trend.png')))
    if config['energy_stability']:
        analysis_steps.append(('Energy stability', lambda: test_energy_stability(total_energies, energy_window_size, analysis_window_ps=5, stability_threshold=0.1, timestep_fs=timestep_fs, ylabel='Average Energy (eV)', file_name='stability_plot.png')))
    if config['fourier_temperature']:
        analysis_steps.append(('Fourier (temperature)', lambda: plot_fourier_transform(total_temperatures, timestep_fs, 'Amplitude', 'Fourier Transform of Temperature Fluctuations', 'temperature_fourier_transform.png', 'Temperature Fluctuations')))
    if config['temperature_blocks']:
        analysis_steps.append(('Temperature blocks', lambda: compute_and_plot_block_averages(total_temperatures, num_blocks=10, target_value=target_temperature, x_label='Block Number', y_label='Temperature (K)', title='Block Averages and Std Dev of Temperature', filename='temperature_block_averages.png')))
    if config['energy_blocks']:
        analysis_steps.append(('Energy blocks', lambda: compute_and_plot_block_averages(total_energies, num_blocks=10, target_value=target_energy, x_label='Block Number', y_label='Energy (eV)', title='Block Averages and Std Dev of Total Energy', filename='total_energy_block_averages.png')))
    if config['temperature_autocorr']:
        analysis_steps.append(('Temperature autocorr', lambda: plot_autocorrelation(total_temperatures, 'Temperature')))
    if config['energy_autocorr']:
        analysis_steps.append(('Energy autocorr', lambda: plot_autocorrelation(total_energies, 'Total Energy')))
    if config['vacf']:
        def run_vacf():
            vacf = compute_vacf(total_velocities)
            plot_vacf(vacf, timestep_fs)
            if config['fourier_vacf']:
                plot_fourier_transform(vacf, timestep_fs, 'Amplitude', 'Fourier Transform of Velocity Fluctuations', 'velocity_fourier_transform.png', 'VACF')
        analysis_steps.append(('VACF analysis', run_vacf))

    # Run enabled analyses with progress bar
    with alive_bar(len(analysis_steps), title='Running analyses', enrich_print=True) as bar:
        for step_name, step_func in analysis_steps:
            bar.text(f'-> {step_name}')
            step_func()
            bar()


def main():
    # ================== USER INPUTS ==================
    mode = 'equilibration'  # Options: 'heating', 'equilibration', 'production'
    num_steps = 30000
    window_size = 100           # Rolling window for temperature/energy plots
    energy_window_size = 3000   # Window size for energy stability analysis
    # =================================================

    print(f"Running in '{mode}' mode")
    run_analysis(mode, MODE_CONFIG[mode], num_steps, window_size, energy_window_size)
    print("Analysis complete. Results saved to equilibrium_analysis_report.txt")

if __name__ == "__main__":
    main()
