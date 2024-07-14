import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd

# Define the number of steps to analyze
NUM_STEPS_TO_ANALYZE = 10000

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

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

    with open(file_path, 'r') as file:
        for line in file:
            if 'free  energy' in line:
                dft_energy = float(re.search(r'[-+]?\d*\.\d+|\d+', line).group())
            elif 'kinetic energy EKIN' in line:
                kinetic_energy = float(re.search(r'[-+]?\d*\.\d+|\d+', line).group())

                # Check if both energies are extracted
                if dft_energy is not None and kinetic_energy is not None:
                    total_energy = dft_energy + kinetic_energy
                    total_energies.append(total_energy)
                    dft_energy = None  # Reset for the next iteration
                    kinetic_energy = None  # Reset for the next iteration

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
    for i in range(num_atoms):
        for j in range(3):  # x, y, z components
            vacf += autocorrelation(velocities[:, i, j])
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

    # Print the mean and standard deviation
    with open(f'equilibrium_analysis_report.txt', 'a') as file:
        file.write(f"Mean {ylabel}: {mean_value:.2f}\n")
        file.write(f"Standard Deviation: {std_dev:.2f}\n")

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

def main():
    # Find all directories that start with 'seg' and are present in the current directory
    seg_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('seg')])

    # Prepend the current directory to the sorted list
    current_directory = os.getcwd()
    seg_dirs.insert(0, current_directory)

    total_temperatures = []
    total_energies = []

    # Assume VDATCAR is in the current directory or a specified path
    vdatcar_path = 'VDATCAR'  # Update this path if VDATCAR is in a different location
    num_atoms = get_num_atoms_from_outcar(f'{seg_dirs[0]}/OUTCAR')
    timestep_fs = float([line.split('=')[-1].strip() for line in open(f'{seg_dirs[0]}/INCAR') if 'POTIM' in line][0])
    total_velocities = read_velocities_from_vdatcar(vdatcar_path, num_atoms)

    for seg_dir in seg_dirs:
        outcar_path = os.path.join(seg_dir, 'OUTCAR')

        if os.path.exists(outcar_path):
            temperatures = read_temperatures_from_outcar(outcar_path)
            energies = extract_total_energies(outcar_path)

            total_temperatures.extend(temperatures)
            total_energies.extend(energies)

    # Truncate the data to the number of steps we want to analyze
    total_temperatures = total_temperatures[:NUM_STEPS_TO_ANALYZE]
    total_energies = total_energies[:NUM_STEPS_TO_ANALYZE]
    target_temperature = total_temperatures[0]
    target_energy = None

    # Clear the report file
    with open('equilibrium_analysis_report.txt', 'w') as file:
        pass
    # Plotting temperature and energy trends
    window_size = 100
    plot_values(total_temperatures, target_temperature, window_size, 'Temperature (K)', 'Temperature per Ionic Step Across Simulation', 'temperature_trend.png')
    plot_values(total_energies, target_energy, window_size, 'Total Energy (eV)', 'Total Energy per Ionic Step Across Simulation', 'total_energy_trend.png')

    # Plotting Fourier transform
    plot_fourier_transform(total_temperatures, timestep_fs, 'Amplitude', 'Fourier Transform of Temperature Fluctuations', 'temperature_fourier_transform.png', 'Temperature Fluctuations')

    # Plotting block averages
    compute_and_plot_block_averages(total_temperatures, num_blocks=10, target_value=target_temperature, x_label='Block Number', y_label='Temperature (K)', title='Block Averages and Std Dev of Temperature', filename='temperature_block_averages.png')
    compute_and_plot_block_averages(total_energies, num_blocks=10, target_value=target_energy, x_label='Block Number', y_label='Energy (eV)', title='Block Averages and Std Dev of Total Energy', filename='total_energy_block_averages.png')

    # Plotting autocorrelation functions
    plot_autocorrelation(total_temperatures, 'Temperature')
    plot_autocorrelation(total_energies, 'Total Energy')
    vacf = compute_vacf(total_velocities)
    plot_vacf(vacf, timestep_fs)

    # Plotting Fourier transform
    plot_fourier_transform(vacf, timestep_fs, 'Amplitude', 'Fourier Transform of Velocity Fluctuations', 'velocity_fourier_transform.png', 'VACF')

if __name__ == "__main__":
    main()
