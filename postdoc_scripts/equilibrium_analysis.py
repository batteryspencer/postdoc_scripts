import matplotlib.pyplot as plt
import os
import re
import numpy as np

# Define the number of steps to analyze
NUM_STEPS_TO_ANALYZE = 10000

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

def autocorrelation(x):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')[-n:]
    result /= np.max(result)
    return result

def plot_values(values, target_value, ylabel, title, file_name):
    plt.figure(figsize=(10, 6))
    steps = range(len(values))
    plt.plot(steps, values, label=ylabel)

    mean_value = np.mean(values)
    std_dev = np.std(values)

    # Print the mean and standard deviation
    print(f"Mean {ylabel}: {mean_value:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")

    plt.axhline(mean_value, color='r', linestyle='dashed', linewidth=1, label=f"Mean {ylabel}: {mean_value:.2f}")
    if target_value is not None:
        plt.axhline(target_value, color='g', linestyle='dashed', linewidth=1, label=f'Target {ylabel}: {target_value:.2f}')

    plt.xlabel('Ionic Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
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

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(y_min, y_max)  # Set the y-axis limits
    plt.legend()
    plt.savefig(filename)

def plot_autocorrelation(values, ylabel):
    plt.figure(figsize=(10, 6))
    acf = autocorrelation(values)
    steps = range(len(acf))
    plt.plot(steps, acf, label='Autocorrelation')
    plt.axhline(0.0, color='r', linestyle='dashed', linewidth=1)

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'{ylabel} Autocorrelation Function')
    plt.legend()
    plt.savefig(f'{ylabel.lower()}_autocorrelation.png')

def main():
    run_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('RUN_')])
    run_dirs.append('.')  # Include the current directory
    total_temperatures = []
    total_energies = []

    for run_dir in run_dirs:
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        oszicar_path = os.path.join(run_dir, 'OSZICAR')

        temperatures = read_temperatures_from_outcar(outcar_path)
        energies = read_energies_from_oszicar(oszicar_path)

        total_temperatures.extend(temperatures)
        total_energies.extend(energies)

    # Truncate the data to the number of steps we want to analyze
    total_temperatures = total_temperatures[:NUM_STEPS_TO_ANALYZE]
    total_energies = total_energies[:NUM_STEPS_TO_ANALYZE]
    target_temperature = total_temperatures[0]
    target_energy = None

    # Plotting temperature and energy trends
    plot_values(total_temperatures, target_temperature, 'Temperature (K)', 'Temperature per Ionic Step Across Simulation', 'temperature_trend.png')
    plot_values(total_energies, target_energy, 'Energy (eV)', 'Energy per Ionic Step Across Simulation', 'energy_trend.png')

    # Plotting block averages
    compute_and_plot_block_averages(temperatures, num_blocks=10, target_value=target_temperature, x_label='Block Number', y_label='Temperature (K)', title='Block Averages and Std Dev of Temperature', filename='temperature_block_averages.png')
    compute_and_plot_block_averages(energies, num_blocks=10, target_value=target_energy, x_label='Block Number', y_label='Energy (eV)', title='Block Averages and Std Dev of Energy', filename='energy_block_averages.png')

    # Plotting autocorrelation functions
    plot_autocorrelation(total_temperatures, 'Temperature')
    plot_autocorrelation(total_energies, 'Energy')

if __name__ == "__main__":
    main()
