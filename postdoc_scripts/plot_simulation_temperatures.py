import matplotlib.pyplot as plt
import os
import re

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

def plot_temperatures(total_temperatures):
    plt.figure(figsize=(10, 6))
    steps = range(len(total_temperatures))
    plt.plot(steps, total_temperatures)

    plt.xlabel('Ionic Step')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature per Ionic Step Across Simulation')
    plt.show()

def main():
    run_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('RUN_')])
    run_dirs.append('.')  # Include the current directory
    total_temperatures = []

    for run_dir in run_dirs:
        outcar_path = os.path.join(run_dir, 'OUTCAR')
        temperatures = read_temperatures_from_outcar(outcar_path)
        total_temperatures.extend(temperatures)

    plot_temperatures(total_temperatures[:10000])

if __name__ == "__main__":
    main()

