import matplotlib.pyplot as plt
import os

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

def plot_energies(total_energies):
    plt.figure(figsize=(10, 6))
    steps = range(len(total_energies))
    plt.plot(steps, total_energies)

    plt.xlabel('Ionic Step')
    plt.ylabel('Total Energy (eV)')
    plt.title('Total Energy per Ionic Step Across Simulation')
    plt.show()

def main():
    run_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('RUN_')])
    run_dirs.append('.')  # Include the current directory
    total_energies = []

    for run_dir in run_dirs:
        oszicar_path = os.path.join(run_dir, 'OSZICAR')
        energies = read_energies_from_oszicar(oszicar_path)
        total_energies.extend(energies)

    plot_energies(total_energies[:10000])

if __name__ == "__main__":
    main()

