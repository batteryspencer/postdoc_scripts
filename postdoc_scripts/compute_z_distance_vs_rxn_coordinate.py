import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

def read_lattice_vectors(xdatcar_path):
    """Read the lattice vectors from an XDATCAR file."""
    with open(xdatcar_path, 'r') as f:
        lines = f.readlines()
        lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])
    return lattice_vectors

def read_z_coordinates_from_xdatcar(xdatcar_path, index1, index2):
    """Read the z-coordinates of the specified indices from an XDATCAR file and compute the z-distance."""
    z_distances = []
    with open(xdatcar_path, 'r') as f:
        lines = f.readlines()
        lattice_vectors = read_lattice_vectors(xdatcar_path)
        z_length = np.linalg.norm(lattice_vectors[2])
        
        # Skip the header lines
        header_lines = 7
        previous_z1 = None
        previous_z2 = None

        i = header_lines
        while i < len(lines):
            if lines[i].strip().startswith("Direct configuration"):
                i += 1  # Skip the "Direct configuration" line
                if i + max(index1, index2) < len(lines):
                    z_coord1 = float(lines[i + index1].split()[2]) * z_length
                    z_coord2 = float(lines[i + index2].split()[2]) * z_length
                    if previous_z1 is not None and previous_z2 is not None:
                        dz1 = z_coord1 - previous_z1
                        dz2 = z_coord2 - previous_z2
                        if dz1 > z_length / 2:
                            z_coord1 -= z_length
                        elif dz1 < -z_length / 2:
                            z_coord1 += z_length
                        if dz2 > z_length / 2:
                            z_coord2 -= z_length
                        elif dz2 < -z_length / 2:
                            z_coord2 += z_length
                    z_distance = abs(z_coord1 - z_coord2)
                    z_distances.append(z_distance)
                    previous_z1 = z_coord1
                    previous_z2 = z_coord2
                i += max(index1, index2) + 1  # Move to the next frame
            else:
                i += 1
    
    return z_distances

def compute_mean_std(z_distances):
    """Compute mean and standard deviation of z-distances."""
    mean_z = np.mean(z_distances)
    std_z = np.std(z_distances)
    return mean_z, std_z

def main():
    base_folder = "."  # Current directory
    index1 = 2  # 0-based index for the first atom
    index2 = 116  # 0-based index for the second atom
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(f) and any(char.isdigit() for char in f)]
    results = []

    for folder in folders:
        segment_folders = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f.startswith("seg")]
        all_z_distances = []

        for seg_folder in segment_folders:
            xdatcar_path = os.path.join(seg_folder, "XDATCAR")
            if os.path.isfile(xdatcar_path):
                z_distances = read_z_coordinates_from_xdatcar(xdatcar_path, index1, index2)
                all_z_distances.extend(z_distances)
            else:
                print(f"XDATCAR not found in {seg_folder}")

        if all_z_distances:
            mean_z, std_z = compute_mean_std(all_z_distances)
            try:
                bond_length = float(folder.split('_')[0])
                results.append((bond_length, mean_z, std_z))
            except ValueError:
                print(f"Error parsing bond length from folder name: {folder}")

    if results:
        results.sort(key=lambda x: x[0])
        df = pd.DataFrame(results, columns=['Bond Length (Å)', 'Mean Z-Distance (Å)', 'Standard Deviation (Å)'])

        # Format the DataFrame to display two decimal places
        pd.options.display.float_format = '{:.2f}'.format

        # Linear trendline
        x = df['Bond Length (Å)'].to_numpy()
        y = df['Mean Z-Distance (Å)'].to_numpy()
        std_dev = df['Standard Deviation (Å)'].to_numpy()
        slope, intercept, r_value, p_value, std_err = linregress(x[1:], y[1:])
        trendline = np.poly1d([slope, intercept])
        
        plt.errorbar(x, y, yerr=std_dev, color='k', fmt='o-', capsize=5, capthick=2)
        plt.plot(x, trendline(x), 'r--')

        # Annotate R² value
        r_squared = f'$R^2$ = {r_value**2:.2f}'
        plt.text(0.80, 0.95, r_squared, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        plt.xlabel("C-H Bond Length (Å)")
        plt.ylabel("Z-Distance from Surface (Å)")
        plt.savefig("z_distance_vs_CH_bond_length.png", dpi=300)

        # Print and save the DataFrame to a text file
        table_string = df.to_string(index=False)
        print(table_string + '\n')
        with open("z_distance_vs_CH_bond_length.txt", "w") as text_file:
            text_file.write(table_string + '\n')
    else:
        print("No valid data found. Please check your folders and files.")

if __name__ == "__main__":
    main()
