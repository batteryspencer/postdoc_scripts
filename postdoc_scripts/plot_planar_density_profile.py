import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Constants
BULK_WATER_DENSITY = 1.0  # g/cm^3, you should set this to your experimental value

def read_contcar(filename='CONTCAR'):
    # Read the CONTCAR file
    atoms = read(filename)
    return atoms

def read_xdatcar(filename='XDATCAR'):
    # Read the XDATCAR file
    atoms = read(filename, index=':')
    return atoms

def planar_density_profile(atoms, excluded_elements, bins=200):
    # Filter out excluded elements
    filtered_atoms = atoms[[atom.symbol not in excluded_elements for atom in atoms]]
    
    # Z-coordinates of the filtered atoms
    z_coords = filtered_atoms.positions[:, 2]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    
    # Histogram of z-coordinates
    hist, bin_edges = np.histogram(z_coords, bins=bins, range=(z_min, z_max), density=True)
    
    # Convert histogram to density
    dz = bin_edges[1] - bin_edges[0]  # thickness of each slab
    volume_per_slab = atoms.cell.volume / bins
    area_per_slab = volume_per_slab / dz
    density = hist * filtered_atoms.get_masses().sum() / area_per_slab  # mass density in g/cm^3

    # Normalize by bulk water density
    normalized_density = density / BULK_WATER_DENSITY
    
    return bin_edges[:-1], normalized_density

def time_averaged_planar_density_profile(atoms, excluded_elements, bins=200):
    for i in range(len(atoms)):
        if i == 0:
            z, density = planar_density_profile(atoms[i], excluded_elements, bins)
            density_sum = density
        else:
            _, density = planar_density_profile(atoms[i], excluded_elements, bins)
            density_sum += density
    density_avg = density_sum / len(atoms)
    return z, density_avg

def plot_density_profile(z, density):
    plt.figure(figsize=(6, 4))
    plt.plot(z, density)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('z-coordinate (Å)')
    plt.ylabel(r'$\rho(z)/\rho_w$')  # Using LaTeX for the ylabel
    plt.title('Planar Averaged Density Profile of Water')
    plt.savefig('density_profile.png')
    
def main():
    atoms = read_contcar()
    # atoms = read_xdatcar()
    excluded_elements=['Pt']
    z, density = planar_density_profile(atoms, excluded_elements)
    # z, density = time_averaged_planar_density_profile(atoms, excluded_elements, bins=200)
    plot_density_profile(z, density)

if __name__ == '__main__':
    main()
