import os
import numpy as np
import matplotlib.pyplot as plt

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

def read_forces_from_report(report_path):
    """
    Read force values from the REPORT file of a VASP simulation.
    """
    forces = []
    with open(report_path, 'r') as file:
        for line in file:
            if "b_m" in line:
                force_value = float(line.split()[1])
                forces.append(force_value)
    return forces

def read_positions_from_xdatcar(xdatcar_path, indices, lattice):
    """
    Read positions of specified atoms from the XDATCAR file of a VASP simulation.
    Convert fractional coordinates to Cartesian coordinates.
    """
    positions = []
    with open(xdatcar_path, 'r') as file:
        lines = file.readlines()
        # Sum the number of atoms from the 7th line
        natoms = sum(map(int, lines[6].strip().split()))
        
        # Read all the lines corresponding to atomic positions
        pos_lines = lines[7:]
        for i in range(0, len(pos_lines), natoms + 1):  # +1 for the timestep header line
            if "Direct configuration=" in pos_lines[i]:
                pos1_frac = np.array([float(x) for x in pos_lines[i + 1 + indices[0]].split()[:3]])
                pos2_frac = np.array([float(x) for x in pos_lines[i + 1 + indices[1]].split()[:3]])
                
                pos1_cart = np.dot(lattice, pos1_frac)
                pos2_cart = np.dot(lattice, pos2_frac)
                
                positions.append((pos1_cart, pos2_cart))
    return positions

def calculate_distances(positions):
    """
    Calculate distances between two atoms for each timestep.
    """
    distances = []
    for pos1, pos2 in positions:
        distance = np.linalg.norm(pos1 - pos2)
        distances.append(distance)
    return distances

def read_lattice_vectors(xdatcar_path):
    """
    Read lattice vectors from the XDATCAR file of a VASP simulation.
    """
    with open(xdatcar_path, 'r') as file:
        lines = file.readlines()
        lattice = np.array([[float(x) for x in lines[2].split()],
                            [float(x) for x in lines[3].split()],
                            [float(x) for x in lines[4].split()]])
    return lattice

def plot_forces_and_distances(constraint_folders, indices):
    """
    Plot force values and distances across the MD simulation trajectory for given constrained values.
    
    Parameters:
    constraint_folders (list of str): List of folder paths for each constrained value.
    indices (tuple of int): Indices of the atoms for which to calculate distances.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    for folder in constraint_folders:
        constrained_value = os.path.basename(folder)
        all_forces = []
        all_distances = []
        
        # Loop through each segment folder
        for segment_folder in sorted(os.listdir(folder)):
            if segment_folder.startswith('seg'):
                report_path = os.path.join(folder, segment_folder, 'REPORT')
                xdatcar_path = os.path.join(folder, segment_folder, 'XDATCAR')
                
                segment_forces = read_forces_from_report(report_path)
                all_forces.extend(segment_forces)
                
                lattice = read_lattice_vectors(xdatcar_path)
                positions = read_positions_from_xdatcar(xdatcar_path, indices, lattice)
                distances = calculate_distances(positions)
                all_distances.extend(distances)
        
        # Plot the force values
        ax1.plot(all_forces, label=f'C-H: {constrained_value.split("_")[0]} Å', color='blue')
        ax2.plot(all_distances, label=f'C-H: {constrained_value.split("_")[0]} Å', color='red')
    
    ax1.set_xlabel('Timestep (fs)', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Force (eV/Å)', fontsize=LABEL_FONTSIZE)
    ax1.set_title('Forces between constrained C-H species\nacross simulation length', fontsize=TITLE_FONTSIZE)
    ax1.legend(fontsize=LEGEND_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)

    ax2.set_xlabel('Timestep (fs)', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('Distance (Å)', fontsize=LABEL_FONTSIZE)
    ax2.set_title('C-Pt distance across simulation length', fontsize=TITLE_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
    
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('md_forces_and_distances_vs_time_for_constraints.png', dpi=300)

constraint_folders = [
    '1.56_1012',
]

indices = (2, 116)

plot_forces_and_distances(constraint_folders, indices)
