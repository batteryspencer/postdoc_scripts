import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Define constants for plot aesthetics
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

# Define indices
propane_center_index = 0
top_layer_pt_indices = [96, 101, 106, 111, 116, 121, 126, 131, 136]
bottom_layer_pt_indices = [92, 97, 102, 107, 112, 117, 122, 127, 132]
window_size = 100

# Pairs of C-H bonds in propane
C_H_pairs = [(0, 29), (0, 30), (1, 31), (1, 33), (1, 34), (2, 32), (2, 35), (2, 36)]

# Function to calculate the distance from the center of the propane molecule to the Pt surface
def min_distance_to_surface(propane_center_z, atoms, cell):
    top_positions = atoms.get_positions(wrap=True)[top_layer_pt_indices]
    bottom_positions = atoms.get_positions(wrap=True)[bottom_layer_pt_indices]

    # Calculate average z-coordinate for top and bottom Pt layers
    avg_top_z = np.mean(top_positions[:, 2])
    avg_bottom_z = np.mean(bottom_positions[:, 2])

    # Calculate z-distance considering PBC
    dist_top = np.abs(np.mod(propane_center_z - avg_top_z + cell[2, 2] / 2, cell[2, 2]) - cell[2, 2] / 2)
    dist_bottom = np.abs(np.mod(propane_center_z - avg_bottom_z + cell[2, 2] / 2, cell[2, 2]) - cell[2, 2] / 2)

    return min(dist_top, dist_bottom)

# Function to calculate the force projection along the direction connecting atoms in a pair
def force_projection(along_vector, force_vector):
    unit_vector = along_vector / np.linalg.norm(along_vector)
    return np.dot(force_vector, unit_vector)

# Function to calculate the moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Lists to store data for plotting
forces = []
distances = []

# Loop over segment folders
for segment in sorted(os.listdir('.')):
    if os.path.isdir(segment) and segment.startswith('seg'):
        seg_path = os.path.join('.', segment)

        # Read the VASP files in the segment folder
        atoms_list = read(os.path.join(seg_path, 'vasprun.xml'), index=':')
        
        for atoms in atoms_list:
            total_force_projection = 0
            for c_index, h_index in C_H_pairs:
                pos_c = atoms.positions[c_index]
                pos_h = atoms.positions[h_index]
                force_c = atoms.get_forces()[c_index]
                force_h = atoms.get_forces()[h_index]

                # Calculate the direction vector between the C and H atoms
                direction_vector = pos_h - pos_c

                # Calculate the projections of the forces along the direction vector
                proj_force_c = force_projection(direction_vector, force_c)
                proj_force_h = force_projection(direction_vector, force_h)

                # Take the average of these projections as the force to be recorded
                avg_force_projection = (proj_force_c + proj_force_h) / 2
                total_force_projection += avg_force_projection
            
            avg_total_force_projection = total_force_projection / len(C_H_pairs)
            forces.append(avg_total_force_projection)

            # Calculate the distance of the propane molecule to the Pt surface
            propane_center_z = atoms.get_positions(wrap=True)[propane_center_index, 2]
            distance_to_surface = min_distance_to_surface(propane_center_z, atoms, atoms.cell)
            distances.append(distance_to_surface)

# Create timesteps array assuming 1 fs timestep
timesteps = np.arange(1, len(forces) + 1)

# Convert lists to numpy arrays for plotting
forces = np.array(forces)
distances = np.array(distances)

# Calculate the moving average of the forces
forces_moving_avg = moving_average(forces, window_size)
timesteps_moving_avg = timesteps[:len(forces_moving_avg)]

# Create the 2x1 subplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot forces
ax1.plot(timesteps, forces, 'gray', label='Force')
ax1.plot(timesteps_moving_avg, forces_moving_avg, 'black', label='Moving Average')
ax1.set_title("Forces between C-H bonds\nin propane along the simulation length", fontsize=TITLE_FONTSIZE)
ax1.set_ylabel("Force (eV/Å)", fontsize=LABEL_FONTSIZE)
ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)

# Plot distances
ax2.plot(timesteps, distances, 'r-')
ax2.set_title("Distance of propane molecule from Pt surface", fontsize=TITLE_FONTSIZE)
ax2.set_xlabel("Timestep (fs)", fontsize=LABEL_FONTSIZE)
ax2.set_ylabel("Distance (Å)", fontsize=LABEL_FONTSIZE)
ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0, 1, 0.975])
plt.subplots_adjust(hspace=0.4)
plt.savefig("Forces_and_Distances.png", dpi=300)
