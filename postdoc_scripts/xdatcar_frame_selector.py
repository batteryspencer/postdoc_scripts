import numpy as np
import os
import matplotlib.pyplot as plt
from ase.io import read, write

def compute_distance(coord1, coord2, lattice):
    diff = np.dot(lattice, coord1 - coord2)
    return np.linalg.norm(diff)

def calculate_CH_distances(trajectory, C_index, H_index):
    """
    Calculate C-H distances across all frames in a trajectory.

    Parameters:
    trajectory (list): List of atomic configurations.
    C_index (int): Index of the carbon atom.
    H_index (int): Index of the hydrogen atom.
    """        
    return [frame.get_distance(C_index, H_index, mic=True) for frame in trajectory]

def find_target_frames(trajectory, C_H_targets, C_index, H_index, Pt_index, initial_tolerance, secondary_tolerance):

    target_frames = []
    for target in C_H_targets:
        closest_frame = find_closest_frame(trajectory, C_index, H_index, Pt_index, target, initial_tolerance)

        # If no frame is found with initial tolerance, try with secondary tolerance
        if not closest_frame:
            closest_frame = find_closest_frame(trajectory, C_index, H_index, Pt_index, target, secondary_tolerance)

        if closest_frame:
            target_frames.append(closest_frame)

    if not target_frames:
        print("No frames found for the given targets.")

    return target_frames

def find_closest_frame(trajectory, C_index, H_index, Pt_index, target_distance, tolerance):
    closest_frame = None
    min_distance = float('inf')

    for frame_index, frame in enumerate(trajectory):
        C_H_distance = frame.get_distance(C_index, H_index, mic=True)
        C_Pt_distance = frame.get_distance(C_index, Pt_index, mic=True)

        if abs(C_H_distance - target_distance) <= tolerance and C_Pt_distance < min_distance:
            closest_frame = (frame_index, C_H_distance)
            min_distance = C_Pt_distance

    return closest_frame

def create_poscar_directories(trajectory, frame_data, base_dir):
    """
    Create directories and write POSCAR files for specified frames.

    Parameters:
    trajectory (list): List of atomic configurations.
    frame_data (list of tuples): Tuples of frame indices and corresponding C-H bond lengths.
    base_dir (str): Base directory to create frame directories.
    """
    for frame_index, bond_length in frame_data:
        rounded_bond_length = np.round(bond_length, 2)  # Round the bond length to 2 decimal places
        dir_name = os.path.join(base_dir, f"{rounded_bond_length:.2f}_{frame_index}")
        os.makedirs(dir_name, exist_ok=True)
        write(f'{dir_name}/POSCAR', trajectory[frame_index], format='vasp')  # Use ASE to write the POSCAR file

def plot_CH_distances(trajectory, C_index, H_index, figname='C-H_distance_plot.png', show_plot=True):
    """
    Plot and optionally save the C-H bond distance against the frame index.

    Parameters:
    trajectory (list): List of atomic configurations.
    C_index (int): Index of the carbon atom.
    H_index (int): Index of the hydrogen atom.
    figname (str): The filename to save the plot.
    show_plot (bool): If True, display the plot; if False, don't display.
    """
    # Calculate C-H distances
    CH_distances = calculate_CH_distances(trajectory, C_index, H_index)

    # Extract frame indices
    frame_indices = np.arange(len(CH_distances))

    # Plotting
    plt.plot(frame_indices, CH_distances, marker='o', linestyle='-')
    plt.xlabel('Frame Index')
    plt.ylabel('C-H Distance (Å)')
    plt.title('C-H Distance vs Frame Index')
    plt.grid(True)

    # Saving the plot
    plt.savefig(figname)

    # Displaying the plot
    if show_plot:
        plt.show()

    # Close the plot to free up memory
    plt.close()

def main():
    # Specified inputs
    C_index, H_index, Pt_index = 2, 35, 116
    C_H_start = 0.70  # Start distance for C-H
    C_H_end = 1.70  # End distance for C-H
    num_images = 21  # Number of target images
    initial_tolerance = 0.01  # Initial Tolerance level
    secondary_tolerance = 0.02  # Secondary Tolerance level

    # Read trajectory
    trajectory = read("XDATCAR", index=':', format='vasp-xdatcar')

    # Find target frames
    C_H_targets = np.linspace(C_H_start, C_H_end, num_images, endpoint=True)
    target_frames = find_target_frames(trajectory, C_H_targets, C_index, H_index, Pt_index, initial_tolerance, secondary_tolerance)

    # Create directories and write POSCAR
    create_poscar_directories(trajectory, target_frames, os.getcwd())

    # Plot distances
    plot_CH_distances(trajectory, C_index, H_index, figname='C-H_distance_plot.png', show_plot=False)

if __name__ == "__main__":
    main()

