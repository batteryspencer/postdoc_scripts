import numpy as np
import os
import matplotlib.pyplot as plt
from ase.io import read, write

def compute_distance(coord1, coord2, lattice):
    diff = np.dot(lattice, coord1 - coord2)
    return np.linalg.norm(diff)

def calculate_CH_distances(trajectory, C_index, H_index):
    distances = []

    for frame_index, frame in enumerate(trajectory):
        distance = frame.get_distance(C_index, H_index, mic=True)
        distances.append((distance, frame_index))  # Append a tuple of distance and frame index

    return distances

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

def plot_CH_distances(trajectory, C_index, H_index):

    # Calculate C-H distances
    distances = calculate_CH_distances(trajectory, C_index, H_index)

    # Extract distances and frame indices
    dist_values = [dist[0] for dist in distances]
    frame_indices = [dist[1] for dist in distances]

    plt.plot(frame_indices, dist_values, marker='o', linestyle='-')
    plt.xlabel('Frame Number')
    plt.ylabel('C-H Distance (Å)')
    plt.title('C-H Distance Across Frames in XDATCAR')
    plt.grid(True)
    plt.savefig('C-H_distance_plot.png')

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
    plot_CH_distances(trajectory, C_index, H_index)

if __name__ == "__main__":
    main()

