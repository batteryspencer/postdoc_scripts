import os
import matplotlib.pyplot as plt

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

def plot_forces(constraint_folders):
    """
    Plot force values across the MD simulation trajectory for given constrained values.
    
    Parameters:
    constraint_folders (list of str): List of folder paths for each constrained value.
    """
    plt.figure(figsize=(10, 6))
    
    for folder in constraint_folders:
        constrained_value = os.path.basename(folder)
        all_forces = []
        
        # Loop through each segment folder
        for segment_folder in sorted(os.listdir(folder)):
            if segment_folder.startswith('seg'):
                report_path = os.path.join(folder, segment_folder, 'REPORT')
                segment_forces = read_forces_from_report(report_path)
                all_forces.extend(segment_forces)
        
        # Plot the force values
        plt.plot(all_forces, label=f'C-H: {constrained_value.split("_")[0]} Å')
    
    plt.xlabel('Timestep')
    plt.ylabel('Force Value (eV/Å)')
    plt.title('Force values across MD simulation trajectory')
    plt.legend()
    plt.savefig('md_forces_vs_time_for_constraints.png', dpi=300)

constraint_folders = [
    '1.25_1104',
    '1.50_1141',
    '1.66_1153'
]

plot_forces(constraint_folders)
