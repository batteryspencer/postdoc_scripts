import pandas as pd
import matplotlib.pyplot as plt
import os

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

# Directories containing the snapshots
snapshot_dirs = [f'snapshot_{i}' for i in range(1, 16)]

# Vacuum level (assumed constant, modify if needed)
vacuum_level = 4.847  # eV, from VASPkit from vacuum calculation

# Input work function values from VASPkit manually (in eV)
work_function_values = [
    -0.227,  0.006, -0.738, -0.299, -0.152,
     0.420,  0.647,  0.316, -0.652,  0.229,
    -1.336, -0.766,  0.757, -0.006, -0.428
]

# Initialize variables for accumulating data and properties
average_data = None
total_work_function = sum(work_function_values)
total_fermi_level = 0.0
num_snapshots = len(snapshot_dirs)

# Loop through each snapshot directory and accumulate data
for i, snapshot_dir in enumerate(snapshot_dirs):
    file_path = os.path.join(snapshot_dir, 'PLANAR_AVERAGE.dat')
    outcar_path = os.path.join(snapshot_dir, 'OUTCAR')
    if os.path.exists(file_path):
        # Load the data, skipping the first line (header)
        data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None, names=['Position', 'Potential'])
        if average_data is None:
            average_data = data
        else:
            average_data['Potential'] += data['Potential']
    
    if os.path.exists(outcar_path):
        with open(outcar_path, 'r') as outcar_file:
            for line in outcar_file:
                if 'E-fermi' in line:
                    total_fermi_level += float(line.split()[2])
                    break  # Assuming E-fermi appears only once in OUTCAR

# Calculate the average potential
average_data['Potential'] /= num_snapshots

# Calculate the average work function and Fermi level
average_work_function = total_work_function / num_snapshots
average_fermi_level = total_fermi_level / num_snapshots

# Calculate the manually computed work function
calculated_work_function = vacuum_level - average_fermi_level

# Print the work functions
print(f'\nManually Calculated Work Function: {calculated_work_function:.3f} eV')
print(f'Average Work Function: {average_work_function:.3f} eV')

# Standard Hydrogen Electrode (SHE) potential relative to vacuum level
she_potential = 4.44  # eV

# Calculate the potential relative to SHE
calculated_she_potential = calculated_work_function - she_potential
average_she_potential = average_work_function - she_potential

# Print the SHE potentials
print(f'\nSHE Potential from Manually Calculated Work Function: {calculated_she_potential:.3f} eV')
print(f'SHE Potential from Average Work Function: {average_she_potential:.3f} eV')

# Plot the electrostatic potential profile
plt.figure(figsize=(10, 6))
plt.plot(average_data['Position'], average_data['Potential'], color='blue', label='Time-Averaged Electrostatic Potential')
plt.axhline(y=vacuum_level, color='red', linestyle='dashed', label=f'Vacuum Level ({vacuum_level} eV)')
plt.axhline(y=average_fermi_level, color='green', linestyle='dashed', label=f'Fermi Level ({average_fermi_level:.3f} eV)')
plt.xlabel('Distance (Å)', fontsize=LABEL_FONTSIZE)
plt.ylabel('Potential (eV)', fontsize=LABEL_FONTSIZE)
plt.title('Time-Averaged Electrostatic Potential Profile', fontsize=TITLE_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
plt.legend(fontsize=LEGEND_FONTSIZE, loc='lower left', bbox_to_anchor=(0.68, 0.1), fancybox=True, shadow=True, ncol=1)
plt.savefig('time_averaged_electrostatic_potential_plot.png', dpi=300)
