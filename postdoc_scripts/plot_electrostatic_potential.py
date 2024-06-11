import pandas as pd
import matplotlib.pyplot as plt

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

# Define the file name
file_name = 'PLANAR_AVERAGE.dat'

# Load the data, skipping the first line (header)
data = pd.read_csv(file_name, delim_whitespace=True, skiprows=1, header=None, names=['Position', 'Potential'])

# Define the vacuum and Fermi levels from their respective sources
vacuum_level = 4.847  # eV, from VASPkit
fermi_level = -0.9412  # eV, from OUTCAR
vaspkit_work_function = 5.789  # eV, directly from VASPkit

# Calculate the manually computed work function
calculated_work_function = vacuum_level - fermi_level

# Print the work functions
print(f'Manually Calculated Work Function: {calculated_work_function:.3f} eV')
print(f'VASPkit Work Function: {vaspkit_work_function:.3f} eV')

# Plot the electrostatic potential profile
plt.figure(figsize=(10, 6))
plt.plot(data['Position'], data['Potential'], color='blue', label='Electrostatic Potential')
plt.axhline(y=vacuum_level, color='red', linestyle='dashed', label=f'Vacuum Level ({vacuum_level} eV)')
plt.axhline(y=fermi_level, color='green', linestyle='dashed', label=f'Fermi Level ({fermi_level} eV)')
plt.xlabel('Distance (Å)', fontsize=LABEL_FONTSIZE)
plt.ylabel('Potential (eV)', fontsize=LABEL_FONTSIZE)
plt.title('Electrostatic Potential Profile with Vacuum and Fermi Levels', fontsize=TITLE_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
plt.legend(fontsize=LEGEND_FONTSIZE, loc='lower left', bbox_to_anchor=(0.68, 0.1), fancybox=True, shadow=True, ncol=1)
plt.savefig('electrostatic_potential_plot.png', dpi=300)
