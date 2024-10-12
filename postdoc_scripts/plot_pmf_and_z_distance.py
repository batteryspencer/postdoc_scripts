import matplotlib.pyplot as plt
import pandas as pd

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

# Define a function to read the data correctly
def read_data(file_path, column_names, skip_footer=0):
    data = []
    footer = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[:-skip_footer] if skip_footer else lines:
            if line.strip() and all(c.isdigit() or c == '.' or c == '-' or c == ' ' for c in line.strip().replace('.', '').replace('-', '').replace('e', '')):
                data.append(line.strip().split())
        footer = lines[-skip_footer:] if skip_footer else []
    df = pd.DataFrame(data, columns=column_names, dtype=float)
    return df, footer

# Define column names for the two files
pmf_columns = ['Constrained_Bond_Length', 'Mean_Force', 'Standard_Deviation', 'MD_Steps']
z_distance_columns = ['Bond', 'Mean', 'Standard']

# Read the first file with the specified columns and footer
pmf_analysis_df, pmf_footer = read_data('pmf_analysis_results.txt', pmf_columns, skip_footer=2)

# Read the second file with the specified columns and footer
z_distance_df, z_distance_footer = read_data('z_distance_vs_CH_bond_length.txt', z_distance_columns, skip_footer=2)

# Extract the Equilibrium Bond Distance for the Initial State and Activation Distance from the footer
initial_state_bond_length = float(z_distance_footer[0].split(':')[1].split('Å')[0].strip())
activation_distance = float(z_distance_footer[1].split(':')[1].split('Å')[0].strip())

# Create a figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot the PMF data with filled error bars
ax1.errorbar(pmf_analysis_df['Constrained_Bond_Length'], pmf_analysis_df['Mean_Force'],
             yerr=pmf_analysis_df['Standard_Deviation'], fmt='o-', ecolor='black', capsize=5, label='Mean Force', color='black')
ax1.set_xlabel('Bond Length (Å)', fontsize=LABEL_FONTSIZE)
ax1.set_ylabel('Mean Force', fontsize=LABEL_FONTSIZE)
ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
ax1.axvline(x=initial_state_bond_length, color='blue', linestyle='--')

# Create a polygon to fill the area under the curve for PMF data
ax1.fill_between(pmf_analysis_df['Constrained_Bond_Length'], pmf_analysis_df['Mean_Force'],
                 facecolor='0.9', edgecolor='0.1')

# Plot the Z-Distance data with filled error bars
ax2.plot(z_distance_df['Bond'], z_distance_df['Mean'], 'o-', color='blue', label='Mean Z-Distance')
ax2.fill_between(z_distance_df['Bond'],
                 z_distance_df['Mean'] - z_distance_df['Standard'],
                 z_distance_df['Mean'] + z_distance_df['Standard'],
                 color='blue', alpha=0.2)
ax2.set_xlabel('Bond Length (Å)', fontsize=LABEL_FONTSIZE)
ax2.set_ylabel('Mean Z-Distance (Å)', fontsize=LABEL_FONTSIZE, color='blue')
ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR, colors='blue')
ax2.axvline(x=initial_state_bond_length, color='blue', linestyle='--')
ax2.text(initial_state_bond_length + 0.01, activation_distance + 1.5, 
         f'C-H Bond Activation: {activation_distance:.2f} Å from Surface', 
         fontsize=LABEL_FONTSIZE, color='blue')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('pmf_z_distance_plot.png')
