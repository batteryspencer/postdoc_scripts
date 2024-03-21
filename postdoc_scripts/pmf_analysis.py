import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.integrate import trapz

# This function reads the force_stats_report.txt and extracts the values
def read_force_stats(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) > 2:  # Adjusted to account for the MD steps line
            cv_data = lines[1].split(',')
            md_steps = int(lines[2].split('=')[1].strip().split()[0])  # Extracts the number of MD steps
            return float(cv_data[0].strip()), -1*np.around(float(cv_data[1].strip()), 2), float(cv_data[2].strip()), md_steps
    return None

# This dictionary will hold our data
data = {'Constrained_Length': [], 'Mean_Force': [], 'Standard_Deviation': [], 'MD_Steps': []}

# Assuming your directories are named in the '1.06_793' format and are in the current working directory
for folder in glob.glob("[0-9].[0-9][0-9]_*"):
    file_path = os.path.join(folder, 'force_stats_report.txt')
    if os.path.isfile(file_path):
        length, force, std_dev, md_steps = read_force_stats(file_path)
        data['Constrained_Length'].append(length)
        data['Mean_Force'].append(force)
        data['Standard_Deviation'].append(std_dev)
        data['MD_Steps'].append(md_steps)

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Sort the DataFrame based on the constrained length
df = df.sort_values(by=['Constrained_Length'])

# Assuming 'df' is the DataFrame with your data sorted by 'Constrained_Length'
x = df['Constrained_Length'].to_numpy()
y = df['Mean_Force'].to_numpy()
std_dev = df['Standard_Deviation'].to_numpy()

# Find zero crossings
zero_crossings = np.where(np.diff(np.sign(y)))[0]

# Initialize the variables to store the results
activation_barrier = None
activation_barrier_error = None

# Check if there are at least two zero crossings to define bounds
if len(zero_crossings) >= 2:
    # Use the first two zero crossings as an example
    start, end = zero_crossings[0], zero_crossings[1]

    # Compute the activation barrier (area under the curve) between these two crossings
    activation_barrier = trapz(y[start:end + 1], x[start:end + 1])

    # Calculate the uncertainty in the activation barrier
    segment_widths = np.diff(x[start:end + 1])
    segment_errors = np.round(std_dev[start:end] * segment_widths, 2)
    activation_barrier_error = np.round(np.sqrt(np.sum(segment_errors ** 2)), 2)

if activation_barrier is not None and activation_barrier_error is not None:
    results_string = f"Activation Barrier (Area under the curve): {activation_barrier:.2f} ± {activation_barrier_error:.2f} eV\n"
else:
    results_string = "Not enough zero crossings found to compute the area and its error."

# Print data in a table format and save it to a text file
table_string = df.to_string(index=False)
print(table_string)
print(results_string)
with open("pmf_analysis_results.txt", "w") as text_file:
    text_file.write(table_string)

# Plotting
plt.figure(figsize=(10, 6))
ax = plt.gca()
plt.errorbar(df['Constrained_Length'], df['Mean_Force'], yerr=df['Standard_Deviation'], fmt='o', color='black', ecolor='black', capthick=2)

# Create a polygon to fill the area under the curve
verts = [(df['Constrained_Length'].iloc[0], 0)] + list(zip(df['Constrained_Length'], df['Mean_Force'])) + [(df['Constrained_Length'].iloc[-1], 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.1')
ax.add_patch(poly)

plt.title('Mean Force vs. Constrained Bond Length')
plt.xlabel('Constrained Bond Length (Å)', fontsize=12)
plt.ylabel('Mean Force (eV/Å)', fontsize=12)
plt.savefig('mean_force_plot.png', dpi=300, bbox_inches='tight')
