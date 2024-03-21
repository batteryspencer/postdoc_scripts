import os
import glob
import numpy as np

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

# Activation barrier using the trapezoidal rule
activation_barrier = trapz(y, x)

# Calculate the uncertainty in the activation barrier
segment_widths = np.diff(x)
segment_errors = np.round(std_dev[:-1] * segment_widths, 2)
activation_barrier_error = np.round(np.sqrt(np.sum(segment_errors[1:-1] * segment_errors[1:-1])), 2)

# Print data in a table format and save it to a text file
table_string = df.to_string(index=False)
results_string = f"Activation Barrier (Area under the curve): {activation_barrier:.2f} ± {activation_barrier_error:.2f} eV\n"
print(table_string)
print(results_string)
with open("pmf_analysis_results.txt", "w") as text_file:
    text_file.write(table_string)

