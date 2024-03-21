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

