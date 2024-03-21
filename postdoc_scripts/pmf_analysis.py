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
