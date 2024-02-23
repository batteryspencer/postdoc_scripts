import numpy as np
from ase.io import read, write

# Replace with your CONTCAR file path
contcar_path = 'CONTCAR'

# Read the CONTCAR file
atoms = read(contcar_path, format='vasp')

# Manually read velocities from CONTCAR
with open(contcar_path, 'r') as file:
    lines = file.readlines()

# The line number where velocities start (adjust based on your file)
velocity_start_line_number = 137  # Adjust this number as needed
velocity_start_line_index = velocity_start_line_number - 1

# Skip blank lines if present
while not lines[velocity_start_line_index].strip():
    velocity_start_line_number += 1
    velocity_start_line_index += 1

# Reading Velocities
velocities = []
for line in lines[velocity_start_line_index:velocity_start_line_index + len(atoms) + 1]:
    if line.strip():  # This checks if the line is not empty
        velocities.append([float(v) for v in line.split()])

# Check if the length of velocities array matches the number of atoms
if len(velocities) == len(atoms):
    # Convert velocities to a NumPy array
    velocities_array = np.array(velocities)

    # Assign velocities to atoms object
    atoms.set_velocities(velocities_array)

    # Write to a JSON file
    atoms.write('restart_with_velocities.json')
else:
    print("Error: Inconsistent number of velocity entries.")

