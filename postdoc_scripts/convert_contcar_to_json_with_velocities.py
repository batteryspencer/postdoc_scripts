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
start_line_velocities = 137  # Adjust this number as needed

# Skip blank lines if present
while not lines[start_line_velocities].strip():
    start_line_velocities += 1

# Reading Velocities
velocities = []
for line in lines[start_line_velocities:start_line_velocities + len(atoms)]:
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

