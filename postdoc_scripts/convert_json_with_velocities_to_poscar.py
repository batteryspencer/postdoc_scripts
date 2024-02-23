import os
import math
from ase.io import read, write

def custom_format_adjusted(v):
    if v == 0:
        exponent = -1
    else:
        exponent = int(math.floor(math.log10(abs(v))))
    mantissa = v / (10 ** exponent)
    if v < 0:
        return '{:.8f}E{:+03d}'.format(mantissa / 10, exponent + 1)
    else:
        return ' {:.8f}E{:+03d}'.format(mantissa / 10, exponent + 1)

def convert_json_to_poscar(json_path, poscar_path):
    # Check if JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        return

    # Load the atoms from the JSON file
    try:
        atoms = read(json_path)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    # Write the POSCAR file with direct coordinates for the positions
    write(poscar_path, atoms, format='vasp', direct=True)

    # Retrieve and check velocities
    if 'momenta' in atoms.arrays:
        velocities = atoms.get_velocities()
        if len(velocities) != len(atoms):
            print("Error: Inconsistent number of velocity entries.")
            return
    else:
        print("Warning: No velocity data found in JSON file.")
        return

    # Append the velocities to the POSCAR file
    with open(poscar_path, 'a') as poscar:
        poscar.write(' \n')  # Write a blank line before velocities
        for velocity in velocities:
            formatted_velocity = ' ' + ' '.join([custom_format_adjusted(v) for v in velocity])
            line = formatted_velocity + '\n'
            poscar.write(line)

# Example usage
json_path = 'restart_with_velocities.json'  # Input JSON file
poscar_path = 'POSCAR'      # Output POSCAR file
convert_json_to_poscar(json_path, poscar_path)

