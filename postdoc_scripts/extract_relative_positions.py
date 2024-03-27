from ase.io import read
import numpy as np

def get_relative_positions_excluding_elements(file_path, original_reference_index, excluded_element):
    # Read the file using ASE
    atoms = read(file_path, format='vasp')
    
    # Filter out the atoms that are not the excluded element and update indices
    filtered_atoms = []
    updated_indices = []
    for i, atom in enumerate(atoms):
        if atom.symbol != excluded_element:
            filtered_atoms.append(atom)
            updated_indices.append(i)
    
    # Adjust the reference index according to the filtered list
    if original_reference_index in updated_indices:
        new_reference_index = updated_indices.index(original_reference_index)
    else:
        raise IndexError("The original reference index does not exist after exclusion.")

    # Get the symbols and positions of the non-excluded atoms
    symbols = [atom.symbol for atom in filtered_atoms]
    positions = np.array([atom.position for atom in filtered_atoms])

    # Calculate relative positions from the new reference atom
    reference_position = positions[new_reference_index]
    relative_positions = positions - reference_position

    # Round the positions to one decimal place
    relative_positions_list = [tuple(np.around(pos, decimals=1)) for pos in relative_positions]

    # Order the elements and their relative positions
    ordered_elements = [symbols[new_reference_index]] + symbols[:new_reference_index] + symbols[new_reference_index+1:]
    ordered_positions = [(0.0, 0.0, 0.0)] + relative_positions_list[:new_reference_index] + relative_positions_list[new_reference_index+1:]

    return ordered_elements, ordered_positions

# File paths
contcar_file_path = 'CONTCAR'  # Path to the CONTCAR file

# Assuming the original reference index is 45 as provided
original_reference_index = 45

# Exclude Pt atoms and get the ordered relative positions and elements
elements_contcar, relative_positions_list_contcar = get_relative_positions_excluding_elements(contcar_file_path, original_reference_index, 'Pt')

# Output the results
print("Elements from CONTCAR (excluding Pt):")
print(elements_contcar)
print("Relative position list from CONTCAR (excluding Pt):")
print(relative_positions_list_contcar)

