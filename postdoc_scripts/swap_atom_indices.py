from ase import Atoms
from ase.io import read, write

# Read the JSON file as an ASE Atoms object
atoms = read('POSCAR_initial')

# Define a list of index pairs to swap
index_pairs = [(49, 52)]  # Add more pairs as needed

# Iterate over the index pairs and perform swaps
for index1, index2 in index_pairs:
    # Get the positions of the atoms to be swapped
    position1 = atoms[index1].position.copy()
    position2 = atoms[index2].position.copy()

    # Swap the positions
    atoms[index1].position = position2
    atoms[index2].position = position1

# Write the modified structure to a new JSON file
write('modified_POSCAR_initial', atoms, format='vasp')

print(f"Swapped specified index pairs")

