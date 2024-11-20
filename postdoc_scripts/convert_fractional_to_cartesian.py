from ase.io import read, write
from ase.geometry import cell_to_cellpar

# Read the POSCAR file
atoms = read('POSCAR', format='vasp')

# Convert the atomic positions from fractional to Cartesian coordinates
atoms.set_scaled_positions(atoms.get_scaled_positions())  # Scales positions to Cartesian

# Save the updated structure back to a new file (optional)
write('POSCAR_cartesian', atoms, format='vasp')

# Print Cartesian coordinates to verify
for atom, position in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
    print(f"{atom}: {position}")

