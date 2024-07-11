from ase.io import read, write
from ase.io.vasp import read_vasp, write_vasp

# Read the POSCAR file
structure = read_vasp('POSCAR')

# Wrap atoms in the structure
structure.wrap()

# Write the wrapped structure to a new POSCAR file
write_vasp('wrapped_POSCAR', structure, direct=True)

