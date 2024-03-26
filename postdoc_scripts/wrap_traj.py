from ase.io import read, write
from ase.io.vasp import read_vasp_xdatcar

# Step 1: Read the XDATCAR file
trajectory = read_vasp_xdatcar('XDATCAR', index=slice(None))

# Step 2: Wrap atoms for each frame in the trajectory
for atoms in trajectory:
    atoms.wrap()

# Step 3: Write the wrapped trajectory to a new XDATCAR file (or another format)
write('wrapped_XDATCAR', trajectory, format='vasp-xdatcar')
