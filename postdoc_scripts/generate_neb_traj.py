import os
from ase.io import read, Trajectory
import gzip

# Function to read gzipped OUTCAR and return an Atoms object
def read_outcar(path):
    # Placeholder for reading OUTCAR directly
    # You need to implement this part
    with open(path, 'r') as f:
        atoms = read(f, format='vasp-out')
    return atoms

def read_gzipped_outcar(path):
    with gzip.open(path, 'rt') as f:  # Open in read-text mode
        atoms = read(f, format='vasp-out')
    return atoms

# Number of intermediate images
num_intermediate_images = 5

# Generate image directories
# Start with '00' (initial state), include intermediate images, and then the final state
image_dirs = ['00'] + [f'{i:02d}' for i in range(1, num_intermediate_images + 1)]
image_dirs.append(f'{num_intermediate_images + 1:02d}')  # Dynamically add the final state

images = []

# Read each OUTCAR.gz and append to images list
for img_dir in image_dirs:
    outcar_path = os.path.join(img_dir, 'OUTCAR')
    outcar_gz_path = os.path.join(img_dir, 'OUTCAR.gz')

    if os.path.exists(outcar_path):
        image = read_outcar(outcar_path)
    elif os.path.exists(outcar_gz_path):
        image = read_gzipped_outcar(outcar_gz_path)
    else:
        print(f"No OUTCAR or OUTCAR.gz found in {img_dir}")
        continue
    images.append(image)

# Write to a trajectory file
traj_file = 'neb.traj'
with Trajectory(traj_file, 'w') as traj:
    for image in images:
        traj.write(image)

print(f'Trajectory written to {traj_file}')

