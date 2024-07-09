import os

# Define the parameters
xdatcar_file = 'XDATCAR_combined'  # Path to your XDATCAR file
interval = 1000  # Interval for 1 ps (1 fs timestep, so 1000 steps for 1 ps)
output_dir = 'snapshots'  # Directory to store the POSCAR files
num_atoms = 45 + 54 + 27  # Total number of atoms (Pt + H + O)

# Header for POSCAR file
header = """Pt  H  O    
    1.000000000000000    
     8.4852813999999999    0.0000000000000000    0.0000000000000000
     4.2426406999999999    7.3484692000000003    0.0000000000000000
     0.0000000000000000    0.0000000000000000   27.1376042999999996
   Pt   H    O   
    45    54    27  
Direct
"""

# Function to read XDATCAR file and extract snapshots
def read_xdatcar(xdatcar_file, start_step, interval, num_snapshots):
    with open(xdatcar_file, 'r') as file:
        lines = file.readlines()

    snapshots = []
    for i in range(num_snapshots):
        step = start_step + i * interval
        start_line = 8 + step * (num_atoms + 1)  # Account for the 'Direct configuration=' line
        end_line = start_line + num_atoms
        snapshot = lines[start_line:end_line]
        snapshots.append(snapshot)

    return header, snapshots

# Create directories and write POSCAR files
def write_poscar_files(header, snapshots, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, snapshot in enumerate(snapshots):
        snapshot_dir = os.path.join(output_dir, f'snapshot_{i+1}')
        os.makedirs(snapshot_dir, exist_ok=True)
        poscar_file = os.path.join(snapshot_dir, 'POSCAR')
        with open(poscar_file, 'w') as file:
            file.write(header)
            file.writelines(snapshot)

# Extract and write snapshots
num_snapshots = 15
start_step = 5000  # Starting from 5 ps mark for the last 15 ps
header, snapshots = read_xdatcar(xdatcar_file, start_step, interval, num_snapshots)
write_poscar_files(header, snapshots, output_dir)

print(f'{num_snapshots} snapshots have been extracted and saved as POSCAR files in {output_dir} directory.')

