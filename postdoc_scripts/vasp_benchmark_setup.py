import os
import shutil

# Define the combinations
combinations = [
    (16, 4, 5), (16, 4, 10), (16, 2, 5), (16, 2, 10),
    (16, 8, 5), (16, 8, 10), (32, 2, 5), (32, 2, 10),
    (32, 4, 5), (32, 4, 10), (64, 1, 5), (64, 1, 10),
    (64, 2, 5), (64, 2, 10), (128, 1, 5), (128, 1, 10),
    (None, None, 5), (None, None, 10)
]

# Include original (baseline) configuration without NCORE/NPAR/KPAR
combinations.insert(0, (None, None, None))

# Define the input files
input_files = ['POSCAR', 'POTCAR', 'INCAR', 'KPOINTS', 'submit_cpu.sh']

# Base directory where the folders will be created
base_dir = './benchmark_folders'

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Function to update INCAR with NCORE, NPAR, KPAR definitions
def update_incar(incar_file, ncore, npar, kpar):
    with open(incar_file, 'a') as incar:
        if ncore is not None:
            incar.write(f"\nNCORE = {ncore}")
        if npar is not None:
            incar.write(f"\nNPAR = {npar}")
        if kpar is not None:
            incar.write(f"\nKPAR = {kpar}")

# Function to update job name inside submit_cpu.sh
def update_submit_script(script_file, ncore, npar, kpar):
    job_suffix = []
    if ncore is not None:
        job_suffix.append(f"NCORE={ncore}")
    if npar is not None:
        job_suffix.append(f"NPAR={npar}")
    if kpar is not None:
        job_suffix.append(f"KPAR={kpar}")
    
    job_suffix_str = "_".join(job_suffix)

    # Modify the job name line inside the submit script
    with open(script_file, 'r') as script:
        lines = script.readlines()
    
    # Look for the line with #SBATCH --job-name= and modify it
    with open(script_file, 'w') as script:
        for line in lines:
            if job_suffix and line.startswith('#SBATCH --job-name='):
                line = line.strip() + f"_{job_suffix_str}\n"
            script.write(line)

# Main loop to create directories and copy files
for idx, (ncore, npar, kpar) in enumerate(combinations, start=1):
    # Create folder name (baseline first, then parameterized)
    if ncore is None and npar is None and kpar is None:
        folder_name = f"{idx:02d}_baseline"
    else:
        folder_name = f"{idx:02d}_NCORE={ncore}_NPAR={npar}_KPAR={kpar}"
    folder_path = os.path.join(base_dir, folder_name)
    
    # Create the directory
    os.makedirs(folder_path, exist_ok=True)
    
    # Copy the input files to the new directory
    for file in input_files:
        shutil.copy(file, folder_path)
    
    # Modify INCAR file
    incar_path = os.path.join(folder_path, 'INCAR')
    update_incar(incar_path, ncore, npar, kpar)
    
    # Modify submit_cpu.sh file
    submit_script_path = os.path.join(folder_path, 'submit_cpu.sh')
    update_submit_script(submit_script_path, ncore, npar, kpar)

print(f"All directories created and files copied successfully in {base_dir}.")

