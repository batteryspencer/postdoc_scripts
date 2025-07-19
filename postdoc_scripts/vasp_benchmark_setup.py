import os
import shutil

# # ------------------------------------------------------------------
# # USER OPTIONS: choose which combos to include (no argparse)
# # ------------------------------------------------------------------
# 1) Full sweep: all (ncore, npar, kpar) that fill cores_per_node
USE_SWEEP = False

# 2) Manual combos: list of tuples (NCORE, NPAR, KPAR)
MANUAL_COMBOS = [
    # e.g. (16, 6, 2), (12, 8, 2)
]

# 3) Load combos from a CSV with columns NCORE,NPAR,KPAR (no header row check)
USE_TOP_CSV = False
TOP_CSV_PATH = "top_configs.csv"

# Include original (baseline) configuration without NCORE/NPAR/KPAR
ADD_BASELINE = False

# ------------------------------------------------------------------
# Generate combinations based on user options
# ------------------------------------------------------------------
cores_per_node = 192               # --ntasks-per-node in submit.sh
ncore_options = [4, 8, 12, 16]     # band‑parallel sizes to probe
kpar_options = [1, 2, 4, 6]        # k‑point groups to probe

combinations = []
if USE_SWEEP:
    for ncore in ncore_options:
        for kpar in kpar_options:
            if cores_per_node % (ncore * kpar) == 0:     # must divide evenly
                npar = cores_per_node // (ncore * kpar)  # communication groups
                combinations.append((ncore, npar, kpar))

if MANUAL_COMBOS:
    combinations.extend(MANUAL_COMBOS)

if USE_TOP_CSV:
    import csv
    with open(TOP_CSV_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # assume columns named NCORE, NPAR, KPAR
            combinations.append((int(row['NCORE']), int(row['NPAR']), int(row['KPAR'])))

# Deduplicate while preserving order
seen = set()
unique_combos = []
for combo in combinations:
    if combo not in seen:
        seen.add(combo)
        unique_combos.append(combo)
combinations = unique_combos

if ADD_BASELINE:
    combinations.insert(0, (None, None, None))

# Define the input files
input_files = ['POSCAR', 'POTCAR', 'INCAR', 'KPOINTS', 'submit.sh']

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

# Function to update job name inside submit.sh
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
    
    # Modify submit.sh file
    submit_script_path = os.path.join(folder_path, 'submit.sh')
    update_submit_script(submit_script_path, ncore, npar, kpar)

print(f"All directories created and files copied successfully in {base_dir}.")
