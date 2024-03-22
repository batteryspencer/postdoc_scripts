#!/bin/bash

# --- SBATCH Options ---
#SBATCH --job-name=new_submit_script
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=128  # max: 128
##SBATCH --tasks-per-node=128  # max: 128
##SBATCH --cpus-per-task=1
##SBATCH --mem=500G  # max: 500 GB
#SBATCH -A standby
#SBATCH --time=1:00:00  # max: (04:00:00 for standby, 14-00:00:00 for jgreeley)
#SBATCH --mail-user=pasumarv@purdue.edu
#SBATCH --mail-type=END

# Start timing
start_time=$(date +%s)

# Define the source and target paths
source_path="/depot/jgreeley/users/pasumarv/lib/vdw_kernel.bindat"
target_path="vdw_kernel.bindat"

# Check if the symbolic link already exists
if [ ! -L "$target_path" ]; then
    # Create the symbolic link
    ln -s "$source_path" "$target_path"
fi

function setup_environment {
    cd $SLURM_SUBMIT_DIR
    source /etc/profile.d/modules.sh
    module load intel/19.1.3.304 impi/2019.9.304 intel-mkl/2019.9.304 hdf5/1.13.2
    export I_MPI_FABRICS=shm:tcp
    module try-load anaconda
    module load "jgreeley/vasp/5.4.4_beef"
    export VASP_PP_PATH=/depot/jgreeley/apps/vasp/vasppot/
    conda activate ase_vasp
    # python generate_input_files.py
    log_job_details
}

function log_job_details {
    export mailbox=pasumarv@purdue.edu
    NUM_CORE=$SLURM_CPUS_ON_NODE
    NUM_NODE=$SLURM_JOB_NUM_NODES
    let PROC_NUM=$NUM_CORE*$NUM_NODE
    echo "Total CPUs per node: $NUM_CORE"
    echo "Total nodes requested: $NUM_NODE"
    echo "Total CPUs for this jobs: nodes x ppn: $PROC_NUM"
    echo $SLURM_JOB_NODELIST > nodefile.$SLURM_JOB_ID
    NIONS=$(sed -n '7p' POSCAR | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum}')
}

function main {
    setup_environment

    # Run the VASP job
    local EXECUTABLE=vasp_std
    srun -n $PROC_NUM --mpi=pmi2 $EXECUTABLE

}

# --- Execute Main Logic ---
main

# End timing
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Calculate hours, minutes, and seconds
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60 ))
seconds=$((execution_time % 60))

# Print the execution time
echo 
echo "Job execution time: $hours hours, $minutes minutes, $seconds seconds ($execution_time seconds)"

