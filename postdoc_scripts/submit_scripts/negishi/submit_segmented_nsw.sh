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

# Define the source and target paths
source_path="/depot/jgreeley/users/pasumarv/lib/vdw_kernel.bindat"
target_path="vdw_kernel.bindat"

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

function setup_simulation_directory {
    # Create segment directory if they don't exist
    seg=$(printf "%0${number_padding}d" $seg)
    mkdir -p "seg"$seg

    # Change to the segment directory
    cd "seg"$seg

    # Check if the symbolic link already exists
    if [ ! -L "$target_path" ]; then
        # Create the symbolic link
        ln -s "$source_path" "$target_path"
    fi

    # Copy files to the segment directory
    if [ $seg -eq 1 ]; then
        cp ../{INCAR,ICONST,KPOINTS,POSCAR,POTCAR} .
    else
        cp ../seg$(printf "%0${number_padding}d" $((seg - 1)))/CONTCAR POSCAR
        cp ../{INCAR,ICONST,KPOINTS,POTCAR} .
    fi

    # Start timing
    start_time=$(date +%s)
}

function log_execution_time {
    # End timing
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))

    # Calculate hours, minutes, and seconds
    hours=$((execution_time / 3600))
    minutes=$(( (execution_time % 3600) / 60 ))
    seconds=$((execution_time % 60))

    # Print the execution time
    echo -e "\nJob execution time: $hours hours, $minutes minutes, $seconds seconds ($execution_time seconds)" >> job.out
}

function post_process {

    # Log the execution time
    log_execution_time

    # Remove the files from the current directory
    for file in $removefiles; do
        rm -f $file
    done

    # Change back to the parent directory
    cd ..
}

function main {
    setup_environment

    # Define the total NSW and the segment size
    total_nsw=10
    segment_size=5

    # Calculate the number of segments
    num_segments=$((total_nsw / segment_size))

    # Modify INCAR file to set NSW to segment size
    sed -i 's/^\(\s*NSW\s*=\s*\).*$/\1'"$segment_size"'/' INCAR

    # Define the VASP executable
    local EXECUTABLE=vasp_std

    for seg in $(seq 1 $num_segments)
    do
        setup_simulation_directory

        # Run the VASP job
        srun -n $PROC_NUM --mpi=pmi2 $EXECUTABLE > job.out 2> job.err

        post_process
    done

    # Remove duplicate files
    for file in $duplicatefiles; do
        rm -f $file
    done

    # Check if compute_bader_charges is set to 1
    if [ "$compute_bader_charges" -eq 1 ]; then
        cd seg$seg
        echo
        echo "Evaluating Bader charges:"
        bader CHGCAR
        cd ..
    fi

}

####################################################
#                 USER VARIABLES                   #
####################################################

number_padding=2

# define a list of files
duplicatefiles="POSCAR POTCAR INCAR ICONST KPOINTS"
removefiles="WAVECAR"

# Set the compute_bader_charges parameter (0 or 1)
compute_bader_charges=0  # Set this to 0 if you don't want to run "bader CHGCAR"

# --- Execute Main Logic ---
main
