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

function setup_environment {
    cd $SLURM_SUBMIT_DIR
    source /etc/profile.d/modules.sh
    module load intel/19.1.3.304 impi/2019.9.304 intel-mkl/2019.9.304 hdf5/1.13.2
    export I_MPI_FABRICS=shm:tcp
    module try-load anaconda
    module load "jgreeley/vasp/5.4.4_beef"
    export VASP_PP_PATH=/depot/jgreeley/apps/vasp/vasppot/
    conda activate ase_vasp
    if [ -f "generate_input_files.py" ]; then
        export ASE_VASP_VDW=/depot/jgreeley/users/pasumarv/lib/
        python generate_input_files.py
    fi
    # Define the source and target paths
    source_path="/depot/jgreeley/users/pasumarv/lib/vdw_kernel.bindat"
    target_path="vdw_kernel.bindat"
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
}

function check_contcar_completeness {
    local contcar_header_lines=$(awk '/Cartesian|Direct/{print NR; exit}' CONTCAR)
    local separator_line=1
    local predictor_block_header_lines=3
    NIONS=$(awk '/Cartesian|Direct/{print sum; exit} {sum=0; for(i=1;i<=NF;i++) sum+=$i}' POSCAR)
    local expected_lines=$((contcar_header_lines + 2 * NIONS + separator_line))
    
    if [ "$IS_MD_CALC" -eq 1 ]; then
        expected_lines=$((expected_lines + separator_line + predictor_block_header_lines + 3 * NIONS))
    fi

    local actual_lines=$(wc -l < CONTCAR)

    if [ "$actual_lines" -eq "$expected_lines" ]; then
        # CONTCAR is complete
        return 0
    else
        # CONTCAR is incomplete
        return 1
    fi
}

# Check if OUTCAR and CONTCAR are complete
function check_convergence_of_last_segment {
    # Change to the last segment directory
    echo "Checking segment $last_seg:"
    cd "seg"$last_seg

    # Check if OUTCAR is complete
    if [ -f "OUTCAR" ]; then
        completed_timesteps=$(grep -c LOOP+ "OUTCAR")
        if [ "$completed_timesteps" -eq "$SEGMENT_SIZE" ]; then
            # OUTCAR is complete
            outcar_complete=0
            echo "OUTCAR is complete."
        else
            # OUTCAR is incomplete
            outcar_complete=1
            echo "OUTCAR is incomplete."
        fi
    else
        # OUTCAR is missing
        outcar_complete=2
        echo "OUTCAR is missing."
    fi

    # Check if CONTCAR is complete
    if [ -f "CONTCAR" ]; then
        if check_contcar_completeness; then
            # CONTCAR is complete
            contcar_complete=0
            echo "CONTCAR is complete."
        else
            # CONTCAR is incomplete
            contcar_complete=1
            echo "CONTCAR is incomplete."
        fi
    else
        # CONTCAR is missing
        contcar_complete=2
        echo "CONTCAR is missing."
    fi

    # convergence_status: 0 if converged, else incomplete
    convergence_status=$((outcar_complete + contcar_complete))

    # Change back to the parent directory
    cd ..
}

function setup_simulation_directory {
    # Start timing
    start_time=$(date +%s)

    if [ $num_segments -eq 1 ] && [ -z "$ASE_VASP_VDW" ]; then
        # Check if the symbolic link already exists
        if [ ! -L "$target_path" ]; then
            # Create the symbolic link
            ln -s "$source_path" "$target_path"
        fi

        # Set NSW to SEGMENT_SIZE
        sed -i 's/^\(\s*NSW\s*=\s*\).*$/\1'"$SEGMENT_SIZE"'/' INCAR

        return
    fi

    # Create segment directory if they don't exist
    seg=$(printf "%0${number_padding}d" $((10#$seg)))
    mkdir -p "seg"$seg

    # Modify the INCAR file for each segment
    if [ $((10#$seg)) -eq $num_segments ] && [ $((TOTAL_NSW % SEGMENT_SIZE)) -ne 0 ]; then
        # For the last segment, if there's a residual, set NSW to the residual
        sed -i 's/^\(\s*NSW\s*=\s*\).*$/\1'"$((TOTAL_NSW % SEGMENT_SIZE))"'/' INCAR
    else
        # For all other segments, set NSW to SEGMENT_SIZE
        sed -i 's/^\(\s*NSW\s*=\s*\).*$/\1'"$SEGMENT_SIZE"'/' INCAR
    fi

    # Change to the segment directory
    cd "seg"$seg

    # Check if the symbolic link already exists
    if [ ! -L "$target_path" ]; then
        # Create the symbolic link
        ln -s "$source_path" "$target_path"
    fi

    # Copy files to the segment directory
    if [ $seg -eq 1 ]; then
        cp ../{INCAR,KPOINTS,POSCAR,POTCAR} . && [ $IS_MD_CALC -eq 1 ] && cp ../ICONST .
    else
        cp ../seg$(printf "%0${number_padding}d" $((10#$seg - 1)))/CONTCAR POSCAR
        cp ../{INCAR,KPOINTS,POTCAR} . && [ $IS_MD_CALC -eq 1 ] && cp ../ICONST .
    fi
}

function check_segment_completion {
    # Check if directories starting with "seg" exist
    seg_dir_exists=false
    for dir in seg*/ ; do
        [ -d "$dir" ] && seg_dir_exists=true && break
    done

    if $seg_dir_exists; then
        echo
        echo "Directories starting with 'seg' exist."

        # Find the last segment number
        last_seg=$(ls -d seg* | sort -n | tail -n 1 | sed 's/seg//')
        echo "Last segment number: $last_seg"
        for last_seg in $(seq $last_seg -1 1)
        do
            last_seg=$(printf "%0${number_padding}d" $((10#$last_seg)))
            
            check_convergence_of_last_segment

            if [ "$convergence_status" -eq 0 ]; then
                # Directories starting with 'seg' exist and are complete
                echo "Segment $last_seg is complete."
                echo
                break
            else
                # Directories starting with 'seg' exist but are incomplete
                echo "Segment $last_seg is incomplete."
                rm -rf "seg"$last_seg
                echo "Segment $last_seg has been removed."
                echo
                if [ "$last_seg" -eq 1 ]; then
                    last_seg=$((10#$last_seg - 1))
                fi
            fi
        done
    else
        # Directories starting with 'seg' do not exist
        last_seg=0
    fi
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
    if [ $num_segments -eq 1 ]; then
        echo -e "\nJob execution time: $hours hours, $minutes minutes, $seconds seconds ($execution_time seconds)"
    else
        echo -e "\nJob execution time: $hours hours, $minutes minutes, $seconds seconds ($execution_time seconds)" >> job.out
    fi
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

    # Calculate the total number of segments, incrementing by 1 if there's a remainder after division
    num_segments=$((TOTAL_NSW / SEGMENT_SIZE + (TOTAL_NSW % SEGMENT_SIZE > 0 ? 1 : 0)))

    check_segment_completion

    start_segment_number=$((10#$last_seg + 1))
    for seg in $(seq $start_segment_number $num_segments)
    do        
        setup_simulation_directory

        # Run the VASP job:
        if [ $num_segments -eq 1 ]; then
            srun -n $PROC_NUM --mpi=pmi2 $EXECUTABLE
        else
            srun -n $PROC_NUM --mpi=pmi2 $EXECUTABLE > job.out 2> job.err
        fi

        post_process
    done

    # Remove duplicate files
    # for file in $duplicatefiles; do
    #     rm -f $file
    # done

    # Check if compute_bader_charges is set to 1
    if [ "$compute_bader_charges" -eq 1 ]; then
        cd seg$seg
        echo -e "\nEvaluating Bader charges:"
        bader CHGCAR
        cd ..
    fi

}

####################################################
#                 USER VARIABLES                   #
####################################################

# Define the total NSW and the segment size
TOTAL_NSW=6000
SEGMENT_SIZE=1000

# define a list of files
# duplicatefiles="POSCAR POTCAR INCAR ICONST KPOINTS"
removefiles="WAVECAR"

# Other definitions
compute_bader_charges=0  # Set this to 0 if you don't want to run "bader CHGCAR"
IS_MD_CALC=1  # Set this to 1 for MD calculations
number_padding=2  # number padding for segment directories
EXECUTABLE=vasp_std  # Define the VASP executable

# --- Execute Main Logic ---
main
