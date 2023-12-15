#!/bin/bash

# --- SBATCH Options ---
#SBATCH --job-name=Pt111_PMF_1PropylH_C-H_1.11
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128  # max: 128
##SBATCH --tasks-per-node=128  # max: 128
##SBATCH --cpus-per-task=1
##SBATCH --mem=500G  # max: 500 GB
#SBATCH -A standby
#SBATCH --time=3:50:00  # max: (04:00:00 for standby, 14-00:00:00 for jgreeley)
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
}

# --- Email Notifications ---
send_mail() {
    local action=$1
    $JAPPSBASE/templates/send_job_mail_slurm.py "${action}"
}

# --- Calculation Functions ---
# Checks the convergence of a process.
# Arguments:
#   None
# Returns:
#   0 if converged
#   1 if non-empty CONTCAR is found
#   2 if empty CONTCAR is found
#   3 if OUTCAR is missing
function check_convergence() {
    COMPLETED_TIMESTEPS=0
    local sub=${SUB:-1}
    if [ $sub -gt 1 ]; then
        for outcar in ${prefix}*/OUTCAR; do
            if [ -f "$outcar" ]; then
                loops=$(grep -c LOOP+ "$outcar")
                COMPLETED_TIMESTEPS=$((COMPLETED_TIMESTEPS + loops))
            fi
        done
    fi

    if [ -s "OUTCAR" ]; then
        echo 
        COMPLETED_TIMESTEPS=$((COMPLETED_TIMESTEPS + $(grep -c LOOP+ "OUTCAR")))
        echo "COMPLETED_TIMESTEPS: $COMPLETED_TIMESTEPS"
        echo "TOTAL_TIMESTEPS: $TOTAL_TIMESTEPS"

        REMAINING_TIMESTEPS=$((TOTAL_TIMESTEPS - COMPLETED_TIMESTEPS))
        echo "REMAINING_TIMESTEPS: $REMAINING_TIMESTEPS"

        echo

        if [ $COMPLETED_TIMESTEPS -ge $TOTAL_TIMESTEPS ]; then
            return 0
        elif [ -s "CONTCAR" ]; then
            return 1
        else
            return 2
        fi
    else
        echo
        echo "TOTAL_TIMESTEPS: $TOTAL_TIMESTEPS"
        echo "COMPLETED_TIMESTEPS: $COMPLETED_TIMESTEPS"
        echo
        return 3
    fi
}

function resubmit() {
    local sub=${SUB:-1}
    sub=$(expr $sub + 1)
    if [ $sub -le $resubmissions ]; then
        [ "${mail_resubmit}" == "TRUE" ] && send_mail "RESUB:$(expr $sub - 1)"
        CANCEL=$(sbatch --dependency="afterany:${SLURM_JOB_ID}" --export=ALL,SUB=$sub,OLD_SLURM_JOB_ID=$SLURM_JOB_ID,TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS $0)
        CANCEL=$(echo $CANCEL | cut -d ' ' -f 4)
    fi
}

function cancel_resubmit() {
    [ -n "${CANCEL}" ] && scancel "${CANCEL}"
}

function completed() {
    cancel_resubmit
    [ "${mail_converge}" == "TRUE" ] && send_mail "CONVERGED"

    # Check if compute_bader_charges is set to 1
    if [ "$compute_bader_charges" -eq 1 ]; then
        echo
        echo "Evaluating Bader charges:"
        bader CHGCAR
    fi
    
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

    sleep 3
    exit
}

function backup_calculation() {
    $JAPPSBASE/templates/backup-files.py --silent --suffix "${suffix}" --prefix "${prefix}" --padding "${padding}" "${backupfiles}"
}

function setup_restart() {
    cp CONTCAR POSCAR

    # Replace NSW value in INCAR
    sed -i "s/NSW *= *[0-9]*/NSW = $REMAINING_TIMESTEPS/" INCAR
}

function clear_calculation() {
    rm $removefiles
}

function main(){
    setup_environment
    check_convergence
    local convergence=$?

    case $convergence in
        0)
            completed
            ;;
        1)
            backup_calculation
            setup_restart
            clear_calculation
            ;;
        2)
            [ "${mail_fail}" == "TRUE" ] && send_mail "CONTCAR EMPTY"
            ;;
        3)
            [ "${mail_fail}" == "TRUE" ] && send_mail "MISSING STARTING FILES"
            ;;
    esac

    resubmit
    # Run the VASP job
    local EXECUTABLE=vasp_std
    touch WAVECAR CHGCAR CHG
    srun -n $PROC_NUM --mpi=pmi2 $EXECUTABLE
    echo $(date) $SLURM_JOB_ID $SLURM_SUBMIT_HOST $SLURM_JOB_NAME $SLURM_SUBMIT_DIR >> ~/jobs.log

    check_convergence
    convergence=$?
    [ $convergence -eq 0 ] && completed

}

####################################################
#                 USER VARIABLES                   #
####################################################

# Email Notifications (TRUE/FALSE)
mail_start="FALSE"     # At job start
mail_converge="TRUE"   # On job convergence
mail_resubmit="FALSE"  # On job resubmission
mail_fail="TRUE"       # On job failure

# Backup Directory Filenames
prefix="RUN_"          # Prefix
suffix=""              # Suffix
padding=2              # Number padding

# Job Environment Settings
resubmissions=20        # Max resubmission count
OLD_SLURM_JOB_ID=${OLD_SLURM_JOB_ID:-$SLURM_JOB_ID}
backupfiles="CONTCAR INCAR KPOINTS ICONST REPORT OSZICAR OUTCAR POSCAR XDATCAR *$OLD_SLURM_JOB_ID.o* *$OLD_SLURM_JOB_ID.e* vasprun.xml WAVECAR CHGCAR nodefile.$OLD_SLURM_JOB_ID"
removefiles="OSZICAR DOSCAR EIGENVAL IBZKPT PCDAT PROCAR FORCECAR nodefile.$OLD_SLURM_JOB_ID *$OLD_SLURM_JOB_ID.o* *$OLD_SLURM_JOB_ID.e*"

# Set the compute_bader_charges parameter (0 or 1)
compute_bader_charges=0  # Set this to 0 if you don't want to run "bader CHGCAR"

# Simulation Parameters
if [ -z "$SUB" ]; then
    TOTAL_TIMESTEPS=$(grep 'NSW' INCAR | awk -F ' *= *' '{print $2}')
fi

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

