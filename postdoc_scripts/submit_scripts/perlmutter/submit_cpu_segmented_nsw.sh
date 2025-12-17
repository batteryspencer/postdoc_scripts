#!/bin/bash

# --- SBATCH Options ---
#SBATCH --job-name=Pt111_PMF_2PropylH_Dissociation_C-H_2.49
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=64  # max: 128
#SBATCH --account=m1399
#SBATCH --qos=regular  # regular
#SBATCH --time-min=4:00:00
#SBATCH --time=24:00:00  # max: 48:00:00 for regular
#SBATCH --mail-user=pasumarv@purdue.edu
#SBATCH --mail-type=END
#SBATCH --constraint=cpu

# This SLURM batch script manages VASP job submissions with support for:
# - Multi-segment molecular dynamics (MD) calculations
# - Non-MD calculations like geometry relaxation
#
# --- Checkpointing and Restarts ---
# - Supports restarting from checkpoints in case of job failure or timeout.
#
# --- Segment Handling ---
# - Divides the simulation into segments to handle long MD calculations or geometry relaxations.
# - Sets up and processes each segment, ensuring completeness of output files.
#
# --- Convergence Checks ---
# - Validates the completeness of CONTCAR and OUTCAR files to determine if each segment has converged.
#
# --- Self-Termination ---
# - Automatically terminates the job if there is insufficient remaining time to complete the current segment.
#
# --- Post-Processing ---
# - Supports Bader charge calculations if specified.

function setup_environment {
    cd $SLURM_SUBMIT_DIR
    module load vasp/6.4.1-cpu
    export OMP_NUM_THREADS=1
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread
    export VASP_PP_PATH=/global/homes/p/pasumart/usr/lib/apps/vasp/vasppot/
    if [ -f "generate_input_files.py" ] && [ ! -f "POSCAR" ] && [ ! -f "POTCAR" ] && [ ! -f "KPOINTS" ] && [ ! -f "INCAR" ]; then
        export ASE_VASP_VDW=/global/homes/p/pasumart/usr/lib/
        echo "Generating input files..."
        python generate_input_files.py
    fi
    # Define the source and target paths
    source_path="/global/homes/p/pasumart/usr/lib/vdw_kernel.bindat"
    target_path="vdw_kernel.bindat"
    log_job_details
}

function log_job_details {
    NUM_CORE=$SLURM_CPUS_ON_NODE
    echo "Total CPUs per node: $NUM_CORE"
    echo "Total nodes requested: $SLURM_JOB_NUM_NODES"

    # Total number of tasks (processes) across all nodes
    TOTAL_TASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
    echo "Total tasks for this job (tasks per node x number of nodes): $SLURM_NTASKS_PER_NODE x $SLURM_JOB_NUM_NODES = $TOTAL_TASKS"

    # CPUs per task, assuming SLURM_CPUS_ON_NODE is the total CPUs on one node
    CPUS_PER_TASK=$(($SLURM_CPUS_ON_NODE / $SLURM_NTASKS_PER_NODE))
    echo "CPUs per task (total CPUs on node / tasks per node): $SLURM_CPUS_ON_NODE / $SLURM_NTASKS_PER_NODE = $CPUS_PER_TASK"

    echo $SLURM_JOB_NODELIST > nodefile.$SLURM_JOB_ID
}

# Log bad nodes to a central file for tracking
function log_bad_nodes {
    local reason="$1"
    local steps_completed="$2"
    local elapsed_minutes="$3"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local nodes=$(cat $SLURM_SUBMIT_DIR/nodefile.$SLURM_JOB_ID 2>/dev/null || echo "unknown")

    # Ensure bad node log directory exists
    mkdir -p "$(dirname "$BAD_NODE_LOG")"

    # Log format: timestamp | job_id | nodes | directory | reason | steps | elapsed_min
    echo "$timestamp | $SLURM_JOB_ID | $nodes | $PWD | $reason | steps=$steps_completed | elapsed=${elapsed_minutes}min" >> "$BAD_NODE_LOG"
    echo "Bad node logged: $nodes (reason: $reason, steps: $steps_completed in ${elapsed_minutes}min)"
}

# Background progress monitor - checks if VASP is making expected progress
function monitor_progress {
    local outcar_path="$1"
    local monitor_start_time=$(date +%s)
    local prev_steps=0
    local stall_count=0

    # Wait for grace period before first check (VASP needs time to initialize)
    sleep ${PROGRESS_GRACE_PERIOD_MIN}m

    while true; do
        # Check if OUTCAR exists and count steps
        if [ -f "$outcar_path" ]; then
            local current_steps=$(grep -c LOOP+ "$outcar_path" 2>/dev/null || echo 0)
            local current_time=$(date +%s)
            local elapsed_seconds=$((current_time - monitor_start_time))
            local elapsed_minutes=$((elapsed_seconds / 60))

            # Calculate expected steps based on elapsed time
            local expected_steps=$(( (elapsed_minutes * EXPECTED_STEPS_PER_MINUTE) ))
            local min_acceptable_steps=$(( expected_steps * MIN_PROGRESS_PERCENT / 100 ))

            # Calculate step delta since last check
            local step_delta=$((current_steps - prev_steps))
            local expected_delta=$((PROGRESS_CHECK_INTERVAL_MIN * EXPECTED_STEPS_PER_MINUTE * MIN_PROGRESS_PERCENT / 100))

            echo "[Monitor] Elapsed: ${elapsed_minutes}min, Steps: $current_steps (+$step_delta), Min expected: $min_acceptable_steps, Stall count: $stall_count/$MAX_STALL_CHECKS"

            # Check 1: Overall progress too slow from start
            if [ "$current_steps" -lt "$min_acceptable_steps" ] && [ "$elapsed_minutes" -ge "$PROGRESS_GRACE_PERIOD_MIN" ]; then
                echo "[Monitor] SLOW PROGRESS DETECTED! Steps: $current_steps, Expected at least: $min_acceptable_steps"
                log_bad_nodes "slow_progress" "$current_steps" "$elapsed_minutes"
                echo "[Monitor] Terminating job to avoid wasting compute time..."
                kill $VASP_PID 2>/dev/null
                scancel $SLURM_JOB_ID
                exit 1
            fi

            # Check 2: Job stalled (no/very few new steps since last check)
            if [ "$step_delta" -lt "$expected_delta" ] && [ "$prev_steps" -gt 0 ]; then
                stall_count=$((stall_count + 1))
                echo "[Monitor] WARNING: Only $step_delta new steps (expected ~$expected_delta). Stall count: $stall_count/$MAX_STALL_CHECKS"

                if [ "$stall_count" -ge "$MAX_STALL_CHECKS" ]; then
                    echo "[Monitor] JOB STALLED! No meaningful progress for $((stall_count * PROGRESS_CHECK_INTERVAL_MIN)) minutes"
                    log_bad_nodes "stalled" "$current_steps" "$elapsed_minutes"
                    echo "[Monitor] Terminating stalled job..."
                    kill $VASP_PID 2>/dev/null
                    scancel $SLURM_JOB_ID
                    exit 1
                fi
            else
                # Reset stall count if progress resumed
                stall_count=0
            fi

            prev_steps=$current_steps
        fi

        # Check every PROGRESS_CHECK_INTERVAL_MIN minutes
        sleep ${PROGRESS_CHECK_INTERVAL_MIN}m

        # Safety: stop monitoring if we've been running too long (job should end naturally)
        local total_elapsed=$(($(date +%s) - monitor_start_time))
        if [ "$total_elapsed" -gt "$((EXPECTED_SEGMENT_RUNTIME_MIN * 60 * 2))" ]; then
            echo "[Monitor] Exceeded 2x expected runtime, stopping monitor"
            break
        fi
    done
}

# Start background progress monitor
function start_progress_monitor {
    local outcar_path="$1"
    if [ "$ENABLE_PROGRESS_MONITOR" -eq 1 ]; then
        echo "Starting progress monitor (grace period: ${PROGRESS_GRACE_PERIOD_MIN}min, check interval: ${PROGRESS_CHECK_INTERVAL_MIN}min)"
        monitor_progress "$outcar_path" &
        MONITOR_PID=$!
    fi
}

# Stop background progress monitor
function stop_progress_monitor {
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
        unset MONITOR_PID
    fi
}

function check_contcar_completeness {
    local contcar_header_lines=$(awk '/Cartesian|Direct/{print NR; exit}' CONTCAR)
    local separator_line=1
    local predictor_block_header_lines=3
    NIONS=$(awk 'NR==7 || (NR==6 && $1+0==$1){for(i=1;i<=NF;i++) sum+=$i; print sum; exit}' POSCAR)
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

    # Determine expected steps for this segment
    local expected_steps=$SEGMENT_SIZE
    if [ $((10#$last_seg)) -eq $num_segments ] && [ $((TOTAL_NSW % SEGMENT_SIZE)) -ne 0 ]; then
        expected_steps=$((TOTAL_NSW % SEGMENT_SIZE))
    fi

    # Check if OUTCAR is complete
    if [ -f "OUTCAR" ]; then
        completed_timesteps=$(grep -c LOOP+ "OUTCAR")
        if [ "$completed_timesteps" -eq "$expected_steps" ]; then
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

    # segment_convergence_status: 0 if converged, else incomplete
    segment_convergence_status=$((outcar_complete + contcar_complete))

    # Change back to the parent directory
    cd ..
}

function setup_simulation_directory {
    # Start timing
    start_time=$(date +%s)

    if [ $num_segments -eq 1 ]; then
        if [ -z "$ASE_VASP_VDW" ]; then
            # Check if the symbolic link already exists
            if [ ! -L "$target_path" ]; then
                # Create the symbolic link
                ln -s "$source_path" "$target_path"
            fi
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
        # For the last segment, if there's a residual, set SEGMENT_SIZE to the residual
        SEGMENT_SIZE=$((TOTAL_NSW % SEGMENT_SIZE))
    fi

    # For all other segments, set NSW to SEGMENT_SIZE
    sed -i 's/^\(\s*NSW\s*=\s*\).*$/\1'"$SEGMENT_SIZE"'/' INCAR

    # Change to the segment directory
    cd "seg"$seg

    # Check if the symbolic link already exists
    if [ ! -L "$target_path" ]; then
        # Create the symbolic link
        ln -s "$source_path" "$target_path"
    fi

    # Copy files to the segment directory
    if [ $seg -eq 1 ]; then
        cp ../{INCAR,KPOINTS,POSCAR,POTCAR} .
        [ $IS_MD_CALC -eq 1 ] && [ -f ../ICONST ] && cp ../ICONST .
    else
        cp ../seg$(printf "%0${number_padding}d" $((10#$seg - 1)))/CONTCAR POSCAR
        cp ../{INCAR,KPOINTS,POTCAR} .
        [ $IS_MD_CALC -eq 1 ] && [ -f ../ICONST ] && cp ../ICONST .
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

            if [ "$segment_convergence_status" -eq 0 ]; then
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

function restart_from_checkpoint {
    CURRENT_RESTART_INDEX=${CURRENT_RESTART_INDEX:-0}
    if (( CURRENT_RESTART_INDEX < MAX_RESTARTS )); then
        CURRENT_RESTART_INDEX=$((CURRENT_RESTART_INDEX + 1))
        dependency_job_id=$(sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,CURRENT_RESTART_INDEX=$CURRENT_RESTART_INDEX,SLURM_OLD_JOB_ID=$SLURM_JOB_ID $0 | awk '{print $NF}')
        echo -e "\nDependency job ID: $dependency_job_id"
    fi

    if [ -n "$SLURM_OLD_JOB_ID" ]; then
        if grep -q "ZBRENT: fatal error: bracketing interval incorrect" job_${SLURM_OLD_JOB_ID}.out; then
            echo -e "\nCopying CONTCAR to POSCAR"
            cp CONTCAR POSCAR
        fi
    fi
}

function log_runtime {
    # End timing
    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    # Calculate hours, minutes, and seconds
    hours=$((runtime / 3600))
    minutes=$(( (runtime % 3600) / 60 ))
    seconds=$((runtime % 60))

    # Print the execution time
    if [ $num_segments -eq 1 ]; then
        echo -e "\nJob execution time: $hours hours, $minutes minutes, $seconds seconds ($runtime seconds)"
    else
        echo -e "\nJob execution time: $hours hours, $minutes minutes, $seconds seconds ($runtime seconds)" >> job.out
    fi
}

function post_process {

    # Log the execution time
    log_runtime

    # Remove the files from the current directory
    for file in $removefiles; do
        rm -f $file
    done

    # Check if the calculation is complete
    if [ $IS_MD_CALC -eq 1 ] && [ $seg -eq $num_segments ]; then
        completed_timesteps=$(grep -c LOOP+ "OUTCAR")
        # Reset outcar_complete to ensure we don't use stale value from check_segment_completion
        outcar_complete=1
        if [ "$completed_timesteps" -eq "$SEGMENT_SIZE" ]; then
            # OUTCAR is complete
            outcar_complete=0
        fi

        if check_contcar_completeness && [ "$outcar_complete" -eq 0 ]; then
            echo -e "\nJob converged"
            job_convergence_status=0
        fi
    else
        if grep -q "reached required accuracy" OUTCAR; then
            echo -e "\nJob converged"
            job_convergence_status=0
        fi
    fi

    if [ $num_segments -ne 1 ]; then
        # Change back to the parent directory
        cd ..
    fi
}

# Function to convert time to seconds
function time_to_seconds {
    local time_str="$1"
    local total_seconds=0

    # Check if the time string contains a day part
    if [[ "$time_str" == *-* ]]; then
        IFS='-' read -r days time_str <<< "$time_str"
        total_seconds=$((days * 86400))
    fi

    # Split the remaining time string and add hours, minutes, and seconds
    IFS=: read -r hours minutes seconds <<< "$time_str"
    if [[ -z "$seconds" ]]; then
        if [[ -z "$minutes" ]]; then
            seconds=$hours
            hours=0
        else
            seconds=$minutes
            minutes=$hours
            hours=0
        fi
    fi

    total_seconds=$((total_seconds + hours * 3600 + minutes * 60 + seconds))
    echo $total_seconds
}

# Function to get remaining time
function get_remaining_time {
    squeue -j $SLURM_JOB_ID -h -o "%L"
}

# Check remaining time
function check_time {
    prev_runtime=$1
    remaining_time=$(get_remaining_time)
    remaining_seconds=$(time_to_seconds "$remaining_time")
    required_time=$(($prev_runtime + $time_limit_min * 60))

    if (( remaining_seconds < required_time )); then
        echo "Not enough time to safely complete the segment. Terminating job."
        scancel $SLURM_JOB_ID
    fi
}

function main {
    setup_environment

    restart_from_checkpoint

    # Calculate the total number of segments. If SEGMENT_SIZE is 0, set num_segments to 1. Otherwise, increment by 1 if there's a remainder after division.
    num_segments=$((SEGMENT_SIZE == 0 ? 1 : (TOTAL_NSW / SEGMENT_SIZE + (TOTAL_NSW % SEGMENT_SIZE > 0 ? 1 : 0))))

    check_segment_completion

    start_segment_number=$((10#$last_seg + 1))
    prev_runtime=0  # Initialize the previous runtime to 0
    for seg in $(seq $start_segment_number $num_segments)
    do
        # Check if enough time is available
        check_time $prev_runtime

        setup_simulation_directory

        # Determine OUTCAR path for monitoring
        if [ $num_segments -eq 1 ]; then
            outcar_monitor_path="$SLURM_SUBMIT_DIR/OUTCAR"
        else
            outcar_monitor_path="$PWD/OUTCAR"
        fi

        # Start progress monitor in background
        start_progress_monitor "$outcar_monitor_path"

        # Run the VASP job:
        if [ $num_segments -eq 1 ]; then
            srun -n $TOTAL_TASKS -c $CPUS_PER_TASK --cpu-bind=cores $EXECUTABLE &
            VASP_PID=$!
            wait $VASP_PID
        else
            srun -n $TOTAL_TASKS -c $CPUS_PER_TASK --cpu-bind=cores $EXECUTABLE > job.out 2> job.err &
            VASP_PID=$!
            wait $VASP_PID
        fi

        # Stop progress monitor
        stop_progress_monitor

        post_process

        # Check if the calculation has converged
        if [ "${job_convergence_status:-1}" -eq 0 ]; then 
            # Cancel the dependency job
            scancel $dependency_job_id
            break
        fi

        # Update previous runtime and time limit
        if (( runtime > prev_runtime )); then
            prev_runtime=$runtime
        fi

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

# Define the total NSW
TOTAL_NSW=10000
# Set SEGMENT_SIZE to 0 for single-point calculations.
SEGMENT_SIZE=1000

# define a list of files
# duplicatefiles="POSCAR POTCAR INCAR ICONST KPOINTS"
removefiles="WAVECAR"

# Other definitions
compute_bader_charges=0  # Set this to 0 if you don't want to run "bader CHGCAR"
IS_MD_CALC=1  # Set this to 1 for MD calculations
number_padding=2  # number padding for segment directories
EXECUTABLE=vasp_std  # Define the VASP executable
MAX_RESTARTS=20  # Set the maximum number of restarts here
time_limit_min=10  # Set minimum additional time in minutes

####################################################
#         PROGRESS MONITORING SETTINGS             #
####################################################
# Enable/disable progress monitoring (1=enabled, 0=disabled)
ENABLE_PROGRESS_MONITOR=1

# Central log file for tracking bad nodes across all jobs
BAD_NODE_LOG="/pscratch/sd/p/pasumart/bad_nodes.log"

# Grace period before first progress check (minutes)
# VASP needs time to initialize - don't check too early
PROGRESS_GRACE_PERIOD_MIN=20

# How often to check progress after grace period (minutes)
PROGRESS_CHECK_INTERVAL_MIN=10

# Expected segment runtime in minutes (1000 steps in ~150 min = 2.5 hours)
EXPECTED_SEGMENT_RUNTIME_MIN=150

# Expected steps per minute - SET THIS BASED ON YOUR BENCHMARK
# Example: 1000 steps / 150 min ≈ 6.7 steps/min (~9 s/step)
EXPECTED_STEPS_PER_MINUTE=7

# Minimum acceptable progress as percentage of expected
# 25% = tolerate up to 4x slower than expected before flagging
MIN_PROGRESS_PERCENT=25

# Number of consecutive stall checks before terminating
# Detects jobs that start well then freeze mid-run
# With 10 min interval, 3 checks = kill after 30 min of no progress
MAX_STALL_CHECKS=3

# --- Execute Main Logic ---
main
