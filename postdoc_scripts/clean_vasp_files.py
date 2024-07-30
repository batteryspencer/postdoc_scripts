#!/usr/bin/env python

"""
clean_vasp_files.py

This script removes specified VASP files from the current directory and from all subdirectories,
excluding XDATCAR, slurm*, and *~. If the maximum directory depth exceeds 10, it prompts the user for confirmation.
"""

import os
import fnmatch
from alive_progress import alive_bar

# List of file patterns to check and clean
file_patterns = [
    "cor*", "vaspr*", "CHG*", "OS*", "PCDAT", "IBZKPT", "EIG*", "DOSCAR", "PROCAR", "REPORT", "WAVE*", "vaspout.h5"
]

# Function to get the relative path
def get_relative_path(base, target):
    return os.path.relpath(target, base)

# Function to delete files matching the patterns in a given directory and track storage cleared
def clean_files_in_directory(directory, base_dir, file_patterns, total_cleared):
    for pattern in file_patterns:
        matched_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and fnmatch.fnmatch(f, pattern)]
        for file in matched_files:
            file_path = os.path.join(directory, file)
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                total_cleared.append(file_size)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Function to calculate the maximum depth of a directory tree
def calculate_max_depth(directory):
    max_depth = 0
    for root, dirs, files in os.walk(directory):
        depth = root[len(directory):].count(os.sep)
        if depth > max_depth:
            max_depth = depth
    return max_depth

# Function to clean files in all subdirectories
def clean_files(file_patterns):
    current_dir = os.getcwd()
    total_cleared = []

    # Calculate the maximum depth of the directory tree
    max_depth = calculate_max_depth(current_dir)
    if max_depth > 10:
        response = input(f"The maximum directory depth is {max_depth}. Do you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup aborted.")
            return
    
    # Calculate total number of directories to process
    total_dirs = sum([len(dirs) for _, dirs, _ in os.walk(current_dir)]) + 1

    # Initialize alive-progress bar
    with alive_bar(total_dirs, dual_line=True, title="Cleaning directories", bar="blocks") as bar:

        # Clean files in the current directory and all subdirectories
        for root, dirs, files in os.walk(current_dir):
            clean_files_in_directory(root, current_dir, file_patterns, total_cleared)
            bar.text(f"Processing: {root}")
            bar()

    total_cleared_bytes = sum(total_cleared)
    total_cleared_human = get_human_readable_size(total_cleared_bytes)
    print(f"\nTotal storage cleared: {total_cleared_human}")

# Function to get the size in a human-readable format
def get_human_readable_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024

if __name__ == "__main__":
    clean_files(file_patterns)
    print("Cleanup completed.")

