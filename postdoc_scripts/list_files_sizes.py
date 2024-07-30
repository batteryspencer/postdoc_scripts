#!/usr/bin/env python

"""
list_files_sizes.py

This script lists specified files in the current directory, prints their sizes in a human-readable format, and mentions if they are not available, all in a table format.
"""

import os
import fnmatch
from tabulate import tabulate

# List of file patterns to check
file_patterns = [
    "cor*", "vaspr*", "CHG*", "OS*", "PCDAT", "IBZKPT", "EIG*", "XDAT*", "*~", 
    "DOSCAR", "PROCAR", "REPORT", "WAVE*", "slurm*"
]

# Function to get the size in a human-readable format
def get_human_readable_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024

# Get the current directory
current_dir = os.getcwd()

# Function to check and print the size of files matching the patterns
def check_files(file_patterns):
    total_size = 0
    file_data = []

    for pattern in file_patterns:
        matched_files = [f for f in os.listdir(current_dir) if os.path.isfile(f) and fnmatch.fnmatch(f, pattern)]
        if matched_files:
            for file in matched_files:
                file_size = os.path.getsize(file)
                human_readable_size = get_human_readable_size(file_size)
                file_data.append([file, file_size, human_readable_size])
                total_size += file_size
        else:
            file_data.append([pattern, 0, "Not available"])

    # Sort by size in descending order
    file_data.sort(key=lambda x: x[1], reverse=True)

    # Add total row
    file_data.append(["Total", total_size, get_human_readable_size(total_size)])

    # Print the table
    print(tabulate(file_data, headers=["File", "Size (Bytes)", "Size (Human Readable)"], tablefmt="grid"))

if __name__ == "__main__":
    from tabulate import tabulate
    check_files(file_patterns)

