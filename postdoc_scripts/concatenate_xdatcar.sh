#!/bin/bash

# Directory pattern to match
DIR_PATTERN="seg?? seg???"

# Frame limit variable
FRAME_LIMIT=10000

# Temporary file for concatenation
TEMP_FILE="temp_xdatcar"

# Check if temp file exists and remove it
[ -f "$TEMP_FILE" ] && rm "$TEMP_FILE"

# Find directories matching the pattern and sort them numerically
dirs=$(ls -d $DIR_PATTERN 2>/dev/null | sort -V)

# Loop through directories and concatenate XDATCAR files
first_file=true
for dir in $dirs; do
    # Ensure the directory exists to avoid errors with missing directories
    if [ -d "$dir" ]; then
        if $first_file; then
            # Copy the header and the rest of the file for the first segment
            cat "${dir}/XDATCAR" >> "${TEMP_FILE}"
            first_file=false
        else
            # Skip the first 7 lines (header) for subsequent segments
            tail -n +8 "${dir}/XDATCAR" >> "${TEMP_FILE}"
        fi
    fi
done

# Extract the number of frames needed
awk -v limit=$FRAME_LIMIT '/^Direct configuration/ {n++} n<=limit {print}' "$TEMP_FILE" > "XDATCAR_combined"

# Clean up
rm "$TEMP_FILE"

echo "XDATCAR files combined. Total frames: $(grep -c "^Direct configuration" "XDATCAR_combined") (Limited to $FRAME_LIMIT)"

