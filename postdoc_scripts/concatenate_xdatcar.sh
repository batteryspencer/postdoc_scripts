#!/bin/bash

# Directory pattern to match
DIR_PATTERN="RUN_??"

# Frame limit variable
FRAME_LIMIT=5000

# Temporary file for concatenation
TEMP_FILE="temp_xdatcar"

# Check if temp file exists and remove it
[ -f "$TEMP_FILE" ] && rm "$TEMP_FILE"

# Loop through directories and concatenate XDATCAR files
for dir in $DIR_PATTERN; do
    if [ -d "$dir" ]; then
        cat "$dir/XDATCAR" >> "$TEMP_FILE"
    fi
done

# Add the current directory's XDATCAR
cat XDATCAR >> "$TEMP_FILE"

# Extract the number of frames needed
awk -v limit=$FRAME_LIMIT '/^Direct configuration/ {n++} n<=limit {print}' "$TEMP_FILE" > "XDATCAR_combined"

# Clean up
rm "$TEMP_FILE"

echo "XDATCAR files combined. Total frames: $(grep -c "^Direct configuration" "XDATCAR_combined") (Limited to $FRAME_LIMIT)"

