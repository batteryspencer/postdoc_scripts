#!/bin/bash

# Directory pattern to match
DIR_PATTERN="seg??"

# Frame limit variable
FRAME_LIMIT=5000

# Temporary file for concatenation
TEMP_FILE="temp_xdatcar"

# Check if temp file exists and remove it
[ -f "$TEMP_FILE" ] && rm "$TEMP_FILE"

# Loop through directories and concatenate XDATCAR files
for dir in $DIR_PATTERN; do
    cat "${dir}/XDATCAR" >> "${TEMP_FILE}"
done

# Extract the number of frames needed
awk -v limit=$FRAME_LIMIT '/^Direct configuration/ {n++} n<=limit {print}' "$TEMP_FILE" > "XDATCAR_combined"

# Clean up
rm "$TEMP_FILE"

echo "XDATCAR files combined. Total frames: $(grep -c "^Direct configuration" "XDATCAR_combined") (Limited to $FRAME_LIMIT)"
