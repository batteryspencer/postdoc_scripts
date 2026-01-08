#!/bin/bash
# Utility script to analyze bad nodes log and identify problematic nodes
# Usage: ./analyze_bad_nodes.sh [bad_nodes.log]

BAD_NODE_LOG="${1:-/scratch/gautschi/pasumarv/bad_nodes.log}"

if [ ! -f "$BAD_NODE_LOG" ]; then
    echo "Bad nodes log not found: $BAD_NODE_LOG"
    echo "No bad nodes have been logged yet."
    exit 0
fi

echo "=========================================="
echo "       BAD NODES ANALYSIS REPORT         "
echo "=========================================="
echo "Log file: $BAD_NODE_LOG"
echo "Total entries: $(wc -l < "$BAD_NODE_LOG")"
echo ""

echo "--- Node Frequency (most problematic first) ---"
# Extract node names and count occurrences
# Node format in log: a[065,305] or similar
grep -oE 'a\[[0-9,]+\]' "$BAD_NODE_LOG" | \
    tr ',' '\n' | \
    sed 's/a\[//; s/\]//' | \
    sort | uniq -c | sort -rn | head -20

echo ""
echo "--- Recent Bad Node Entries (last 10) ---"
tail -10 "$BAD_NODE_LOG"

echo ""
echo "--- Entries by Date ---"
cut -d'|' -f1 "$BAD_NODE_LOG" | cut -d' ' -f1 | sort | uniq -c

echo ""
echo "--- Generate SLURM exclude list ---"
echo "Copy this to exclude problematic nodes:"
echo ""

# Generate exclude list for nodes that appear 2+ times
nodes_to_exclude=$(grep -oE 'a\[[0-9,]+\]' "$BAD_NODE_LOG" | \
    tr ',' '\n' | \
    sed 's/a\[//; s/\]//' | \
    sort | uniq -c | \
    awk '$1 >= 2 {print "a" $2}' | \
    tr '\n' ',' | sed 's/,$//')

if [ -n "$nodes_to_exclude" ]; then
    echo "#SBATCH --exclude=$nodes_to_exclude"
else
    echo "No nodes have failed 2+ times yet."
fi

echo ""
echo "=========================================="
