#!/bin/bash
# Utility script to analyze bad nodes log and identify problematic nodes
# Usage: ./analyze_bad_nodes.sh [bad_nodes.log] [days_back] [min_failures]
# Defaults: last 3 days, 3+ failures to flag a node

BAD_NODE_LOG="${1:-/scratch/gautschi/pasumarv/bad_nodes.log}"
DAYS_BACK="${2:-3}"  # Only consider entries from last N days
MIN_FAILURES="${3:-3}"  # Minimum failures to flag a node

# Calculate cutoff date
CUTOFF_DATE=$(date -d "$DAYS_BACK days ago" +%Y-%m-%d)

if [ ! -f "$BAD_NODE_LOG" ]; then
    echo "Bad nodes log not found: $BAD_NODE_LOG"
    echo "No bad nodes have been logged yet."
    exit 0
fi

echo "=========================================="
echo "       BAD NODES ANALYSIS REPORT         "
echo "=========================================="
echo "Log file: $BAD_NODE_LOG"
echo "Total entries (all time): $(wc -l < "$BAD_NODE_LOG")"
echo "Filtering: last $DAYS_BACK days (since $CUTOFF_DATE)"
echo "Threshold: $MIN_FAILURES+ failures to flag"
echo ""

# Filter log to only recent entries
RECENT_LOG=$(awk -v cutoff="$CUTOFF_DATE" '$1 >= cutoff' "$BAD_NODE_LOG")
RECENT_COUNT=$(echo "$RECENT_LOG" | grep -c .)
echo "Entries in window: $RECENT_COUNT"
echo ""

echo "--- Node Frequency (most problematic first, last $DAYS_BACK days) ---"
# Extract node names and count occurrences from recent entries only
# Node format in log: a[065,305] or similar
echo "$RECENT_LOG" | grep -oE 'a\[[0-9,]+\]' | \
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
echo "Copy this to exclude problematic nodes (${MIN_FAILURES}+ failures in last $DAYS_BACK days):"
echo ""

# Generate exclude list for nodes that appear MIN_FAILURES+ times in recent window
nodes_to_exclude=$(echo "$RECENT_LOG" | grep -oE 'a\[[0-9,]+\]' | \
    tr ',' '\n' | \
    sed 's/a\[//; s/\]//' | \
    sort | uniq -c | \
    awk -v min="$MIN_FAILURES" '$1 >= min {print "a" $2}' | \
    tr '\n' ',' | sed 's/,$//')

if [ -n "$nodes_to_exclude" ]; then
    echo "#SBATCH --exclude=$nodes_to_exclude"
else
    echo "No nodes have failed ${MIN_FAILURES}+ times in the last $DAYS_BACK days."
fi

echo ""
echo "=========================================="
