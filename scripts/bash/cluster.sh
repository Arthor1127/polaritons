#!/bin/bash

# cluster - Execute a program multiple times and concatenate results
# Usage: ./cluster <program> <total_steps> <output_file>

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: ./cluster <program> <total_steps> <output_file>"
    echo ""
    echo "Arguments:"
    echo "  program      - Path to executable (must accept: program <steps> <index> <output>)"
    echo "  total_steps  - Number of times to run the program (0 to total_steps-1)"
    echo "  output_file  - Final output file to create in current directory"
    exit 1
fi

PROGRAM="$1"
TOTAL_STEPS="$2"
OUTPUT_FILE="$3"

# Validate program exists and is executable
if [ ! -f "$PROGRAM" ]; then
    echo "Error: Program '$PROGRAM' not found"
    exit 1
fi

if [ ! -x "$PROGRAM" ]; then
    echo "Error: Program '$PROGRAM' is not executable"
    echo "Try: chmod +x $PROGRAM"
    exit 1
fi

# Validate total_steps is a positive integer
if ! [[ "$TOTAL_STEPS" =~ ^[0-9]+$ ]] || [ "$TOTAL_STEPS" -le 0 ]; then
    echo "Error: total_steps must be a positive integer"
    exit 1
fi

# Get absolute paths
PROGRAM_ABS=$(realpath "$PROGRAM")
OUTPUT_ABS=$(realpath "$OUTPUT_FILE")
CURRENT_DIR=$(pwd)

# Create temporary directory
TEMP_DIR=$(mktemp -d -t cluster_XXXXXX)
echo "Created temporary directory: $TEMP_DIR"

# Trap to cleanup on exit
cleanup() {
    echo "Cleaning up temporary directory..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Change to temp directory
cd "$TEMP_DIR" || exit 1

# Execute program for each step
echo "Running $TOTAL_STEPS jobs..."
for (( i=0; i<$TOTAL_STEPS; i++ )); do
    TEMP_OUTPUT="output_${i}.dat"
    
    # Run the program
    "$PROGRAM_ABS" "$TOTAL_STEPS" "$i" "$TEMP_OUTPUT"
    
    # Check if execution was successful
    if [ $? -ne 0 ]; then
        echo "Error: Program failed at step $i"
        exit 1
    fi
    
    # Progress indicator
    if [ $((($i + 1) % 10)) -eq 0 ] || [ $i -eq $(($TOTAL_STEPS - 1)) ]; then
        echo "  Completed: $(($i + 1))/$TOTAL_STEPS"
    fi
done

echo "All jobs completed successfully"

# Concatenate all output files in order
echo "Concatenating results to $OUTPUT_FILE..."
> "$OUTPUT_ABS"  # Create/truncate output file

for (( i=0; i<$TOTAL_STEPS; i++ )); do
    TEMP_OUTPUT="output_${i}.dat"
    if [ -f "$TEMP_OUTPUT" ]; then
        cat "$TEMP_OUTPUT" >> "$OUTPUT_ABS"
    else
        echo "Warning: Output file $TEMP_OUTPUT not found"
    fi
done

echo "Results written to: $OUTPUT_FILE"
echo "Number of lines: $(wc -l < "$OUTPUT_ABS")"

# Return to original directory
cd "$CURRENT_DIR" || exit 1

echo "Done!"