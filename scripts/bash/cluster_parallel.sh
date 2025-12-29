#!/bin/bash

# cluster - Execute a program multiple times in parallel and concatenate results
# Usage: ./cluster <program> <total_steps> <output_file> [num_cores]

# Check arguments
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: ./cluster <program> <total_steps> <output_file> [num_cores]"
    echo ""
    echo "Arguments:"
    echo "  program      - Path to executable (must accept: program <steps> <index> <output_file>)"
    echo "  total_steps  - Number of times to run the program (0 to total_steps-1)"
    echo "  output_file  - Final output file to create in current directory"
    echo "  num_cores    - Optional: number of parallel jobs (default: all available cores)"
    echo ""
    echo "Requires: GNU parallel (install with: sudo apt-get install parallel)"
    exit 1
fi

PROGRAM="$1"
TOTAL_STEPS="$2"
OUTPUT_FILE="$3"
NUM_CORES="${4:-0}"  # Default to 0 (use all cores)

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed"
    echo "Install with: sudo apt-get install parallel"
    echo ""
    echo "Falling back to sequential execution..."
    USE_PARALLEL=false
else
    USE_PARALLEL=true
fi

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

# Validate num_cores if provided
if [ "$NUM_CORES" -ne 0 ] && ! [[ "$NUM_CORES" =~ ^[0-9]+$ ]]; then
    echo "Error: num_cores must be a positive integer or 0 for auto"
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

# Function to run a single job
run_job() {
    local total=$1
    local index=$2
    local program=$3
    local temp_dir=$4
    
    local output_file="${temp_dir}/output_${index}.dat"
    "$program" "$total" "$index" "$output_file"
    
    if [ $? -ne 0 ]; then
        echo "Error: Job $index failed" >&2
        return 1
    fi
}

export -f run_job

if [ "$USE_PARALLEL" = true ]; then
    # Parallel execution
    if [ "$NUM_CORES" -eq 0 ]; then
        CORES_FLAG=""
        echo "Running $TOTAL_STEPS jobs in parallel (using all available cores)..."
    else
        CORES_FLAG="-j $NUM_CORES"
        echo "Running $TOTAL_STEPS jobs in parallel (using $NUM_CORES cores)..."
    fi
    
    # Create job list and run with GNU parallel
    # --keep-order ensures jobs are processed in order
    # --halt now,fail=1 stops immediately if any job fails
    seq 0 $((TOTAL_STEPS - 1)) | \
        parallel $CORES_FLAG --bar --keep-order --halt now,fail=1 \
        run_job "$TOTAL_STEPS" {} "$PROGRAM_ABS" "$TEMP_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Error: Some jobs failed"
        exit 1
    fi
else
    # Sequential execution fallback
    echo "Running $TOTAL_STEPS jobs sequentially..."
    for (( i=0; i<$TOTAL_STEPS; i++ )); do
        TEMP_OUTPUT="output_${i}.dat"
        
        "$PROGRAM_ABS" "$TOTAL_STEPS" "$i" "$TEMP_OUTPUT"
        
        if [ $? -ne 0 ]; then
            echo "Error: Program failed at step $i"
            exit 1
        fi
        
        # Progress indicator
        if [ $((($i + 1) % 10)) -eq 0 ] || [ $i -eq $(($TOTAL_STEPS - 1)) ]; then
            echo "  Completed: $(($i + 1))/$TOTAL_STEPS"
        fi
    done
fi

echo "All jobs completed successfully"

# Concatenate all output files in order
echo "Concatenating results to $OUTPUT_FILE..."
> "$OUTPUT_ABS"  # Create/truncate output file

# Wait a moment to ensure all files are written
sync

for (( i=0; i<$TOTAL_STEPS; i++ )); do
    TEMP_OUTPUT="output_${i}.dat"
    
    # Wait for file to exist (with timeout)
    timeout=10
    counter=0
    while [ ! -f "$TEMP_OUTPUT" ] && [ $counter -lt $timeout ]; do
        sleep 0.1
        counter=$((counter + 1))
    done
    
    if [ -f "$TEMP_OUTPUT" ]; then
        cat "$TEMP_OUTPUT" >> "$OUTPUT_ABS"
    else
        echo "Error: Output file $TEMP_OUTPUT not found after waiting" >&2
        exit 1
    fi
done

echo "Results written to: $OUTPUT_FILE"
echo "Number of lines: $(wc -l < "$OUTPUT_ABS")"

# Return to original directory
cd "$CURRENT_DIR" || exit 1

echo "Done!"