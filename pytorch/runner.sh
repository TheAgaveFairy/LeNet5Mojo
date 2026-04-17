#!/bin/bash

# Claude 4 generated, only minor edits
# LeNet GPU Benchmark Runner
# Runs ./lenetgpu multiple times and appends results to resultsO3.txt

# Configuration
EXECUTABLE="./pytorch.py"
OUTPUT_FILE="resultspytorch.txt"
DEFAULT_RUNS=10

# Parse command line arguments
RUNS=${1:-$DEFAULT_RUNS}

# Validate inputs
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -le 0 ]; then
    echo "Error: Number of runs must be a positive integer"
    echo "Usage: $0 [number_of_runs]"
    echo "Example: $0 10"
    exit 1
fi

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found"
    echo "Make sure the file exists and is executable"
    exit 1
fi

# Check if executable is actually executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: '$EXECUTABLE' is not executable"
    echo "Run: chmod +x $EXECUTABLE"
    exit 1
fi

# Create header with timestamp if file doesn't exist
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "=== LeNet GPU Benchmark Results ===" > "$OUTPUT_FILE"
    echo "Started: $(date)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
else
    echo "" >> "$OUTPUT_FILE"
    echo "--- New benchmark session: $(date) ---" >> "$OUTPUT_FILE"
fi

echo "Running LeNet GPU benchmark $RUNS times..."
echo "Results will be appended to: $OUTPUT_FILE"
echo "Executable: $EXECUTABLE"
echo ""

# Run the benchmark multiple times
for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS..."
    
    # Add run header to output file
    echo "=== Run $i/$RUNS - $(date) ===" >> "$OUTPUT_FILE"
    
    # Run the executable and append output
    if "$EXECUTABLE" >> "$OUTPUT_FILE" 2>&1; then
        echo "  ✓ Run $i completed successfully"
    else
        echo "  ✗ Run $i failed (exit code: $?)"
        echo "ERROR: Run $i failed with exit code $?" >> "$OUTPUT_FILE"
    fi
    
    # Add separator between runs
    echo "" >> "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Small delay between runs to let system settle
    if [ $i -lt $RUNS ]; then
        sleep 1
    fi
done

# Add completion timestamp
echo "Benchmark completed: $(date)" >> "$OUTPUT_FILE"
echo ""

echo ""
echo "Benchmark completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "Total runs: $RUNS"

# Show a summary of the output file
if command -v wc >/dev/null 2>&1; then
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo "Output file has $LINES lines"
fi

if command -v tail >/dev/null 2>&1; then
    echo ""
    echo "Last few lines of results:"
    echo "=========================="
    tail -n 10 "$OUTPUT_FILE"
fi
