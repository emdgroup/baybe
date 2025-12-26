#!/bin/bash
#SBATCH --job-name=baybe_CT_I_BM_tl
#SBATCH --output=logs/aryl_halide_CT_I_BM_tl_%j.out
#SBATCH --error=logs/aryl_halide_CT_I_BM_tl_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# Change to project directory
cd ~/projects/baybe

# Activate environment (relative to project directory)
source baybe/bin/activate

# Create shared output directory with timestamp
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
SHARED_DIR="results_aryl_halide_benchmarks_${TIMESTAMP}"
mkdir -p "$SHARED_DIR"
mkdir -p logs

# Set parallelization environment variables
export BAYBE_PARALLEL_PERCENTAGE_RUNS=True
export BAYBE_PARALLEL_SIMULATION_RUNS=True

# Set default run mode if not provided
RUNMODE=${BENCHMARK_RUNMODE:-DEFAULT}

echo "Starting aryl_halide_CT_I_BM_tl benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Run Mode: $RUNMODE"
echo "Shared results directory: $SHARED_DIR"
echo "Parallelization: PERCENTAGE=$BAYBE_PARALLEL_PERCENTAGE_RUNS, SIMULATION=$BAYBE_PARALLEL_SIMULATION_RUNS"
echo ""

# Change to shared output directory
cd "$SHARED_DIR"

# Create benchmark-specific subdirectory
mkdir -p CT_I_BM_results
cd CT_I_BM_results

# Record start time
start_time=$(date +%s)

# Run benchmark
echo "Running benchmark: aryl_halide_CT_I_BM_tl (mode: $RUNMODE)"
python -m benchmarks --benchmark-list aryl_halide_CT_I_BM_tl --runmode $RUNMODE

exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "BENCHMARK COMPLETED SUCCESSFULLY"
    echo "Duration: ${duration}s ($(echo "$duration / 60" | bc -l | xargs printf "%.1f")m)"
else
    echo "BENCHMARK FAILED"
    echo "Duration: ${duration}s"
    echo "Exit code: $exit_code"
fi
echo "Results saved to: $(pwd)"
echo "=========================================="

exit $exit_code