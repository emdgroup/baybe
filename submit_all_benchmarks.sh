#!/bin/bash

# BayBE Aryl Halide Benchmarks - Multi-Node Launcher
# This script submits each benchmark as a separate SLURM job
# Usage: ./submit_all_benchmarks.sh

set -e

echo "BayBE Aryl Halide Benchmarks - Multi-Node Launcher"
echo "=================================================="
echo "Submitting three benchmarks as separate SLURM jobs..."

# Get timestamp for unique job naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TIMESTAMP

# Create shared results directory and logs
mkdir -p "results_aryl_halide_benchmarks_${TIMESTAMP}"
mkdir -p logs

echo "Shared results directory: results_aryl_halide_benchmarks_${TIMESTAMP}"
echo "Each benchmark will run on a separate node with 32 CPUs and 64GB RAM"
echo ""

# Submit all three benchmark jobs
echo "Submitting benchmark jobs..."

# Job 1: CT-I-BM Transfer Learning
JOB1_ID=$(sbatch --export=TIMESTAMP="$TIMESTAMP" aryl_halide_CT_I_BM_tl.slurm)
echo "CT-I-BM job submitted: $JOB1_ID"

# Job 2: CT-IM Transfer Learning
JOB2_ID=$(sbatch --export=TIMESTAMP="$TIMESTAMP" aryl_halide_CT_IM_tl.slurm)
echo "CT-IM job submitted: $JOB2_ID"

# Job 3: IP-CP Transfer Learning
JOB3_ID=$(sbatch --export=TIMESTAMP="$TIMESTAMP" aryl_halide_IP_CP_tl.slurm)
echo "IP-CP job submitted: $JOB3_ID"

echo ""
echo "All jobs submitted successfully!"

# Extract job IDs for monitoring
JOB1_NUM=$(echo $JOB1_ID | grep -o '[0-9]\+')
JOB2_NUM=$(echo $JOB2_ID | grep -o '[0-9]\+')
JOB3_NUM=$(echo $JOB3_ID | grep -o '[0-9]\+')

echo "Job IDs: $JOB1_NUM, $JOB2_NUM, $JOB3_NUM"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  squeue -j $JOB1_NUM,$JOB2_NUM,$JOB3_NUM"
echo ""
echo "Monitor individual job logs:"
echo "  tail -f logs/aryl_halide_CT_I_BM_tl_${JOB1_NUM}.out"
echo "  tail -f logs/aryl_halide_CT_IM_tl_${JOB2_NUM}.out"
echo "  tail -f logs/aryl_halide_IP_CP_tl_${JOB3_NUM}.out"
echo ""
echo "Results will be collected in:"
echo "  results_aryl_halide_benchmarks_${TIMESTAMP}/"
echo "    ├── CT_I_BM_results/"
echo "    ├── CT_IM_results/"
echo "    └── IP_CP_results/"
echo ""
echo "When all jobs complete, analyze results with:"
echo "  python analyze_benchmark_results.py results_aryl_halide_benchmarks_${TIMESTAMP}/"
echo ""
echo "Expected runtime: 60-120 minutes per benchmark"
echo "Jobs may run in parallel if multiple nodes are available"