# BayBE Aryl Halide Benchmarks - HPC Cluster Execution Guide (Multi-Node)

This guide provides step-by-step instructions to run the three aryl halide transfer learning benchmarks on your HPC cluster using a multi-node approach.

## Prerequisites

- Access to HPC cluster with SLURM workload manager
- BayBE codebase in your home directory (`~/baybe/`)
- Python 3.11+ environment with BayBE installed
- Access to A-series CPU compute nodes

## Multi-Node Architecture

**Approach**: Each benchmark runs as a separate SLURM job on its own dedicated node
- **3 separate jobs** submitted simultaneously
- **32 CPUs and 64GB RAM** per benchmark
- **Full nested parallelization** within each benchmark
- **Better scheduling** - medium-sized jobs get scheduled faster
- **Fault tolerance** - if one benchmark fails, others continue

## Files Overview

The following files have been created for multi-node cluster execution:

1. **`aryl_halide_CT_I_BM_tl.slurm`** - SLURM script for CT-I-BM benchmark
2. **`aryl_halide_CT_IM_tl.slurm`** - SLURM script for CT-IM benchmark
3. **`aryl_halide_IP_CP_tl.slurm`** - SLURM script for IP-CP benchmark
4. **`submit_all_benchmarks.sh`** - Launcher that submits all three jobs
5. **`analyze_benchmark_results.py`** - Results analysis script
6. **`CLUSTER_EXECUTION_GUIDE.md`** - This guide

## Step-by-Step Execution

### Step 1: Prepare the Environment

```bash
# 1. Connect to your cluster login node
ssh your_username@cluster.example.com

# 2. Navigate to your BayBE directory
cd ~/baybe

# 3. Ensure your Python environment is set up
module load python/3.11
source ~/baybe_env/bin/activate  # Adjust path to your virtual environment

# 4. Verify BayBE installation
python -c "import baybe; print('BayBE version:', baybe.__version__)"

# 5. Create logs directory
mkdir -p logs

# 6. Make scripts executable
chmod +x submit_all_benchmarks.sh
```

### Step 2: Submit All Benchmark Jobs

```bash
# Submit all three benchmarks with one command
./submit_all_benchmarks.sh
```

You should see output like:
```
BayBE Aryl Halide Benchmarks - Multi-Node Launcher
==================================================
Submitting three benchmarks as separate SLURM jobs...

Shared results directory: results_aryl_halide_benchmarks_20241226_143022
Each benchmark will run on a separate node with 32 CPUs and 64GB RAM

Submitting benchmark jobs...
CT-I-BM job submitted: Submitted batch job 1234567
CT-IM job submitted: Submitted batch job 1234568
IP-CP job submitted: Submitted batch job 1234569

All jobs submitted successfully!
Job IDs: 1234567, 1234568, 1234569
```

### Step 3: Monitor Job Progress

```bash
# Check status of all your jobs
squeue -u $USER

# Monitor specific benchmark jobs (replace with your job IDs)
squeue -j 1234567,1234568,1234569

# Watch jobs in real-time
watch 'squeue -u $USER'

# Check detailed job information
scontrol show job 1234567  # Replace with actual job ID
```

### Step 4: Monitor Individual Benchmark Logs

```bash
# Follow logs from individual benchmarks (replace JOBID with actual job IDs)
tail -f logs/aryl_halide_CT_I_BM_tl_1234567.out
tail -f logs/aryl_halide_CT_IM_tl_1234568.out
tail -f logs/aryl_halide_IP_CP_tl_1234569.out

# Check for errors
tail -f logs/aryl_halide_CT_I_BM_tl_1234567.err
```

### Step 5: Check Resource Usage (Optional)

```bash
# Check resource usage while jobs are running
sstat -j 1234567,1234568,1234569 --format=JobID,AveCPU,AvePages,AveRSS,AveVMSize

# Get job efficiency after completion
seff 1234567  # Replace with actual job ID
```

### Step 6: Analyze Results

After all jobs complete (typically 1-3 hours total):

```bash
# Find your results directory
ls -la results_aryl_halide_benchmarks_*

# The directory structure will be:
# results_aryl_halide_benchmarks_YYYYMMDD_HHMMSS/
# ├── CT_I_BM_results/     # Results from CT-I-BM benchmark
# ├── CT_IM_results/       # Results from CT-IM benchmark
# └── IP_CP_results/       # Results from IP-CP benchmark

# Run analysis script
python analyze_benchmark_results.py results_aryl_halide_benchmarks_YYYYMMDD_HHMMSS/

# Save analysis report
python analyze_benchmark_results.py results_aryl_halide_benchmarks_YYYYMMDD_HHMMSS/ --save-report benchmark_analysis.txt

# View detailed analysis
python analyze_benchmark_results.py results_aryl_halide_benchmarks_YYYYMMDD_HHMMSS/ --verbose
```

### Step 7: Retrieve Results

```bash
# Create a summary of all result files
ls -la results_aryl_halide_benchmarks_*/*/

# Copy results to your local machine (run from your local terminal)
scp -r your_username@cluster.example.com:~/baybe/results_aryl_halide_benchmarks_* ./local_results/
```

## Expected Resource Usage Per Job

**Individual Job Configuration:**
- Time limit: 2 hours per benchmark
- CPUs: 32 per benchmark
- Memory: 64GB per benchmark
- Node: 1 per benchmark (total: 3 nodes)

**Total Resource Requirements:**
- Total CPUs: 96 (across 3 nodes)
- Total Memory: 192GB (across 3 nodes)
- Total Nodes: 3

**Parallelization Strategy:**
- **Job level**: 3 benchmarks run on separate nodes
- **Percentage level**: Each benchmark parallelizes over percentages (`BAYBE_PARALLEL_PERCENTAGE_RUNS=True`)
- **Simulation level**: Each percentage parallelizes simulation runs (`BAYBE_PARALLEL_SIMULATION_RUNS=True`)

## Advantages of Multi-Node Approach

1. **Faster Scheduling**: 32-CPU jobs schedule faster than 96-CPU jobs
2. **Better Resource Utilization**: Uses available nodes efficiently
3. **Fault Tolerance**: Failed benchmark doesn't affect others
4. **Flexibility**: Works with various node sizes in your A-series CPU compute nodes
5. **Parallel Execution**: Jobs run simultaneously if multiple nodes available

## Troubleshooting

### Jobs Don't Start
```bash
# Check queue and node availability
sinfo -N -l
squeue -u $USER

# Check your account limits
sacctmgr show user $USER withassoc

# Check partition access
sinfo -s
```

### Job Fails Due to Resource Constraints
```bash
# Check individual job error logs
cat logs/aryl_halide_CT_I_BM_tl_JOBID.err

# If memory issues, increase memory in SLURM scripts:
# Edit *.slurm files: change --mem=64G to --mem=96G
```

### One Benchmark Fails
```bash
# Check which benchmark failed
python analyze_benchmark_results.py results_aryl_halide_benchmarks_*/

# Resubmit individual failed benchmark
sbatch aryl_halide_CT_I_BM_tl.slurm  # Example for CT_I_BM
```

### Disable Nested Parallelization (if needed)
Edit individual `.slurm` files to reduce parallelization:

```bash
# Option 1: Disable percentage parallelization
# Change: export BAYBE_PARALLEL_PERCENTAGE_RUNS=False
# Reduce: --cpus-per-task=16

# Option 2: Disable simulation parallelization
# Change: export BAYBE_PARALLEL_SIMULATION_RUNS=False
# Keep:   --cpus-per-task=32
```

## Expected Timeline

**Typical execution timeline:**
- Job submission: < 1 minute
- Queue wait time: 5-60 minutes (depends on cluster load)
- Parallel execution: 60-120 minutes (all three benchmarks)
- Results analysis: 1-2 minutes

**Best case**: All jobs start immediately → 60-90 minutes total
**Typical case**: Some queue time → 90-150 minutes total
**Worst case**: High cluster load → 3+ hours total

## Monitoring Commands Summary

```bash
# Essential monitoring commands
squeue -u $USER                                    # Check job status
tail -f logs/aryl_halide_CT_I_BM_tl_JOBID.out     # Follow specific log
sstat -j JOBID --format=JobID,AveCPU,AveRSS       # Resource usage
seff JOBID                                         # Job efficiency (after completion)
```

## Expected Output

**Successful execution produces:**
- Shared results directory: `results_aryl_halide_benchmarks_YYYYMMDD_HHMMSS/`
- Benchmark subdirectories: `CT_I_BM_results/`, `CT_IM_results/`, `IP_CP_results/`
- JSON result files in each subdirectory
- Individual SLURM logs: `logs/aryl_halide_*_JOBID.out`
- Analysis report when running analysis script

**Performance validation:**
- Each benchmark should utilize ~32 CPUs when running
- Memory usage should be 30-60GB per benchmark
- All three benchmarks may run simultaneously if nodes available
- Total execution time should be similar to single longest benchmark

You're now ready to run all three aryl halide benchmarks efficiently on your HPC cluster!