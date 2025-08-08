#!/bin/bash

while getopts b:e:o:l:s: flag
do
    case "${flag}" in
        b) baybe=${OPTARG};; # Path to baybe repository for running benchmarks
        e) env=${OPTARG};; # Conda env to use for uv command
        o) outdir=${OPTARG};; # Optional output directory for results
        l) benchmark_list=${OPTARG};; # Optional list of benchmarks to run.
        # If multiple should be space-separated and enclosed in quotes, e.g. "benchmark1 benchmark2"
        s) smoketest=${OPTARG};; # Optional smoketest setting
    esac
done

# Prerequisites for creating uv env
source $HOME/.bashrc
conda activate $env
cd $baybe

# Create result file name suffix to store information on node and prevent name collisions
node=$(hostname)
name="$node"_"$RANDOM"
echo $name

# Optional benchmark flags
flags=""
[ -n "$benchmark_list" ] && flags+="--benchmark-list $benchmark_list "
[ -n "$outdir" ] && flags+="--outdir $outdir "
[ -n "$smoketest" ] && flags+="--smoketest $smoketest "

uv run --python 3.12 --with "baybe[benchmarking] @ ." python -m benchmarks --name $name $flags

echo "FINISHED!"