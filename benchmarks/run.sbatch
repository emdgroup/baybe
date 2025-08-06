#!/bin/bash

#SBATCH -o /dfs/is/home/m332296/slurm/baybe_benchmark_%j.job
#SBATCH -e /dfs/is/home/m332296/slurm/baybe_benchmark_%j.job
#SBATCH -J baybe_benchmark
#SBATCH -c 10  #CPU cores required
#SBATCH -n 1
#SBATCH --mem=10G #Memory required
#SBATCH -t 23:00:00 #Job runtime
#SBATCH --nice=0
#SBATCH --nodelist=stdg22 # Specify the node to run on


while getopts p:e: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
        e) env=${OPTARG};;
    esac
done

cd $path

conda activate $env

python -m benchmarks --name $SLURM_NODELIST