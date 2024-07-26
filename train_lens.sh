#!/bin/bash
#SBATCH --cpus-per-task=18
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -t 0:02:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

# load modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

python3 TunedLens.py
