#!/bin/bash
# Background-guard check on our TCGA slides (scripts/models/check_background_guard.py), 1 GPU, ~10 min.
#     sbatch scripts/models/guard_check_gpu.sbatch
#SBATCH --job-name=guard_check
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/models/check_background_guard.py --slides /projects/bfqi/$USER/tcga_slides/*.svs --run guard_v1 --workers 14
