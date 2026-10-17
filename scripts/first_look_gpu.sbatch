#!/bin/bash
# First-look test on 1 GPU. From the project folder on Delta:
#
#     mkdir -p data/logs && sbatch scripts/first_look_gpu.sbatch
#     squeue -u $USER                      # is it waiting / running?
#     tail -f data/logs/first_look_gpu-<jobid>.out
#
# Uses the team's GPU hours (bfqi-delta-gpu, 10 hours total for the whole team).
# Billing is for the time the job actually runs, not the --time limit.
#SBATCH --job-name=first_look_gpu
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/first_look.py \
    --slide /work/hdd/bfqi/natmakur/tcga_slides/TCGA-A6-3809-01Z-00-DX1.c26f03e8-c285-4a66-925d-ae9cba17d7b3.svs
