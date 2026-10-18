#!/bin/bash
# Same first-look test on CPUs only (slower, but uses the team's large CPU allocation
# bfqi-delta-cpu instead of the 10 GPU hours). From the project folder on Delta:
#
#     mkdir -p data/logs && sbatch scripts/models/first_look_cpu.sbatch
#     squeue -u $USER
#     tail -f data/logs/first_look_cpu-<jobid>.out
#SBATCH --job-name=first_look_cpu
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/models/first_look.py \
    --slide /work/hdd/bfqi/natmakur/tcga_slides/TCGA-A6-3809-01Z-00-DX1.c26f03e8-c285-4a66-925d-ae9cba17d7b3.svs
