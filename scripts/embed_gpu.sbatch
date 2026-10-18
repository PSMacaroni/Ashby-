#!/bin/bash
# CONCH embeddings for a folder of labeled tiles, on 1 GPU. Arguments go straight to embed_tiles.py.
# From the project folder on Delta:
#
#     mkdir -p data/logs
#     sbatch scripts/embed_gpu.sbatch --root /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples --name nct_100k
#     squeue -u $USER                      # is it waiting / running?
#     tail -f data/logs/embed_gpu-<jobid>.out
#
# Uses the GPU hours (bfqi-delta-gpu, 10 hours). Billing is for the time the job actually runs,
# not the --time limit. ~90k tiles take a few minutes; test first with --limit 5.
#SBATCH --job-name=embed_gpu
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=00:45:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/embed_tiles.py --workers 14 "$@"
