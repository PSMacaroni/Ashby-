#!/bin/bash
# CONCH + MUSK embeddings of every labeled tile set for the backbone comparison (scripts/compare_backbones.py),
# on 1 GPU. Each model is loaded once; sets already embedded are skipped, so it is safe to re-submit.
# From the project folder on Delta, after the downloads (see docs/musk_experiment.md):
#
#     mkdir -p data/logs && sbatch scripts/embed_compare_gpu.sbatch            # both models, all sets
#     sbatch scripts/embed_compare_gpu.sbatch --sets spider_colon_train ...    # e.g. later, once SPIDER is in
#     tail -f data/logs/embed_compare_gpu-<jobid>.out
#
# Uses the team's GPU hours (bfqi-delta-gpu). Billing is for the time the job actually runs.
#SBATCH --job-name=embed_compare_gpu
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/embed_many.py --models conch musk --workers 14 "$@"
