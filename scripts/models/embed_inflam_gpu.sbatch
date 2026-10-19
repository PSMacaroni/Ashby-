#!/bin/bash
# CONCH embeddings for all SPIDER "Inflammation" tiles (colon + skin), used to train the lymphocyte / immune detector.
# Get the tiles first (CPU, a few minutes each):
#     sbatch scripts/models/get_spider_cpu.sbatch --organ colorectal --out /projects/bfqi/$USER/datasets/spider_inflam_colorectal \
#         --targets Inflammation=inflammation --other-per-class 0
#     sbatch scripts/models/get_spider_cpu.sbatch --organ skin --out /projects/bfqi/$USER/datasets/spider_inflam_skin \
#         --targets Inflammation=inflammation --other-per-class 0
# then: sbatch scripts/models/embed_inflam_gpu.sbatch   (~1-2 min on 1 GPU)
#SBATCH --job-name=embed_inflam
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=00:20:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=/projects/bfqi/$USER/datasets
for organ in colorectal:colon skin:skin; do
    for split in train test; do
        python scripts/models/embed_tiles.py --root $D/spider_inflam_${organ%%:*}/$split --name spider_inflam_${organ##*:}_$split \
            --labels inflammation --workers 14 || exit 1
    done
done
