#!/bin/bash
# CONCH embeddings for the SPIDER nerve / vessel / other tiles (from get_spider.py), 1 GPU, ~25k tiles,
# then trains + tests the side classifier (train_structures.py) in the same job.
# From the project folder on Delta (after both get_spider jobs finished):
#
#     sbatch scripts/embed_spider_gpu.sbatch
#SBATCH --job-name=embed_spider
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
D=/projects/bfqi/$USER/datasets
for organ in colorectal:colon skin:skin; do
    for split in train test; do
        python scripts/embed_tiles.py --root $D/spider_${organ%%:*}/$split --name spider_${organ##*:}_$split \
            --labels other nerve vessel --workers 14 || exit 1
    done
done

# Train + test right away in the same job (CPU work, ~2 min), so there is no second wait in the queue.
python scripts/train_structures.py --data colon=$D/spider_colorectal skin=$D/spider_skin --run conch_structures_v1
