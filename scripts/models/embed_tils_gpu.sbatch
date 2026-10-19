#!/bin/bash
# Embed the TCGA-TILs colon/rectum patches for the immune-detector test, 1 GPU (~5 min):
#   CONCH embeddings of the TCGA-TILs colon/rectum patches (from the get-tils CPU job) -> tils_<cancer>_<split>.npz
#   (the slide background-guard check is separate: scripts/models/guard_check_gpu.sbatch)
# From the project folder:  sbatch scripts/models/embed_tils_gpu.sbatch
#SBATCH --job-name=embed_tils
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
T=/projects/bfqi/$USER/datasets/tcga_tils/TCGA-TILs/images-tcga-tils
for cancer in coad read; do
    for split_dir in $T/$cancer/*/; do
        split=$(basename $split_dir)
        python scripts/models/embed_tiles.py --root $split_dir --name tils_${cancer}_${split} \
            --labels til-negative til-positive --workers 14 || exit 1
    done
done
