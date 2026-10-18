#!/bin/bash
# Whole-slide pipeline for every slide in a folder, 1 GPU: process_slide.py (tiles + CONCH + tissue model +
# lymphocyte detector) -> score_phrases.py (flag phrases) -> analyze_slide.py (front, features, flags, risk).
# From the project folder on Delta:
#
#     sbatch scripts/process_slides_gpu.sbatch /projects/bfqi/$USER/tcga_slides
#     sbatch scripts/process_slides_gpu.sbatch /projects/bfqi/$USER/tcga_slides --extend   # only new mask cells
#
# Each result folder data/results/<slide_id>/ can then be copied to the laptop for the app.
#SBATCH --job-name=process_slides
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=03:30:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
SLIDES=${1:-/projects/bfqi/$USER/tcga_slides}
nvidia-smi --query-gpu=name,memory.total --format=csv
for s in $SLIDES/*.svs; do
    id=$(basename "$s" | cut -d. -f1)
    python scripts/process_slide.py --slide "$s" --slides-csv "$SLIDES/slides.csv" --workers 14 "${@:2}" || continue
    python scripts/score_phrases.py data/results/$id --model conch
    python scripts/analyze_slide.py data/results/$id
done
