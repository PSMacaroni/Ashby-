#!/bin/bash
# GPU part of the whole-slide pipeline for many slides in ONE python process: CONCH + MUSK are loaded once, and the
# job holds the GPU only for process_slide.py. Run phrases + analysis afterwards on CPU (analyze_slides_cpu.sbatch).
# Finished slides are skipped (re-run safe). From the project folder on Delta:
#
#     sbatch scripts/embed_slides_gpu.sbatch <file with one .svs path per line> <out dir> <slides.csv> \
#         [--stop-at 2026-10-18T07:10]       # don't start another slide after this time
#SBATCH --job-name=embed_slides
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
LIST=$1; OUT=$2; CSV=$3
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/process_slide.py --slide $(cat "$LIST") --out "$OUT" --slides-csv "$CSV" --workers 14 --skip-done "${@:4}"
