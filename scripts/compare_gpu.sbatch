#!/bin/bash
# Same as compare_cpu.sbatch, but on a GPU node, for when the CPU queues are backed up (Oct 17: regular `cpu`
# queue ~2 days, cpu-interactive ~1 hour). The work is CPU-only (scikit-learn), so the GPU sits idle, but the
# GPU reservation starts right away. Runs the tissue and structures parts side by side (8 cores each), then
# combines them into one report. Costs GPU hours (1 GPU for the run time). From the project folder on Delta:
#
#     mkdir -p data/logs && sbatch scripts/compare_gpu.sbatch musk_v1
#SBATCH --job-name=compare_backbones_gpu
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62g
#SBATCH --time=01:30:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
RUN=${1:-musk_v1}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python scripts/compare_backbones.py --run $RUN --parts tissue > data/logs/compare_tissue-$SLURM_JOB_ID.log 2>&1 &
python scripts/compare_backbones.py --run $RUN --parts structures > data/logs/compare_structures-$SLURM_JOB_ID.log 2>&1 &
wait
python scripts/compare_backbones.py --run $RUN --parts report
