#!/bin/bash
# Train + test the CONCH / MUSK / CONCH+MUSK classifiers (scripts/models/compare_backbones.py) on CPU.
# Uses the team's CPU hours (bfqi-delta-cpu), not GPU hours. From the project folder on Delta:
#
#     mkdir -p data/logs && sbatch scripts/models/compare_cpu.sbatch --run musk_v1
#     tail -f data/logs/compare_backbones-<jobid>.out
#SBATCH --job-name=compare_backbones
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/models/compare_backbones.py "$@"
