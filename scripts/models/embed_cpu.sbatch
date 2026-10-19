#!/bin/bash
# Same as embed_gpu.sbatch but on CPUs (bfqi-delta-cpu, the big allocation): fine for quick tests
# (--limit 5) and small sets; too slow for ~100k tiles. From the project folder on Delta:
#
#     mkdir -p data/logs
#     sbatch scripts/models/embed_cpu.sbatch --root /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples --name test --limit 5
#SBATCH --job-name=embed_cpu
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/models/embed_tiles.py --workers 8 --batch-size 32 "$@"
