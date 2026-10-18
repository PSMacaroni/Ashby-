#!/bin/bash
# Test our trained models on the provided datasets they were NOT trained on (1 GPU, ~10 min):
# embed NCT-CRC-HE-100K with CONCH, then run eval_provided.py (8 classes + nerve/vessel + PathGen captions).
# Needs data/models/conch_lr_starc9_lr.joblib and data/models/conch_structures_v1.joblib. From the project folder:
#
#     sbatch scripts/models/eval_provided_gpu.sbatch
#SBATCH --job-name=eval_provided
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
[ -f data/embeddings/nct_100k.npz ] || python scripts/models/embed_tiles.py \
    --root /projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples --name nct_100k --workers 14 || exit 1
python scripts/models/eval_provided.py --run provided_v1
