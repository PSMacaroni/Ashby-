#!/bin/bash
# Showcase run (main 17bdafc): 5 slides through the whole pipeline in ONE GPU job:
# process_slide.py (tiles, QC, CONCH tissue model, immune, MUSK blood check) -> score_phrases.py -> analyze_slide.py
#SBATCH --job-name=showcase_v1
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:15:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=/projects/bfqi/hjeong6/showcase_v1
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "start $(date +%T)"
python scripts/process_slide.py --slide $(cat showcase/slides.txt) --out "$OUT" --slides-csv showcase/slides.csv \
    --workers 14 --skip-done
echo "slides done $(date +%T)"
for d in "$OUT"/TCGA-*/; do
    python scripts/score_phrases.py "$d" --model conch > "$d/phrases.log" 2>&1; p=$?
    python scripts/analyze_slide.py "$d" > "$d/analysis.log" 2>&1; a=$?
    echo "$(date +%T) $(basename "$d") phrases_exit=$p analysis_exit=$a"
done
echo "done $(date +%T)"
