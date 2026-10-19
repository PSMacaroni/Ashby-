#!/bin/bash
# Same as showcase_gpu.sbatch, for a given list of slides, analyzing only those slides' folders
# (so parallel jobs never touch each other's folders):  sbatch showcase/showcase_gpu_one.sbatch <slide list file>
#SBATCH --job-name=showcase_v1
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
OUT=/projects/bfqi/hjeong6/showcase_v1
LIST=$1
echo "start $(date +%T)"
python scripts/process_slide.py --slide $(cat "$LIST") --out "$OUT" --slides-csv showcase/slides.csv \
    --workers 14 --skip-done
echo "slides done $(date +%T)"
for s in $(cat "$LIST"); do
    n=$(basename "$s"); d=$OUT/${n%%.*}
    python scripts/score_phrases.py "$d" --model conch > "$d/phrases.log" 2>&1; p=$?
    python scripts/analyze_slide.py "$d" > "$d/analysis.log" 2>&1; a=$?
    echo "$(date +%T) $(basename "$d") phrases_exit=$p analysis_exit=$a"
done
echo "done $(date +%T)"
