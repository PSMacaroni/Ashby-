#!/bin/bash
# CPU part of the whole-slide pipeline: score_phrases.py + analyze_slide.py for every processed slide folder in <out
# dir>, several slides at a time. Run after embed_slides_gpu.sbatch (e.g. sbatch --dependency=afterany:<gpu job>).
#
#     sbatch scripts/analyze_slides_cpu.sbatch <out dir> [slides at once, default 8]
#SBATCH --job-name=analyze_slides
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
OUT=$1; PAR=${2:-8}
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / PAR)) MKL_NUM_THREADS=$((SLURM_CPUS_PER_TASK / PAR))
one() {
    d=$1
    [ -f "$d/summary.json" ] || { echo "$(basename "$d"): not processed, skipped"; return; }
    python scripts/score_phrases.py "$d" --model conch > "$d/phrases.log" 2>&1; p=$?
    python scripts/analyze_slide.py "$d" > "$d/analysis.log" 2>&1; a=$?
    echo "$(date +%T) $(basename "$d") phrases_exit=$p analysis_exit=$a"
}
export -f one
ls -d "$OUT"/TCGA-*/ | xargs -P "$PAR" -I{} bash -c 'one "{}"'
echo "done $(date +%T)"
