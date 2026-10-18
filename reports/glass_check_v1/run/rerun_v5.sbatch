#!/bin/bash
# Full re-test with main 7d32348 (glass / pen-ink texture check + cutoffs_v1). CPU only, no models.
# Copies results_v4 / results_holdout_v1 (baseline stays untouched) -> results_v5 / results_holdout_v2,
# adds texture measures (re-reads tiles from the .svs), re-applies QC, re-analyzes, evaluates.
#SBATCH --job-name=rerun_v5
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200g
#SBATCH --time=03:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
R=/work/hdd/bfqi/natmakur
SLIDES="/projects/bfqi/hjeong6/tcga_calib $R/tcga_slides $R/tcga_holdout"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

copy_set() {   # small files copied, big files / previews linked (read-only use)
    src=$1; dst=$2; mkdir -p "$dst"
    for d in "$src"/TCGA-*/; do
        n=$(basename "$d"); o=$dst/$n
        [ -f "$d/summary.json" ] || continue
        [ -f "$o/tiles.csv" ] && continue                      # re-run safe
        mkdir -p "$o"
        cp "$d/tiles.csv" "$d/summary.json" "$d/phrase_scores.csv" "$d/thumbnail.png" "$o/"
        for f in embeddings.npy embeddings_musk.npy tiles; do
            [ -e "$d/$f" ] && ln -sfn "$(readlink -f "$d/$f")" "$o/$f"
        done
    done
    echo "$(date +%T) copied $(ls -d "$dst"/TCGA-*/ | wc -l) folders to $dst"
}

analyze_one() { python scripts/analyze_slide.py "$1" > "$1/analysis.log" 2>&1; echo "$(date +%T) $(basename "$1") analysis_exit=$?"; }
export -f analyze_one

run_set() {
    src=$1; dst=$2; ev=$3
    copy_set "$src" "$dst"
    python scripts/add_texture_measures.py "$dst"/TCGA-*/ --slide-dirs $SLIDES --workers $SLURM_CPUS_PER_TASK
    python scripts/recheck_quality.py "$dst"/TCGA-*/
    ls -d "$dst"/TCGA-*/ | xargs -P 6 -I{} bash -c 'analyze_one "{}"'
    python scripts/evaluate_slides.py --results "$dst" --labels calibration/clinical_labels.csv --out "$ev"
}

echo "start $(date +%T), code $(git log --oneline -1)"
run_set $R/results_v4 $R/results_v5 $R/eval_v5_calib
run_set $R/results_holdout_v1 $R/results_holdout_v2 $R/eval_v5_holdout
python scripts/calibrate_cutoffs.py --parts mucin --results-calib $R/results_v5 --results-holdout $R/results_holdout_v2
echo "done $(date +%T)"
