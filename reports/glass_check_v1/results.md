# Glass / pen-ink check on the full test set (v1)

**Written:** Sun Oct 18 · **Code:** main `7d32348` (texture-based glass / pen-ink check from `preprocess-test` +
the `cutoffs_v1` cut-offs) · **Before:** `reports/cutoffs_v1` (same slides, same code minus the glass / ink check) ·
**Slides:** 56 calibration (`results_v4` → `results_v5`), 64 held-out scored (`results_holdout_v1` → `results_holdout_v2`;
66 folders in the tile counts) · **Labels:** Claude, by eye, on 128 px previews (no pathologist); mucinous and pT from TCGA ·
**Cost:** one Delta job, 1 h on 1 A100 + 16 cores (~1 GPU-hour; no model was run, only CPU work)

**Verdict:** the check cleans the tile quality a lot (about twice as many glass / ink tiles skipped, ~3/4 of the
"blood" calls turn out to be ink) for a small price (7 more real tissue tiles lost out of 367 labeled). The clinical
numbers (depth hint, serosal flag, mucinous flag, growth pattern) are unchanged: they rest on tumor and lymphocyte
tiles, which were already mostly right. This matches the prediction in `docs/preprocessing_test.md`.

## How it was run
Nothing was re-embedded. For each slide: copy the old result folder (embeddings linked), add the texture measures by
reading every tile from the `.svs` again (`scripts/add_texture_measures.py`), re-apply the quality check
(`scripts/recheck_quality.py`), re-analyze (`scripts/analyze_slide.py`), score (`scripts/evaluate_slides.py`,
`scripts/calibrate_cutoffs.py --parts mucin`). Job: `run/rerun_v5.sbatch`, submitted from a checkout of `7d32348` with
`sbatch -A bfqi-delta-gpu -p gpuA100x4 --gpus-per-node=1 --cpus-per-task=16 --mem=62g -t 02:00:00 data/rerun_v5.sbatch`
(the CPU queue was days long). Comparison: `run/compare_v5.py` → `compare.json`, `compare_*.csv`; `run/tissue_v5.py` →
`tissue.json`.

## Clinical results (before → after)

| | calibration | held-out |
|---|---|---|
| depth hint claims true | 15/16 → 14/15 (one slide now "can't judge") | 11/11 → 11/11 |
| exact / shallower / deeper than pT | 7 / 8 / 1 → 7 / 7 / 1 | 5 / 6 / 0 → same |
| serosal flag: pT4 found, false alarms | 0/13, 1/43 → same | 0/21, 0/43 → same |
| mucinous flag (0.015): flagged (mucinous) | 13 (6) → 13 (6) | 15 (7) → 15 (7) |
| mucin vs TCGA mucinous, AUROC | 0.74 → 0.76 | 0.81 → 0.81 |
| growth pattern agrees with the by-eye call | 15/19 → 15/19 | 15/19 → 15/19 |
| infiltrative / pushing / can't judge | 15 / 22 / 19 → 15 / 20 / 21 | 24 / 27 / 13 → same |
| immune at the edge vs MSI-high, AUROC | 0.70 → 0.71 | 0.51 → 0.53 (only 5 MSI-high) |
| "unsure here", median slide | 27.9% → 27.4% | 30.7% → 29.9% |

Slides whose call changed: calibration pattern AA-3875, DY-A1DF; depth AG-A026, DY-A1DF. Held-out pattern A6-2679,
CA-6716, DM-A1DA, EI-7004; depth A6-2679, EI-7004; mucinous flag AG-4015. None changed a score.

## Quality check (before → after)

| | calibration | held-out (66) |
|---|---|---|
| skipped as background (glass / ink) | 28,623 → **59,057** | 43,962 → **76,440** |
| skipped as blood | 7,307 → **1,583** | 9,757 → **2,650** |
| skipped as blur | 18,855 → 6,224 | 21,170 → 6,231 |
| by-eye glass / ink tiles skipped | 21/98 → **53/98** | 5/39 → **14/39** |
| by-eye real tissue wrongly skipped | 13/263 → 15/263 | 4/104 → 9/104 |
| by-eye real blood still caught | 23/42 → 23/42 | 8/12 → 8/12 |

Glass / ink is checked before blood and blur, so most tiles that move to "background" were called blood (red pen ink,
e.g. F4-6808, F5-6813) or blur before. Real tissue newly lost: 3 mucus, 2 lymphocytes (both AA-A00J), 2 muscle.
Glass caught here (36–54%) is lower than the 71% in `docs/preprocessing_test.md` because these labels were made on
128 px previews, where many "glass" tiles are really half-tissue edge tiles.

## Tissue map (before → after)

Average share of a slide's tissue:

| | tumor | stroma | normal mucosa | muscle | necrosis | fat | mucus | lymphocytes |
|---|---|---|---|---|---|---|---|---|
| calibration | 55.9 → 56.9% | 13.6 → 13.5 | 7.8 → 8.0 | 7.4 → 7.2 | **5.6 → 4.9** | 5.0 → 4.9 | 4.1 → 4.0 | 0.6 → 0.6 |
| held-out | 58.0 → 58.4% | 15.6 → 15.7 | 6.4 → 6.5 | 5.6 → 5.5 | 5.7 → 5.5 | 3.7 → 3.7 | 4.7 → 4.5 | 0.3 → 0.3 |

Map label right, by eye (labeled tiles that pass the quality check; "unclear" left out):

| map says | calibration | held-out |
|---|---|---|
| tumor | 32/44 → 32/44 | 14/15 → 14/15 |
| lymphocytes | 32/40 → 32/40 | 11/18 → 9/14 |
| stroma | 24/39 → 24/38 | 13/19 → 13/19 |
| normal mucosa | 15/31 → 15/31 | 10/16 → 10/16 |
| muscle | 13/38 → **13/32** | 10/17 → 9/15 |
| mucus | 12/37 → 11/34 | 5/20 → 3/16 |
| fat | 9/45 → **9/32** | 7/21 → 7/19 |
| necrosis | 3/72 → 3/61 | 1/12 → 1/10 |

The classifier is unchanged; the check only drops junk tiles, so fat / muscle / necrosis get a little cleaner. These
tiles over-sample rare and hard classes on purpose, so they are not the map's overall accuracy (tile-weighted: 72–86%
right above confidence 0.9, `reports/cutoffs_v1`). Necrosis, fat and mucus stay weak: a classifier problem, not a
preprocessing one.

## Limits
Labels by eye on previews, not a pathologist; small samples per class (a few tiles is noise); the glass / ink cut-offs
were chosen on overlapping tiles (`docs/preprocessing_test.md`). Results folders on Delta:
`/work/hdd/bfqi/natmakur/results_v5`, `results_holdout_v2` (not in git).
