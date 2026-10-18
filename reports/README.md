# Reports

Small copies of evaluation outputs: numbers, confusion matrices, and example image sheets (large PNGs saved as JPEG).
Per-tile prediction tables are not included (too big). They live in `data/results/` on Delta / the laptop.

| folder | what | write-up |
|---|---|---|
| `starc9_lr/` | STARC-9 tissue classifier vs zero-shot on STARC-9 VAL-LARGE, STARC-9 TCGA, CRC-VAL-HE-7K | `docs/dataset_starc9.md` |
| `conch_structures_v1/` | nerve / vessel classifier vs zero-shot on SPIDER colon + skin test slides; artery/vein example sheets | `docs/structures_classifier.md` |
| `split_v1/` | **split models**: tissue8 (NCT + STARC-9) vs zero-shot / STARC-only / NCT-only on 4 test sets; structures model (nerve / vessel / blood) | `docs/split_models.md` |
| `kather2016/` | tissue models on Kather 2016 (non-normalized); lymphocyte / patient-08 example sheets | `docs/split_models.md` |
| `immune_v1/`, `fixes/` | immune detector v1 tests; background-guard per-tile stats | `docs/fixes_background_lymph.md` |
| `tils_v1/` | lymphocyte scores on **TCGA-TILs** (4,889 raw TCGA colon/rectum patches, 186 patients): immune v1 AUROC 0.831 vs tissue model 0.749 | `docs/fixes_background_lymph.md` |
| `fixes_existing_v1/` | fixes re-tested on untouched data: PathGen captions (immune), NCT-CRC-HE-100K "no harm" (guard) | `docs/fixes_background_lymph.md` |
| `guard_check/` | background guard on our 5 TCGA slides: current mask vs looser mask (saturation > 12); example sheets | `docs/fixes_background_lymph.md` |
| `provided_v1/` | both models on the provided datasets NCT-CRC-HE-100K and PathGen | `docs/eval_provided.md` |
| `musk_v1/` | CONCH vs MUSK vs both for the tissue and structures models, same tiles and recipe | `docs/musk_experiment.md` |
| `blood_check_v1/` | why blood was mapped as necrosis (color, zoom), MUSK blood check, pipeline test on 2 slides | `docs/slide_pipeline.md` |

The depth-hint calibration on 58 TCGA slides lives in `calibration/` (write-up: `docs/calibration.md`).

Each folder has `metrics.json` (all numbers) and, where available, `results.md` (readable table).
