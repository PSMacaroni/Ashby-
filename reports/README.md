# Reports

Small copies of evaluation outputs: numbers, confusion matrices, and example image sheets (large PNGs saved as JPEG).
Per-tile prediction tables are not included (too big). They live in `data/results/` on Delta / the laptop.

| folder | what | write-up |
|---|---|---|
| `starc9_lr/` | STARC-9 tissue classifier vs zero-shot on STARC-9 VAL-LARGE, STARC-9 TCGA, CRC-VAL-HE-7K | `docs/dataset_starc9.md` |
| `conch_structures_v1/` | nerve / vessel classifier vs zero-shot on SPIDER colon + skin test slides; artery/vein example sheets | `docs/structures_classifier.md` |
| `split_v1/` | **split models**: tissue8 (NCT + STARC-9) vs zero-shot / STARC-only / NCT-only on 4 test sets; structures model (nerve / vessel / blood) | `docs/split_models.md` |
| `provided_v1/` | both models on the provided datasets NCT-CRC-HE-100K and PathGen | `docs/eval_provided.md` |

Each folder has `metrics.json` (all numbers) and, where available, `results.md` (readable table).
