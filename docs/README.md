# Docs

Write-ups, in the order the work happened. Numbers behind them are in `reports/` (and `calibration/` for the depth hint).

| doc | what it covers |
|---|---|
| `handoff_small_classifier.md` | first zero-shot test of CONCH and MUSK on NCT-CRC tiles; the plan for a small trained classifier |
| `dataset_starc9.md` | STARC-9, the second training dataset (Stanford, + a TCGA test set) |
| `structures_classifier.md` | nerve / vessel side classifier (SPIDER) for perineural / venous invasion |
| `eval_provided.md` | our models on the hackathon's provided datasets (NCT-CRC-HE-100K, PathGen) |
| `split_models.md` | **current tissue model** `tissue8` (NCT + STARC-9) and the nerve / vessel / blood model; Kather 2016 test |
| `fixes_background_lymph.md` | background (glass) guard and the immune detector |
| `musk_experiment.md` | CONCH vs MUSK vs both: keep CONCH for the map, use MUSK for blood |
| `slide_pipeline.md` | **the whole-slide pipeline**, first run on our 5 TCGA slides, and the MUSK blood check |
| `calibration.md` | depth hint and serosal flag calibrated on 58 TCGA slides with known pT (v1–v3) |
| `summary_example.json` | example `summary.json` (fake infiltrative slide), the format the app reads |
| `img/` | figures used by the docs and the app |
