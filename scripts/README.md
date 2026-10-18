# Scripts

Run everything from the project folder (`python scripts/...`, `sbatch scripts/...`). On Delta, first
`source scripts/delta_env.sh`. `.sbatch` files are Delta batch jobs; their headers say which account (CPU or GPU hours).

## The slide pipeline (`scripts/`)

| script | what it does | where |
|---|---|---|
| `check_setup.py` | checks Python packages, GPU, tokens, folders | laptop / Delta |
| `delta_env.sh` | loads Delta's PyTorch and our `pip --user` packages | Delta |
| `process_slide.py` | one `.svs` slide → tiles, CONCH + MUSK embeddings, tissue map, blood / glass / blur check (`docs/slide_pipeline.md`) | Delta, GPU |
| `process_slides_gpu.sbatch` | `process_slide.py` + `score_phrases.py` + `analyze_slide.py` for every slide in a folder | Delta, GPU job |
| `score_phrases.py` | text-phrase scores for the "look here" flags → `phrase_scores.csv` | laptop / Delta, CPU |
| `analyze_slide.py` | tumor front, measurements, flags, risk map → `summary.json`, `tile_analysis.csv` | laptop / Delta, CPU |
| `make_fake_slide.py` | fake slide folders for building and testing the analysis and the app | laptop |
| `get_gdc_slides.py` | download open-access TCGA slides by manifest (`calibration/manifest_v1.csv`) | Delta |
| `calibrate_analysis.py` | calibrate the depth hint / serosal flag against real pT (`docs/calibration.md`) | laptop / Delta, CPU |

## How the models were built and tested (`scripts/models/`)

Finished experiments, kept so every number in `docs/` and `reports/` can be reproduced.

| topic | scripts | write-up |
|---|---|---|
| first zero-shot test (CONCH, MUSK) | `first_look.py`, `first_look_{cpu,gpu}.sbatch` | `docs/handoff_small_classifier.md` |
| training data | `get_starc9.py`, `get_spider.py` + `get_spider_cpu.sbatch`, `get_public_tests.py`, `prepare_compare_data_cpu.sbatch` | `docs/dataset_starc9.md`, `docs/structures_classifier.md` |
| embeddings | `embed_tiles.py`, `embed_many.py`, `embed_{cpu,gpu,spider_gpu,inflam_gpu,tils_gpu,compare_gpu}.sbatch` | |
| tissue classifier | `train_classifier.py` (STARC-9), `train_split_models.py` (tissue8 + structures) | `docs/split_models.md` |
| nerve / vessel classifier | `train_structures.py` | `docs/structures_classifier.md` |
| immune detector, background guard | `train_immune.py`, `check_background_guard.py` + `guard_check_gpu.sbatch`, `eval_tils.py`, `eval_fixes_existing.py` | `docs/fixes_background_lymph.md` |
| tests on the provided datasets | `eval_provided.py` + `eval_provided_gpu.sbatch` | `docs/eval_provided.md` |
| CONCH vs MUSK | `compare_backbones.py` + `compare_{cpu,gpu}.sbatch` | `docs/musk_experiment.md` |
| blood vs necrosis (color, zoom, MUSK) | `blood_check.py`, `blood_check2.py`, `blood_check3.py` | `reports/blood_check_v1/results.md` |
