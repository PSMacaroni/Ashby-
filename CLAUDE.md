# CLAUDE.md — Ashby Hackathon 2026 project context

## Who we are
Beginner team (UIUC undergrads) at the 2026 Ashby Prize Hackathon (NCSA / Cancer Center at Illinois).
Hack days Sat–Sun Oct 17–18; submission (abstract + ≤20-slide deck) due Wed Oct 21.
We understand ML concepts but are new to pathology, Delta, and these models. **Explain things simply**,
prefer simple working code over clever code, and tell us when something is risky or uncertain.

## Challenge
Use multimodal (image + text) pathology foundation models for a clinical application that makes pathology
more precise, reproducible, or interpretable. Judged on: innovation, development progress, clinical impact,
communication.

## Our project: Colon Tumor Front Analyzer
A pathologist views a colon cancer H&E whole-slide image as a color-coded tissue map (hover a spot →
probabilities), plus a summary panel describing the tumor and how it is spreading:
- tissue map: tumor, normal mucosa, smooth muscle, fat, lymphocytes (immune cells), mucus, necrosis, stroma
- tumor %; border shape (smooth/"pushing" vs ragged/"infiltrative")
- depth of invasion + direction (does tumor border touch muscle / fat? where?) → rough T-stage hint
- immune response: lymphocytes at tumor center vs edge, by side (a stain-free approximation of the Immunoscore idea)
- dead core (necrosis inside tumor)
- high-risk features relevant to the stage II colon cancer "chemo or not" decision
Clinical framing: decision support that flags features that are tedious and inconsistent to assess by eye. Not a diagnosis.

## Honest limits (never overclaim in code comments, UI text, or slides)
- H&E cannot distinguish CD4 vs CD8 T cells; we only measure overall lymphocytes.
- One slide is a 2D snapshot: "direction" = where it is invading in this slice, not a prediction of the future.
- T stage needs more than one slide; ours is a "consistent with" hint.
- Tumor budding detection is experimental at tile level.
- No mutation prediction, no treatment recommendations, no patient data collection.

## Environment (important)
- **Delta** (NCSA GPU cluster, used via Open OnDemand Jupyter in the browser, or SSH + `sbatch`) runs everything
  heavy: reading slides, running models. See "Delta specifics" below for accounts, paths, and data.
- **Laptop / VS Code**: write code, run the Streamlit app on small result files downloaded from Delta.
- Code moves via GitHub: write on laptop → push → `git pull` on Delta → run.
- Never commit: `.env`, tokens, slides, model weights, big arrays, anything in `data/`.
  Exception: our own small trained classifier heads (scikit-learn, a few KB) live in `models/` with a model card
  (`models/README.md`); compact evaluation outputs live in `reports/`.
- Delta home quota is small: set `HF_HOME` to a project/work directory before loading models.
- Code must run headless on Delta (no GUI), use GPU if available, fall back to CPU, show progress with tqdm,
  and save results to files (sessions end and memory is lost).

## Models (Hugging Face, gated, non-commercial research use)
- Main: CONCH (`MahmoodLab/CONCH`), install `pip install git+https://github.com/Mahmoodlab/CONCH.git`
  ```python
  from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
  model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=HF_TOKEN)
  # or a local checkpoint path instead of the hf_hub string if organizers preloaded weights
  img_emb = model.encode_image(preprocess(img).unsqueeze(0), proj_contrast=True, normalize=True)
  txt_emb = model.encode_text(tokenize(texts=prompts, tokenizer=get_tokenizer()))
  probs = (img_emb @ txt_emb.T * model.logit_scale.exp()).softmax(-1)
  ```
- MUSK (`xiangjx/musk`): `load_musk()` in `src/models.py` (same interface as CONCH; own `MuskTokenizer` because
  transformers 5 can't load its .spm file). Zero-shot on NCT-CRC (400 tiles): CONCH 90.0%, MUSK 90.5%, average 92.0%;
  MUSK better at tumor/stroma and far better calibrated. Details: `docs/handoff_small_classifier.md`.
- Stretch: PathGen-CLIP (`jamessyx/PathGen-CLIP`): follow the model card's usage.
- Approach: zero-shot with prompt ensembles (several phrasings per class, averaged). We do NOT train big models.
  If a class is inaccurate, train a small logistic regression on CONCH embeddings (labels from PathGen captions
  by keyword, or a small hand-labeled set).

## Data
- TCGA colon (COAD/READ) diagnostic slides (`.svs`, read with openslide). Barcode e.g.
  `TCGA-AA-3844-01Z-00-DX1`: patient = `TCGA-AA-3844`; sample code `01` = primary tumor, `11` = normal tissue.
- PathGen: JSON records {wsi_id, position [x, y], caption, file_id}; 672×672 patches from TCGA slides with AI-written captions.
- Quilt-1M, PathMMU also available (education images / multiple-choice questions).
- Always split train/test by PATIENT, never by tile.

## Pipeline
Delta: (1) tile slide at ~20x (~0.5 µm/px), skip background via tissue mask on thumbnail → (2) QC: blur
(Laplacian variance), pen marks → (3) CONCH embeddings per tile (batched) → (4) zero-shot tissue probabilities
→ (5) grid maps + light smoothing → (6) analysis with numpy/scipy/scikit-image (connected components, border
tiles, solidity/irregularity, border-touch fractions, core vs margin band by side, necrosis centrality,
rule-based flags) → (7) save results.
Laptop: (8) Streamlit + Plotly app reads results; summary text from templates, optional Lumen LLM rewrite
(pathologist + patient versions) that may only restate numbers from summary.json.

## Data contract (all code must follow this)
Tiles are **224 px at 0.5 µm/px = 112 µm** (same as NCT-CRC; one tile = 0.0125 mm²).
Results per slide in `data/results/<slide_id>/`.

Made on Delta by the whole-slide pipeline (GPU):
- `thumbnail.png`: downsampled slide image; its downsample factor (level-0 px per thumbnail px) goes in summary.json
- `tiles.csv`: one row per tissue tile, columns:
  `row, col, x, y, tile_size_px, qc_ok, qc_reason, p_tumor, p_normal_mucosa, p_muscle, p_fat, p_lymphocytes,
  p_mucus, p_necrosis, p_stroma, label, confidence`
  (x, y = level-0 pixel top-left; label = argmax class name; confidence = max prob)
  Optional: `p_nerve`, `p_artery` (trained yes/no detectors, 0-1), `alt_label` (a second model's top class,
  e.g. MUSK's when the p_* columns are CONCH's; kept for reference, the analysis doesn't use it: MUSK is used
  only for the blood check, "unsure here" = low confidence only), `p_immune` (lymphocyte detector,
  0-1; `features.immune()` uses it instead of P(lymphocytes) when present), `p_blood` (MUSK blood check, 0-1).
  `scripts/process_slide.py` writes `p_immune`, `p_blood`, `alt_label` (MUSK's tissue8), and also `colorful`
  (share of stained pixels), `blur` (Laplacian variance), `mask_coverage`; `qc_reason` is `background`
  (src/detectors.quality_check: colorless and not fat, or glass), `blood` (src/detectors.is_blood: p_blood >=
  cutoffs.BLOOD_PROB, not labeled fat) or `blur` (src/detectors.quality_check: < cutoffs.BLUR_REL x the
  slide's typical sharpness). `scripts/recheck_quality.py` re-applies these rules to saved tiles (CPU).
- `embeddings.npy`: float32 [n_tiles, dim], rows in the same order as tiles.csv (CONCH 512, MUSK 1024)
- `embeddings_musk.npy`: optional, MUSK embeddings [n_tiles, 1024] when embeddings.npy is CONCH's
- `tiles/<row>_<col>.jpg`: optional 128 px tile previews for click-to-zoom
- `structures.csv` (optional, not made by `process_slide.py` yet; `src/structures.py`): nerve / vessel side
  classifier, NOT one of the 8 map classes: `row, col, p_nerve, p_vessel, p_other, structure, p_artery`
  (p_artery = zero-shot artery-vs-vein, experimental and unvalidated). For perineural / venous invasion flags.
- `summary.json` slide info: `slide_id, mpp (0.5), tile_size_px (224), thumbnail_downsample, grid_shape,
  model {name, prompts_version}, embeddings_model`, optional `project, pT, pN, stage, qc, blood_check, alt_model`

Made on the laptop or Delta (CPU only):
- `phrase_scores.csv`: `row, col` + one column per key in `src/phrases.py` (`scripts/score_phrases.py`;
  needs embeddings.npy + the text side of the same model)
- `scripts/analyze_slide.py` adds to `summary.json`: `tissue_fractions, tumor, border` (growth pattern + front
  arcs), `satellites, invasion` (border_touch_fraction, deepest_layer, t_stage_hint, direction, heading_into),
  `fat, spread, stroma, necrosis, mucin, immune, lymph_nodes, outer_edge, uncertainty, flags, risk,
  high_risk_flags, caveats`. Full example: `docs/summary_example.json` (the fake infiltrative slide).
  Keys from the old contract are kept (e.g. `border.irregularity` = front roughness in µm).
- `tile_analysis.csv`: per tissue tile `row, col, label` (smoothed), `unsure, risk_points, zone, border,
  dist_to_front_um, in_main_tumor, on_front`: for the app's overlays.
Update this contract here if it must change.

## Trained vs computed (who makes what)
- **Trained** (Models teammate): the 8-class classifier on CONCH/MUSK embeddings (Kather tiles), and yes/no
  nerve and artery detectors (pathologist-labeled tiles) -> `p_*` and `p_nerve`/`p_artery` in tiles.csv.
- **Text match, no training**: search bar and flag phrases (`src/phrases.py`).
- **Computed, no model** (`src/front.py`, `src/features.py`): tumor mask, invasive front, zones, growth
  pattern, direction, depth hint, fat, stroma ratio, necrosis + rim, mucin, immune zones, lymph nodes,
  deposits, outer edge, uncertainty, risk map (`src/risk.py`).
- **Model + geometry** (`src/flags.py`): perineural, venous, lymphatic, budding hotspot, stroma maturity,
  leftover adenoma. "Look here" hints only; "strong" ones feed the risk map.
- **Not doing**: nucleus model (budding counts, neutrophils, eosinophils), MSI predictor.
- Every analysis number is in `src/cutoffs.py`, marked doc / literature / PLACEHOLDER. PLACEHOLDER cutoffs
  need the pathologist before results are shown. The risk table's section-7 rows are placeholders (that
  section wasn't in the PDF). Test on fake slides: `python scripts/make_fake_slide.py`, `pytest -q`.

## Delta specifics
- Accounts: `bfqi-delta-gpu` (**10 GPU-hours for the whole team**; spend carefully) and
  `bfqi-delta-cpu` (~9,500 CPU-hours). Prefer batch jobs (`sbatch`) that end when done over
  interactive sessions that bill while idle; use CPU jobs for anything that doesn't need a GPU.
- GPU partition: `gpuA100x4` (hackathon reservation on the account, picked up automatically;
  Sat 07:30 → Sun 07:59 and Sun 08:00 → 18:00). CPU partition: `cpu`. All nodes are RH9.6.
- Python: `source scripts/delta_env.sh` (loads `pytorch-conda/2.12` = Python 3.12, torch 2.12 cu130,
  transformers 5.x, and unsets `PYTHONNOUSERSITE` so `pip install --user` packages are visible).
- Shared datasets (`DATA_DIR`): `/projects/bfqi/shared/datasets`
  - `NCT-CRC-HE-100K/data-samples/<ADI|BACK|DEB|LYM|MUC|MUS|NORM|STR|TUM>/*.tif`: 100k labeled
    224 px tiles at 0.5 µm/px (no patient IDs). Mapping to our classes in `src/datasets.py`.
  - `pathgen_dataset/pathgen_2k_output/<file_id>_<x>_<y>.png|.txt`: ~2,100 PathGen patches + captions
    from 10 TCGA slides (only 3 colorectal); records in `pathgen_dataset/PathGen-2k_samples.json`.
  - `pathmmu/`: PathMMU question set + several small image sets.
  - No TCGA whole slides and no model weights are preloaded.
- Our TCGA slides (downloaded from GDC, open access): `/work/hdd/bfqi/natmakur/tcga_slides/`
  with `slides.csv` (file_id, md5, project, pT, pN, stage). AA-3875 T1N0, CM-4746 T2N0, AH-6547 T3N0
  (READ), AA-3844 T3N2, A6-3809 T4N0 (stage IIB). AA-3844/CM-4746/AH-6547 have PathGen patches.
- Model weights: downloaded from Hugging Face into `HF_HOME=/projects/bfqi/natmakur/hf_cache` (private).
- `/projects/bfqi/<user>` and `/work/hdd/bfqi/<user>` are readable by the whole hackathon group:
  keep `.env`/tokens in the home folder (`~`), which is private. Don't read other teams' folders.
- Login nodes: light work only (git, pip, file listing). Long commands there can get cut off.

## Repo layout
`app/` Streamlit app · `src/` shared code · `scripts/` the slide pipeline · `scripts/models/` how the classifiers were
built and tested · `models/` trained heads + model card · `reports/` evaluation outputs (one folder per run) ·
`calibration/` depth-hint calibration · `docs/` write-ups (index: `docs/README.md`) · `tests/` pytest ·
`notebooks/` experiments · `data/` git-ignored.
- Models: `src/models.py` (CONCH, MUSK, zero-shot, `encode_texts`), `src/prompts.py`, `src/datasets.py`, `src/slides.py`,
  `src/classifier.py` (tissue8), `src/detectors.py` (glass / blur check, immune, blood), `src/structures.py`
- Analysis (CPU): `src/tilemap.py` (tiles.csv -> grids) -> `src/front.py` -> `src/features.py`,
  `src/flags.py` (+ `src/phrases.py`), `src/risk.py`; `src/analysis.py` runs them all; `src/cutoffs.py`; `src/viz.py`
- Scripts (`scripts/README.md`): `process_slide.py` (whole-slide pipeline: fat-aware tissue mask + 224 px tiles + CONCH
  tissue model `models/tissue8_split_v1.joblib` + lymphocyte detector `models/immune_v1.joblib` + MUSK blood check
  `models/structures_musk_v1.joblib` + second opinion `models/tissue8_musk_v1.joblib` + glass / blur check ->
  tiles.csv, embeddings.npy, embeddings_musk.npy, tiles/, thumbnail.png, summary.json), `process_slides_gpu.sbatch`
  (it + phrases + analysis), `score_phrases.py`, `analyze_slide.py`, `make_fake_slide.py`, `check_setup.py`,
  `get_gdc_slides.py`, `calibrate_analysis.py`. Why MUSK for blood: `docs/slide_pipeline.md`.

## Validation we want for judges
Normal (-11) slides → near-zero tumor; tissue accuracy on ~100 hand-checked tiles (zero-shot vs small classifier);
predicted depth vs pT stage in TCGA reports on 20–50 slides; (stretch) MUSK/PathGen-CLIP votes vs CONCH alone.

## Working style
- Small, testable steps; give a quick way to test each piece (one tile, then one slide).
- When something needs Delta-specific info (account, partition, paths), ask instead of guessing.
- Don't commit or push unless asked.
