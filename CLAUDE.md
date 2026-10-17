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
- Stretch: MUSK (`xiangjx/musk`), PathGen-CLIP (`jamessyx/PathGen-CLIP`): follow each model card's usage.
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
Results per slide in `data/results/<slide_id>/`:
- `thumbnail.png`: downsampled slide image; record its downsample factor in summary.json
- `tiles.csv`: one row per tissue tile, columns:
  `row, col, x, y, tile_size_px, qc_ok, qc_reason, p_tumor, p_normal_mucosa, p_muscle, p_fat, p_lymphocytes,
  p_mucus, p_necrosis, p_stroma, label, confidence`
  (x, y = level-0 pixel top-left; label = argmax class name; confidence = max prob)
- `tiles/<row>_<col>.jpg`: optional 128 px tile previews for click-to-zoom
- `summary.json`:
  ```json
  {
    "slide_id": "", "mpp": 0.5, "tile_size_px": 448, "thumbnail_downsample": 64,
    "grid_shape": [0, 0], "n_tissue_tiles": 0,
    "tissue_fractions": {"tumor": 0, "normal_mucosa": 0, "muscle": 0, "fat": 0,
                         "lymphocytes": 0, "mucus": 0, "necrosis": 0, "stroma": 0},
    "tumor": {"fraction_of_tissue": 0, "area_mm2": 0, "centroid_rc": [0, 0]},
    "border": {"n_border_tiles": 0, "solidity": 0, "irregularity": 0, "pattern": "pushing|infiltrative"},
    "invasion": {"border_touch_fraction": {"muscle": 0, "fat": 0, "normal_mucosa": 0, "stroma": 0},
                 "deepest_layer": "mucosa|muscle|fat", "t_stage_hint": "",
                 "direction": {"side": "", "vector_rc": [0, 0]}},
    "immune": {"core_fraction": 0, "margin_fraction": 0,
               "margin_by_side": {"top": 0, "bottom": 0, "left": 0, "right": 0}, "pattern": ""},
    "necrosis": {"fraction_in_tumor": 0, "central": false},
    "high_risk_flags": [], "caveats": [],
    "model": {"name": "CONCH", "prompts_version": ""}
  }
  ```
Update this contract here if it must change.

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
`app/` Streamlit app · `src/` shared code (tiling, models, maps, analysis, summary) · `scripts/` runnable steps
(e.g. `scripts/process_slide.py --slide PATH --out data/results/`) · `notebooks/` experiments · `data/` git-ignored.

## Validation we want for judges
Normal (-11) slides → near-zero tumor; tissue accuracy on ~100 hand-checked tiles (zero-shot vs small classifier);
predicted depth vs pT stage in TCGA reports on 20–50 slides; (stretch) MUSK/PathGen-CLIP votes vs CONCH alone.

## Working style
- Small, testable steps; give a quick way to test each piece (one tile, then one slide).
- When something needs Delta-specific info (account, partition, paths), ask instead of guessing.
- Don't commit or push unless asked.
