# Showcase slides (v1): 5 TCGA slides through the current pipeline

Five colon cancer slides picked to show every part of the model and the app (why each one:
`docs/showcase_slides.md`), run through the whole-slide pipeline at main `17bdafc` on Sun Oct 18.
What changed in the pipeline since the overnight test, how this run went, and what's next: `REPORT.md`.

| Slide | TCGA pT / pN / stage | Diagnosis | Story |
|---|---|---|---|
| TCGA-DM-A282-01Z-00-DX1 | T3 / N0 / IIA | Mucinous adenocarcinoma | Stage II "chemo or not": all wall layers in one section |
| TCGA-DC-6154-01Z-00-DX1 | T4a / N2b / IVA | Adenocarcinoma, NOS | Ragged, high-risk front |
| TCGA-DC-6160-01Z-00-DX1 | T2 / N0 / I | Adenocarcinoma, NOS | Low-risk contrast: smooth front against muscle |
| TCGA-AZ-4614-01Z-00-DX1 | T4a / N1 / IVA | Adenocarcinoma, NOS | Dead core (central necrosis) |
| TCGA-CK-4952-01Z-00-DX1 | T4 / N2 / IIIC | Mucinous adenocarcinoma | Mucin pools + lymphocytes inside the tumor |

## The original slides

The `.svs` files are 0.5–1.7 GB each (4.6 GB in all), too big for GitHub (100 MB per file). They are open-access
TCGA slides, so `slides.csv` records each one's GDC file ID and md5 checksum, and this downloads the exact same files
and checks them:

```bash
python scripts/get_gdc_slides.py --manifest showcase/slides.csv --out <folder for the slides>
```

A copy for the team is also on Delta: `/projects/bfqi/hjeong6/showcase_v1/slides/` (md5s checked against GDC).

## Results (`results/<slide_id>/`)

Per slide, the files the app and the analysis read (format: `CLAUDE.md` → Data contract):
`summary.json`, `tiles.csv`, `tile_analysis.csv`, `phrase_scores.csv`, `thumbnail.png`, `analysis_overview.png`,
and the logs (`analysis.log`, `phrases.log`).

**Not in git** (too big; on Delta in `/projects/bfqi/hjeong6/showcase_v1/<slide_id>/`): `embeddings.npy` (CONCH),
`embeddings_musk.npy` (MUSK), and `tiles/` (one 128 px preview per tile, 15,000–30,000 per slide). The app works
without them; the previews only add the real tile picture when you click a square.

**View in the app:** copy a slide folder into `data/results/` and it shows up in the Slide Box.

## How it was run

`showcase_gpu.sbatch` (all 5 in one GPU job) and `showcase_gpu_one.sbatch` (one job per slide list, used to run
the last two slides in parallel), from a copy of main `17bdafc` on Delta:
`process_slide.py` (tiles, glass / pen-ink / blur / blood check, CONCH tissue model, lymphocyte detector, MUSK blood
check) → `score_phrases.py` → `analyze_slide.py`.
