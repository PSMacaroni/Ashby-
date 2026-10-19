# Whole-slide pipeline: linking the models to the slide analysis

**Written:** Sat Oct 17 · **Branch:** `slide-pipeline` (from `main` at eaf2c20) · **Code:** `scripts/process_slide.py`,
`scripts/process_slides_gpu.sbatch` · **First run:** Delta job 22962239, 5 TCGA slides, 7.5 min on one A100 (≈ 0.13 GPU-h)
· **Blood check added:** Sat Oct 17 evening, branch `musk-blood-check` (section "Blood check" below)

## What it does

`process_slide.py` is the missing step from the plan ("tile + embed + classify → tiles.csv"). Per slide:

1. **Tissue mask** on a 64× thumbnail with a fixed cut-off (saturation > `cutoffs.TISSUE_MASK_SAT` = 12). The old
   automatic cut-off (Otsu, 31–70 on our slides) dropped pale submucosa, loose stroma and fat.
2. **Exact grid** of 224 px tiles at 0.5 µm/px (112 µm; tiles touch, no overlap). Keep cells that are ≥ 50% tissue.
3. Per tile: colorful fraction, blur score, **128 px preview** (`tiles/<row>_<col>.jpg`, for the app).
4. **CONCH** embeddings → `embeddings.npy`, and **MUSK** embeddings of the same tiles → `embeddings_musk.npy`.
5. **Tissue model** `models/tissue8_split_v1.joblib` (8 classes; NCT + STARC-9; see `docs/split_models.md`)
   → `p_<class>`, `label` (raw argmax), `confidence`.
6. **Lymphocyte detector** `models/immune_v1.joblib` → `p_immune` (see `docs/fixes_background_lymph.md`).
7. **Blood check** (MUSK) `models/structures_musk_v1.joblib` → `p_blood`, and MUSK's own tissue model
   `models/tissue8_musk_v1.joblib` → `alt_label` (second opinion; see "Blood check" below).
8. **QC** (`src/detectors.quality_check` + `is_blood`): `qc_reason = background` (< 10% colorful and not labeled fat,
   or glass: very flat and pale), `blood` (`p_blood` ≥ `cutoffs.BLOOD_PROB` and not labeled fat; a placeholder) or
   `blur` (< 10% of the slide's own typical sharpness; measured, see `docs/calibration.md` v2).
9. `thumbnail.png` + `summary.json` slide info (incl. project / pT / pN / stage from `slides.csv`).

Then `process_slides_gpu.sbatch` runs `score_phrases.py` and `analyze_slide.py` on each folder, unchanged.

### Small changes to the analysis code (all additive; the 19 tests in `tests/` still pass)
- `src/slides.py`: `tissue_mask(thumbnail, min_sat=None)`; the default behavior is unchanged.
- `src/cutoffs.py`: `TISSUE_MASK_SAT = 12`; the `IMMUNE_LOW` comment now covers `p_immune`.
- `src/tilemap.py`: `p_immune` added to the optional columns.
- `src/features.py`: `immune()` averages `p_immune` when present (else P(lymphocytes)) and says which in `measure`.

## First run: our 5 TCGA slides

| slide | real pT / pN | tiles kept (dropped: glass, blur) | tumor % | deepest layer (analysis) | consistent with pT? |
|---|---|---|---|---|---|
| AA-3875 | T1 N0 | 2,709 (0, 19) | 45% | mucosa, no muscle/fat contact | ✅ yes |
| CM-4746 | T2 N0 | 33,744 (54, 178) | 43% | mucosa, no muscle contact | ❌ (a stroma band separates tumor from muscle at 112 µm scale) |
| AA-3844 | T3 N2 | 2,715 (0, 74) | 95% | reaches muscle | 🟡 partly (≥ T2; no fat in the section) |
| AH-6547 | T3 N0 | 1,180 (0, 0) | 87% | mucosa | ❌ (section shows no deep wall layers) |
| A6-3809 | T4 N0 | 4,273 (13, 130) | 91% | mucosa | ❌ (3 small tumor fragments, no wall layers) |

5 slides is an anecdote, not a statistic. There are no tile labels for these slides, so tile accuracy comes from the
labeled tests (`docs/split_models.md`).

**By eye** (`data/results/<slide>/analysis_overview.png`, not in git): CM-4746's map matches the slide well: tumor on
top, normal mucosa on the right, a submucosal stroma band, and muscularis propria below. The invasive front is found
along the tumor / stroma interface.

## What went wrong, and where (diagnosis of the first run)

We checked every mismatch: thumbnails vs maps, distances on the grid, and the actual tile images at the critical spots
(`data/results/diagnosis/`, not in git). Tile judgments below are ours by eye, **not a pathologist's**.

### Depth hint vs real pT: who caused each mismatch

| slide | real pT | analysis said | cause | evidence |
|---|---|---|---|---|
| **CM-4746** | T2 | "mucosa" | **analysis** | The model correctly mapped tumor glands against / inside the muscle wall (**99 raw tumor tiles touch muscle**, confirmed by eye: glands between smooth-muscle bundles). The analysis erased it: 3×3 smoothing 99 → 9, removing pieces < 0.3 mm → 7, "largest tumor piece only" → **0** |
| AH-6547 | T3 | "mucosa" | **slide** | one solid tumor piece, no muscle wall or fat anywhere in the section (thumbnail) |
| A6-3809 | T4 | "mucosa" | **slide** | three small tumor-only fragments, no wall layers |
| AA-3844 | T3 | "muscle" | **slide** (+ 1 stray tile) | tumor-only pieces; the "muscle" came from a single mislabeled tile |
| AA-3875 | T1 | "mucosa" | right by luck | no muscle wall in the section either |

TCGA's pT describes the **whole surgical specimen**; a diagnostic slide is one section and often doesn't include the
deepest invasion. **3 of 5 slides can't show depth at all.**

**Fix tested** (same maps, new rule): count raw tumor tiles touching a **large muscle region (≥ 2 mm², the muscle wall)**,
and say **"can't be judged"** when the section has no muscle wall or fat. Result: CM-4746 → "reaches the muscle wall
(consistent with ≥ T2)" ✅; the other four → "can't be judged" (honest). That's **1 right, 4 honest, 0 wrong**, vs
**3 wrong, 1 lucky, 1 partly right** now.

### Serosal flag ("possible pT4a"): analysis
It fires on **all 5 slides**. 4 of them (T1, T2, T3, T3) can't have serosal involvement by definition. `outer_edge()`
treats any place where tumor reaches the tissue edge with some stroma nearby as the outer surface, and tumor reaches the
cut edge on most TCGA sections. It needs real serosa / fat evidence and a minimum length, or should be shown as low-trust.

### The model's own labels on these real slides (30 random tiles per label, spread over the 5 slides)

| label | looks right | main errors |
|---|---|---|
| tumor (21,750 tiles) | ~27/30 | a few blood / loose-tissue tiles |
| stroma (5,063) | ~22/30 | smooth muscle, tumor edges (inflamed stroma counted as fine) |
| muscle (10,557) | correct on CM-4746's muscle wall (10,463) | the few "muscle" tiles on tumor-only slides are mostly tumor edges |
| mucus (2,651) | ~15/24 | debris in mucin, near-empty edge tiles |
| **necrosis (1,356)** | **~10–12/30** | **solid red-blood-cell sheets (clots)**, blood-filled granulation tissue (AA-3844's small fragment: 70 of its 106 tiles), a few lymphocyte / muscle tiles |

- **Blood is the main model-side problem:** it inflates "necrosis" and the dead-core / necrosis-rim features.
  - The structures model's blood detector (trained on color-normalized STARC-9, where blood looks purple) finds only 27 blood tiles on CM-4746 and wrongly flags 14 tumor tiles on AA-3844, so it doesn't transfer to raw slides.
  - A simple "bright red" color rule also fails: CM-4746 is stained heavily with eosin, so its normal muscle looks "red".
  - **Fixed later with MUSK:** see "Blood check" below.
- Stain varies a lot between these slides (AA-3844 is faded and pale; CM-4746 is deep red), and tumor still comes out right on both.

### Bottom line
- The most important miss (**T2 called "mucosa"**) was the **analysis**, not CONCH or our classifier. The model saw it.
- **3 of 5 "misses" are the slides themselves** (no deep layers in the section); the analysis should say "can't judge".
- On the model side, the real weakness is **necrosis precision (~35–40%), mostly because of blood**. Tumor labels are good.

## Issues for the analysis owners (found on real slides)

1. **The depth hint has no "submucosa" level.** Tumor whose front touches submucosal stroma is reported as deepest layer
   "mucosa", but invading submucosa already means ≥ T1. Suggest: front touches stroma below the mucosa → "submucosa".
2. **"Not assessable" is missing.** When a section has no muscle or fat at all (AH-6547, A6-3809: tumor-only fragments),
   the hint should say the depth can't be judged, not "mucosa". Many TCGA diagnostic slides are tumor-rich pieces like
   these, which matters for the planned "depth hint vs pT on 20–50 slides" check: pick slides that show the full wall.
3. **"Possible serosal involvement (pT4a)" fires on all 5 slides**, including T1 and T2. Tumor reaching the edge of a
   fragment or the luminal surface gets counted as the outer edge. It needs stricter rules (e.g. require fat or
   serosa-like tissue outside, and a minimum length), or should be shown as low-trust.
4. **Blood is called necrosis on the map:** CM-4746 has a bright red clot (left side) mapped as necrosis. The tissue model has
   no blood class. **Done on `musk-blood-check`:** blood tiles are now skipped as `qc_reason = blood`, using the MUSK
   blood model, because the CONCH one (`structures_split_v1`) misses most of our blood. See "Blood check" below.
5. **App ↔ analysis mismatch:** the app README expects `risk_points` / `risk_level` / `risk_flags` in `tiles.csv`; the
   analysis writes them to `tile_analysis.csv`.
6. ~~`cutoffs.BLUR_MIN` placeholder~~: replaced by a slide-relative rule + glass check (v2, `docs/calibration.md`).

## Blood check (MUSK)

**Why the map said "necrosis":** `tissue8` has no blood class, so it must pick one of its 8 classes. It calls even
STARC-9's own blood tiles necrosis 78% of the time, so no color change can fix the map by itself. The pipeline needed
a separate blood step ("identify and ignore", as planned in `docs/dataset_starc9.md`).

**Why the CONCH blood model missed our blood:** it learned blood only from STARC-9 tiles, which differ from ours in
two ways:
- **Color:** STARC-9 blood is purple, which is what per-tile Macenko normalization does to red blood; ours is bright red.
- **Zoom:** STARC-9 tiles are 128 px (64 µm) and CONCH's preprocessing enlarges them to 224 px, so the model saw cells
  1.75× bigger than in our 224 px / 112 µm tiles. This affects every model trained on STARC-9 tiles, `tissue8`
  included (not measured there).

**Test set:** 2,237 tiles from the 5 slides:
- 376 bright-red tiles from CM-4746's clot (40 of them checked by eye: all blood)
- 200 CM-4746 muscle tiles (deep red stain)
- 411 other tiles that `tissue8` calls necrosis
- 1,250 random tiles

Share of tiles called blood (tiles as scanned unless noted):

| blood check | clot | CM-4746 muscle | other "necrosis" | random |
|---|---|---|---|---|
| CONCH `structures_split_v1` | 42% | 0% | 9% | 0.7% |
| same, tiles color-normalized per tile + STARC-9 zoom | 85% | 2% | 13% | 2.3% |
| zero-shot CONCH, "blood" prompt added | 99% | 3.5% | 31% | 5.4% |
| **MUSK `structures_musk_v1` (used now, p ≥ 0.5, not fat)** | **98%** | **0%** | **7%** | **0.5%** |
| MUSK, p ≥ 0.9 | 91% | 0% | 2% | 0% |

The MUSK model is the structures model from the MUSK experiment (`musk-experiments`, run `musk_v1`). There it also
called only 0.1% of STARC-9 TCGA fat tiles "blood" (CONCH: 16%).

**`alt_label` (MUSK's tissue8):** a second opinion only. On our random tiles it agrees with CONCH's `tissue8` 82% of
the time; where they disagree, CONCH looked right more often by eye (about 15–19 of 40 vs 6), and MUSK calls 4× more
tiles necrosis (10.9% vs 2.8%). Disagreements feed the "unsure here" overlay.

**One overrule (`cutoffs.ALT_OVERRULES`, `src/tilemap.py`):** where CONCH says necrosis and MUSK says muscle, the
analysis uses muscle (tiles.csv keeps the models' numbers; the tiles stay marked "unsure"). On CM-4746 these are
eosin-heavy smooth muscle with black specks: 16 of 20 checked by eye were plainly muscle, none necrosis. On our other
4 slides MUSK never called a CONCH-necrosis tile muscle, and on labeled tests MUSK calls real necrosis muscle 0–6.8%
of the time. Effect: 140 tiles on CM-4746; the smoothed map's necrosis drops from about 100 to 55 tiles; no other
finding changes. Requiring MUSK to agree on *all* necrosis was rejected: MUSK finds only 66% of SPIDER's necrosis.

**Limits:** the only blood "truth" is one clot on one slide, judged by eye (not a pathologist). Blood on AA-3844 and
AA-3875 looks magenta and wasn't measured. `cutoffs.BLOOD_PROB` stays a placeholder until the pathologist labels ~50
blood / not-blood tiles across the slides.

**Cost:** the pipeline now also runs MUSK on every tile. The investigation used 4 short GPU jobs (≈ 0.19 GPU-h). Its
scripts and results are on Delta in `/work/hdd/bfqi/natmakur/blood_check/`.

## Run it

```bash
# on Delta, from the repo folder (slides + slides.csv in one folder)
sbatch scripts/process_slides_gpu.sbatch /projects/bfqi/$USER/tcga_slides
# then copy a slide's folder to the laptop for the app
scp -r delta:<repo>/data/results/<slide_id> data/results/
```
