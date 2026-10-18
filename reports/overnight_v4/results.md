# Overnight test (v4): the current pipeline on all 58 calibration slides, + a held-out set

**Run:** night of Sat Oct 17 → Sun Oct 18 · **Code:** `main` at f06a74f + `scripts/process_slide.py` (many slides per
run, models loaded once), `scripts/embed_slides_gpu.sbatch`, `scripts/analyze_slides_cpu.sbatch`,
`scripts/evaluate_slides.py`, `scripts/make_label_sheets.py`, `scripts/pick_gdc_slides.py`
**Compute:** 58 slides: 1 h 50 min on one A100 (585,954 tiles); analysis 5 min on CPU. **Nothing was tuned:** every
number uses the settings in `src/cutoffs.py` as they are on `main`.
**Tile labels are Claude's, by eye, not a pathologist's** (the project has no pathologist).

## What's new compared with the v3 calibration

The v3 numbers (`docs/calibration.md`) came from v2 tissue maps made **before** the MUSK changes. This run re-made all
58 maps with the current pipeline: fat-aware mask, glass + blur check, **MUSK blood check** (`qc_reason = blood`),
**MUSK second opinion** (`alt_label`) with the necrosis → muscle overrule, and the thick-wall depth defaults
(`WALL_MIN_WIDTH_UM = 600`, depth judged only when the wall is in the section).

`--ablation` re-analyzes the **same tiles without the MUSK parts** (blood tiles kept, no `alt_label`), so the MUSK
effect is measured on identical maps.

## 1. Depth hint and serosal flag (58 slides, real pT from TCGA)

| | judged | exact | shallower (claim still true) | **deeper than pT** | claim true | T3/T4 reaching "≥ pT3" | serosal flag: pT4 found / false alarms |
|---|---|---|---|---|---|---|---|
| v3b, v2 maps (docs/calibration.md) | 17 | 7 | 9 | **1** | 16/17 | 2/27 | 0/13 · 1/45 |
| **v4, current pipeline** | **16** | **7** | **8** | **1** | **15/16 (94%)** | **2/27** | **0/13 · 1/45** |
| v4 maps without MUSK | 17 | 6 | 9 | 2 | 15/17 | 2/27 | 0/13 · 1/45 |

Rows = real pT, columns = what the hint said (v4): can't judge / 1 = wall not reached / 2 = reaches the wall / 3 = beyond it

| | can't judge | 1 | 2 | 3 |
|---|---|---|---|---|
| T1/Tis | 14 | 3 | 0 | 1 ❌ |
| T2 | 10 | 1 | 2 | 0 |
| T3 | 11 | 1 | 2 | 0 |
| T4 | 7 | 0 | 4 | 2 |

- **Basically unchanged.** The depth hint depends on tumor / muscle / fat, which the MUSK changes barely touch.
- MUSK changed 2 slides: **F4-6808** (pT1) went from "reaches the wall" (an over-call) to "wall not reached" (correct);
  **G4-6321** (pT2) went from "wall not reached" to "can't judge". So MUSK removed one over-call.
- The one remaining over-call is **DC-5337** (pT1 → "beyond the wall"), the slide flagged since v1 as a possible
  label issue.
- Still only **16 of 58** sections can be judged (the rest show no thick muscle wall), and the serosal flag still
  finds no pT4. Both are limits of single TCGA sections, as in v3.

## 2. Quality check and map numbers

| | tiles | share of tiles read |
|---|---|---|
| read | 585,954 | |
| kept as tissue | 530,986 | 90.6% |
| skipped: background (glass) | 28,804 | 4.9% |
| skipped: blood (MUSK) | 7,307 | 1.2% |
| skipped: blur | 18,857 | 3.2% |

- **"Unsure here" now covers 25% of the tissue (median slide 24%, worst 56%), up from 3% without MUSK.** Every tile
  where CONCH and MUSK disagree is marked unsure. In the app that overlay would cover a quarter of every slide, so it
  probably needs a stricter rule (e.g. only disagreements that are also low-confidence). **Changed afterwards:
  see section 8.**
- Necrosis share of tissue: 6.6% (6.7% without MUSK). The blood check removes little necrosis on these slides.

## 3. How often the map labels are right (my labels, 528 tiles)

`scripts/make_label_sheets.py` took 48 tiles per set, at most 4 per slide, from all 58 slides, and **shuffled them
together** into 11 sheets (`label_sheets/`), so I labeled without knowing what the model said. Labels:
`labels_claude_v4.csv` (with each tile's colorful share, blur and mask coverage). "Glass" = the preview is mostly
empty glass, ink or an artifact. "Unclear" = I couldn't tell at 128 px.

| set (what the model said) | n | unclear | **glass** | agree (of judged) | agree (tissue tiles only) | main disagreements |
|---|---|---|---|---|---|---|
| tumor | 48 | 8 | 1 | **80%** | 82% | normal mucosa 3, stroma 2 |
| lymphocytes | 48 | 8 | 0 | **80%** | 80% | tumor 4 |
| stroma | 48 | 13 | 1 | 66% | 68% | muscle 4, tumor 4 |
| normal mucosa | 48 | 17 | 2 | 48% | 52% | **tumor 11** |
| mucus | 48 | 13 | 10 | 34% | 48% | glass 10, stroma 6 |
| muscle | 48 | 9 | 10 | 33% | 45% | **stroma 12**, glass 10 |
| fat | 48 | 3 | **27** | 20% | 50% | **glass 27** |
| necrosis | 48 | 18 | 13 | **10%** | 18% | glass 13, tumor 7 |
| skipped as blood | 48 | 12 | 11 | 44% blood | 64% blood | glass 11, stroma 6 |
| necrosis → muscle overrule | 48 | 8 | 13 | 16 muscle, **0 necrosis** | 59% muscle | glass 13, blood 6 |
| 0.3 ≤ p_blood < 0.7 (border) | 48 | 10 | 13 | 0.3–0.5: 6 blood / 22 not · 0.5–0.7: 7 blood / 7 not | | |

**Main finding: glass gets through the quality check and is labeled as tissue.** Of the tiles I called glass, the
model had labeled 28 necrosis, 27 fat, 10 muscle and 10 mucus. They pass because:
- the background guard never removes a tile the model calls **fat** (on purpose: fat is pale too);
- glass on these slides is often **tinted** (median colorful share 0.21) and not perfectly flat, so the
  "colorless" and "glass" rules miss it;
- most are **edge tiles** (tissue in one corner) or ink / scanner artifacts (solid red, green, blue tiles).

A simple color cut-off can't fix it: real fat (my 12 fat tiles) has a median colorful share of 0.16, below glass.
Cut-offs that remove most glass also remove nearly all real fat:

| colorful share below | glass removed | tissue removed | real fat removed |
|---|---|---|---|
| 0.2 | 49% | 4% | 7 / 12 |
| 0.3 | 67% | 8% | 10 / 12 |
| 0.4 | 82% | 13% | 12 / 12 |

This inflates **fat, necrosis and mucus** on every slide (and "dead core" / necrosis features). Tumor and lymphocyte
labels, the ones most of the analysis is built on, hold up (80%).

Other points:
- **Necrosis is still the weakest class (10%)**, now mostly because of glass, not blood.
- **The blood check:** 64% of the skipped tissue tiles were blood; the rest were stroma (6) and tumor (2). In the
  border zone (0.3–0.7) blood and not-blood are mixed, as expected.
- **The overrule is right to drop necrosis:** none of the 40 judged tiles was necrosis.
- **Muscle vs stroma** and **normal mucosa vs tumor** are the main tissue-on-tissue confusions (as in the labeled tests).

**Limits of these labels:** one labeler, not a pathologist, 128 px previews (very thin fat membranes can look like
empty glass at this size), 8–18 "unclear" per set.

## 4. TCGA's own labels (descriptive checks, no tuning)

TCGA labels for every COAD/READ case: `calibration/clinical_labels.csv` (`scripts/pick_gdc_slides.py`; MSI scores from
cBioPortal, TCGA PanCancer Atlas; MSI-high = MANTIS > 0.4).

| check | slides | result |
|---|---|---|
| immune score at the tumor margin, MSI-high vs MSS | 12 vs 45 | **AUROC 0.76** (core 0.60): more lymphocytes at the front in MSI-high tumors, as expected |
| `mucin.fraction_in_tumor` vs TCGA "mucinous adenocarcinoma" | 9 vs 49 | AUROC 0.54: **the measure misses it** (7 of 9 mucinous slides say "No mucin inside the tumor") |
| plain mucus share of all tissue vs the same label | 9 vs 49 | **AUROC 0.92** (median 7.6% vs 0.2%) |

The map sees mucin; the "inside the tumor" measure loses it, probably because mucin pools aren't part of the tumor
mask (it is built from tumor tiles only, and only holes up to `SMALL_HOLE_TILES` are filled). A definition change, not
a cut-off. Not changed here.

## 5. Held-out slides (new patients, never used to design anything)

`scripts/pick_gdc_slides.py` picked 80 open-access TCGA COAD/READ diagnostic slides from patients not in the
calibration set (untreated, one per patient; T stages rotated so a cut-short run stays balanced). The GPU job hit its
time cap (2 h 45 min, keeping the night at ~4.6 GPU-hours) after **67 slides** (T1 2, T2 23, T3 21, T4 21; 869,301
tiles). Results: `holdout/`.

| | judged | exact | shallower | **deeper than pT** | claim true | T3/T4 reaching "≥ pT3" | serosal flag: pT4 found / false alarms |
|---|---|---|---|---|---|---|---|
| **held-out, current pipeline** | **11 / 67** | 5 | 6 | **0** | **11/11** | 3/42 | 0/21 · 0/46 |
| held-out maps without MUSK | 13 / 67 | 5 | 7 | 1 | 12/13 | 3/42 | 1/21 · 0/46 |

- **On slides the rules never saw, the depth hint made no over-call** (11 judged). The 600 µm thick-wall rule was
  picked on the calibration slides, so this is the first independent check of it. 11 slides is still small.
- Coverage is the limit again: only 11 of 67 sections (16%) show a thick muscle wall.
- Map labels by eye (216 tiles, `holdout/labels_claude_holdout.csv`, 24 per set): tumor **88%** (94% of tissue
  tiles), stroma 68%, normal mucosa 65%, lymphocytes 61% (79% of tissue tiles), muscle 59%, fat 33% (**12 of 21
  judged were glass**), mucus 25% (10 of 20 glass), necrosis **8%**, skipped-as-blood 47% blood. Same picture as the
  58 slides: tumor is reliable, glass is the main error.
- TCGA labels: mucin inside the tumor vs "mucinous" AUROC 0.76 (12 vs 55); immune score vs MSI-high AUROC 0.55
  (only 5 MSI-high slides, too few to say anything).

## 6. CORRECTED: the "muscle" the blood check removed was mostly pen ink

**Correction (Sun Oct 18, midday).** This section first said the MUSK blood check deletes real muscle. That was
wrong. After protecting muscle-labeled tiles from the blood check, the same tiles were removed as **blur** instead
(F4-6808: blur 6,087; F5-6813: 6,156). Looking at 24 of them (12 per slide), **all 24 are solid pen ink** (red,
blue, green marker on the glass), not tissue: CONCH's tissue model calls ink "muscle", MUSK's blood model calls it
"blood", and the blur rule catches it because ink is perfectly flat. So:
- the blood check was removing **ink**, which is the right outcome for the wrong reason;
- F4-6808's over-call without MUSK came from ink counted as a muscle wall, so MUSK's fix there was real;
- F5-6813's "correct" serosal flag without MUSK was built on ink, so losing it was right;
- the pipeline has no pen-ink check; ink is removed only by luck (blood or blur).

The original numbers, kept for the record (what they show is ink, not muscle):

| slide | real pT | CONCH muscle tiles | of those skipped as "blood" | depth hint: without → with MUSK |
|---|---|---|---|---|
| F4-6808 | T1 | 5,428 | **3,695 (68%)** | reaches the wall (over-call) → wall not reached |
| F5-6813 | T4a | 6,848 | **3,150 (46%)** | reaches the wall → can't judge; **loses the only correct serosal flag** |
| AA-3544 | T2 | 291 | 163 | reaches the wall (exact) → can't judge |
| D5-6540 | T2 | 4,649 | 116 | beyond the wall (over-call) → reaches the wall (exact) |
| G4-6321 | T2 | 2,985 | 5 | wall not reached → can't judge |

- F4-6808 alone is 4,730 of the 7,307 blood-skipped tiles on the 58 slides: mostly ink.
- Lesson: per-slide counts pointed at the right slides, but the label "muscle" had to be checked by eye before
  drawing a conclusion.
- Fix (not applied): a pen-ink check (ink is strongly colored and perfectly flat; blue and green are outside the H&E
  color range).

## 7. Five slides have no resolution information and were tiled at the wrong scale

Some held-out tiles looked zoomed out. Five slides have **no microns-per-pixel information** in the file
(`openslide.mpp-x`, `aperio.MPP` and `aperio.AppMag` all missing) and are small images (about 10,000 px on the long
side): AA-3495, AG-3732, AG-3742, AG-3891 (calibration set) and AA-3506 (held-out). `process_slide.py` then assumes
0.25 µm/px (`get_mpp(slide) or 0.25`). A whole section in 10,000 px means the real resolution is much coarser
(roughly 1–2 µm/px), so each "112 µm" tile actually covers several hundred µm and every distance on these slides is
wrong. All five are "can't judge" for depth, so **the depth and serosa numbers above are not affected**, but their
tissue maps and measurements shouldn't be used (they're also in v1–v3). Fix (not applied): skip slides without a
resolution, or ask for it, instead of guessing.

## 8. Changes made after the test (Sun Oct 18, afternoon)

Team decision: **MUSK is used only for the blood check.** Re-run on the same 125 slides (CPU only:
`scripts/recheck_quality.py` + `scripts/analyze_slide.py` + `scripts/evaluate_slides.py`; results in
`metrics_after_fixes.json`, `per_slide_after_fixes.csv`, and the same under `holdout/`).

| change | why |
|---|---|
| "unsure here" = low confidence only (`src/features.uncertainty`) | CONCH/MUSK disagreement marked 20–28% of the tissue; most of it was MUSK calling tumor or stroma "necrosis" (MUSK's tissue model over-calls necrosis about 4×) |
| MUSK's necrosis → muscle overrule off (`cutoffs.ALT_OVERRULES = ()`) | MUSK only for blood; one line brings it back |
| blood check unchanged: spares fat only | a "spare muscle too" rule was tried and reverted: the "muscle" it protected was pen ink (section 6) |
| `alt_label` still written to tiles.csv | for reference only; the analysis ignores it |

| | calibration (58) | held-out (67) |
|---|---|---|
| "unsure here" (mean share of tissue) | 25.5% → **2.8%** (worst slide 8.4%) | 28.4% → **3.1%** (worst slide 10.6%) |
| depth hint, serosal flag | unchanged (15/16 claims true; 0/13 pT4, 1/45 false alarms) | unchanged (11/11; 0/21, 0/46) |
| necrosis share of tissue | 6.61% → 6.62% | 6.64% → 6.73% |
| quality check (glass / blood / blur) | unchanged | unchanged |

A CONCH/MUSK disagreement count by tissue pair (1.33 M tissue tiles): tumor → MUSK necrosis 5.2% of all tissue,
stroma → necrosis 2.5%, normal mucosa ↔ tumor 2.3%, mucus → necrosis 0.9%, everything else under 1% each.

## Suggested next steps (not done)

0. **Pen ink** (section 6): a real ink check instead of relying on the blood and blur rules. Also stop guessing
   0.25 µm/px for slides without a resolution (section 7).
1. **Glass on edges:** a small glass-vs-tissue check trained on raw TCGA tiles (the 76 glass tiles labeled here are a
   start), or a texture check that tells fat membranes from empty glass.
2. **Mucin:** measure mucin in and around the tumor (e.g. mucus within the filled tumor outline plus a margin).
3. ~~"Unsure here"~~: done (section 8).
4. The 13 held-out slides that didn't fit (listed at the end of `calibration/manifest_holdout_v1.csv`) can be run
   later with `scripts/embed_slides_gpu.sbatch` (finished slides are skipped), about 40 GPU-minutes.

## Files

`metrics.json`, `per_slide.csv` (one row per slide, with the without-MUSK columns), `tissue_fractions.csv`,
`labels_claude_v4.csv`, `labels_summary.json`, `label_sheets/` (11 sheets + `index.csv`); `holdout/` (same for the
held-out run, 5 sheets). Full result folders (tiles, embeddings, previews) are on Delta in
`/work/hdd/bfqi/natmakur/results_v4/` and `results_holdout_v1/` (not in git).
