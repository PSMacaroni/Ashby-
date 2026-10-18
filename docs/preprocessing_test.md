# Would better preprocessing fix the map errors? (test, Sun Oct 18)

**Question:** NeelA28's overnight test (`reports/overnight_v4/results.md`) found glass and ink labeled as fat / necrosis /
mucus / muscle, and the MUSK blood check removing real tissue. Can preprocessing (a better glass / ink check, color
normalization) fix this, and is the gain big enough to matter?

**Code:** `scripts/eval_preprocessing.py` (Delta GPU, 21 min, ~0.35 GPU-h), `scripts/analyze_preprocessing.py`.
**Tiles:** 980 tiles from 123 slides: the 744 tiles labeled in the overnight test + 250 extra (model-called fat, and
pale tiles), each read again at full resolution (224 px at 0.5 µm/px) exactly like the pipeline.
**Labels:** by eye (Claude), not a pathologist. The overnight labels were made on 128 px previews; I re-labeled the
262 "glass" and extra-fat tiles **at full resolution**, blind and shuffled (`data/preproc_test/sheets/`, codes:
fat / glass / ink / edge = part tissue part glass / tissue / unclear).

## 1. What the "glass" errors really are
At full resolution, the 140 tiles labeled glass on 128 px previews were:

| | tiles |
|---|---|
| empty glass | 35 |
| ink / scanner artifact | 21 |
| **edge tiles** (tissue on one side, glass on the other) | **70** |
| fat | 3 |
| mostly tissue | 8 |

And of 122 tiles the model calls fat, only 40 are real fat (33 edge, 30 glass or ink, 11 other tissue).
So the problem is half **empty glass / ink** and half **edge tiles**.

## 2. A glass / ink check that keeps fat
Color can't tell fat from glass (real fat is as pale as tinted glass). **Texture can:** fat has a net of thin
membranes across the whole tile; glass is flat; an edge tile has structure on one side only.

| measure | glass vs real fat (AUROC) | glass vs other tissue |
|---|---|---|
| colorful share (current check) | 0.58 | 0.88 |
| tissue pixels vs the slide's own glass color | 0.54 | 0.81 |
| **edge density** (share of pixels on a sharp edge) | **0.85** | **0.96** |
| structure coverage (share of 4×4 sub-blocks with tissue or edges) | 0.67 | 0.85 |

**Proposed rule:** glass if **ink** (blue/green > 20%, black > 30% or flat bright red > 50% of pixels) **or edge
density < 0.02 or structure coverage < 0.4**.

| (full-resolution labels) | glass + ink removed | real fat lost | edge tiles removed | tissue lost |
|---|---|---|---|---|
| current pipeline check | 5% | 0 / 62 | 0% | 0 / 427 |
| **proposed rule** | **71%** | **0 / 62** | 26% | 7 / 427 (2%; 3 mucus, 2 lymphocyte, 2 muscle) |
| same rule, held-out slides only | 77% (of 13) | 0 / 8 | | 5 / 116 |

What it does to the map labels (share of tiles with that label that are real tissue / real fat):

| model label | before | after the rule |
|---|---|---|
| muscle | 66% | **88%** |
| necrosis | 65% | **78%** |
| fat (real fat) | 33% | 43% |
| mucus | 72% | 73% |

## 3. Color normalization (Macenko, stain colors estimated per slide)
| | raw | normalized |
|---|---|---|
| CONCH tissue model agrees with the eye (373 tissue tiles) | 60% | 60% |
| MUSK second opinion agrees | 60% | 58% |
| tiles whose label changed | | 71: 23 fixed, 25 broken |
| **MUSK blood check: real blood caught** (of 24) | 24 | 22 |
| **MUSK blood check: real tissue wrongly removed** | **11** | **3** |

- For the tissue labels, **no gain** (muscle +13 points, stroma −13, the rest about even). Not worth adding.
- For the **blood check**, normalization removes most false "blood" calls while keeping nearly all real blood. Worth
  trying, but the sample is small (blood set: at most 4 tiles per slide). (The big "muscle removed as blood" counts
  on F4-6808 / F5-6813 turned out to be red pen ink, main 3effa02; the ink check below now handles those.)

## Verdict
| fix | gain | recommendation |
|---|---|---|
| texture-based glass / ink check | **large for QC** (glass caught 5% → 71%, no fat lost); **moderate for the map** (muscle / necrosis labels much cleaner) | **adopt** |
| color normalization for the tissue model | none | skip |
| color normalization for the MUSK blood check | promising (false removals 11 → 3) | try later on more blood tiles (e.g. CM-4746's clot) |
| any of these for depth, tumor %, immune score | small: those rely on tumor / lymphocytes, already 80–88% right | |

## Added to the pipeline (branch `preprocess-test`, on top of `main` 3effa02)
- `src/detectors.texture_measures` / `texture_glass` / `glass_color`; cut-offs `GLASS_EDGES_MAX`,
  `GLASS_STRUCTURE_MIN`, `INK_*` in `src/cutoffs.py`.
- `scripts/process_slide.py` measures them for every tile while reading it (no extra slide reads) and saves the
  columns `edges, structure_cov, ink_bluegreen, ink_black, ink_red` in tiles.csv. `quality_check()` uses them only
  when the columns exist, so older folders and `--extend` on them behave exactly as before.
- Glass and ink are skipped as `background` **before** the blood check: the red pen ink on F4-6808 / F5-6813 that
  MUSK called "blood" (reports/overnight_v4, section 6) is now skipped for the right reason. Real blood is not hit
  (0 of 54 labeled blood tiles).
- Already-processed slides: `scripts/add_texture_measures.py` (CPU, reads the tiles again, no model), then
  `scripts/recheck_quality.py` and `scripts/analyze_slide.py`.
- The color normalization is **not** in the pipeline (no gain for the tissue labels; the blood-check gain needs a
  bigger check first). `qc_reason` values are unchanged (`background` / `blood` / `blur`).
- Checked: the pipeline's function reproduces the test (glass removed 70%, fat lost 0 / 62, tissue lost 7 / 427 on the
  same tiles); `tests/test_texture_check.py`.

**Limits:** labels by eye (not a pathologist); the cut-offs were picked on these same tiles (the held-out slides look
the same, but only 13 glass / 8 fat tiles there); edge tiles are a judgment call (how much tissue makes a tile
"tissue"?).
