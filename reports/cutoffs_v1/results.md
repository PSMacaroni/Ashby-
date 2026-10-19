# Four cut-offs set from evidence (v1)

**Written:** Sun Oct 18 · **Labels by:** Claude, by eye, blind (no pathologist on the project); mucinous = TCGA's
diagnosis · **Slides:** 56 calibration (`results_v4`) to choose each cut-off, 64 held-out (`results_holdout_v1`) to
report it · **Reproduce:** `python scripts/calibrate_cutoffs.py --results-calib <results_v4> --results-holdout
<results_holdout_v1>` · **Sheets:** `scripts/make_pattern_sheets.py`, `scripts/make_label_sheets.py --sets blood_bins`

| cut-off (`src/cutoffs.py`) | before | now | evidence | held-out |
|---|---|---|---|---|
| `MUCIN_FLAG_FRACTION` (new) | none (only WHO 50%) | **0.015** → "check for mucinous features" | TCGA mucinous diagnosis | flags 15 slides, 7 mucinous; finds 7 of 12 |
| `LOW_CONFIDENCE` | 0.5 | **0.9** | 437 tiles labeled by eye | map right on 22% of tiles below 0.9 vs 77% above |
| `SOLIDITY_INFILTRATIVE`, `INFILTRATIVE_MIN_PROTRUSIONS` | 0.85, 3 | **0.5, 7** | 88 slides called pushing / infiltrative by eye | 15 of 19 agree (was 10 of 19) |
| `BLOOD_PROB` | 0.5 | **0.5** (kept) | 192 tiles in 4 p_blood bands | no clean cut-off; see below |

Re-analysis of all 120 slides with the new cut-offs: depth hint and serosal flag unchanged (claims true 15/16 and
11/11; pT4 found 0/13 and 0/21); no slide's depth hint changed (`metrics_calib.json`, `metrics_holdout.json`).

## 1. Mucinous flag

The map sees much less mucin than a pathologist: only 1 slide per set is over WHO's 50%, so the "Mucinous" wording
almost never fires. A second, lower cut-off now says "check for mucinous features", calibrated against TCGA's
diagnosis (one slide per case; a tumor can be called mucinous from other blocks, so some misses are expected).

| mucin in tumor ≥ | calibration (9 of 56 mucinous) | held-out (12 of 64 mucinous) |
|---|---|---|
| 0.5 (WHO) | flags 1, 1 mucinous | flags 1, 1 mucinous |
| 0.1 | flags 7, 5 mucinous, bal. acc. 0.76 | flags 7, 4 mucinous, 0.64 |
| **0.015** | flags 13, **6** mucinous, **0.76** | flags **15**, **7** mucinous, **0.71** |
| 0.005 | flags 14, 6 mucinous, 0.75 | flags 22, 9 mucinous, 0.75 |

Any value from 0.015 to 0.02 is equally good on calibration; 0.02 sits exactly on one mucinous slide (A6-3809 at
2.0%), so 0.015 is the middle of the gap. **About half the flagged slides are mucinous** (6 of 13, 7 of 15), and it
finds 6 of 9 and 7 of 12. The text says so: "*x*% of the tumor … is mucin: check for mucinous features (on TCGA slides
at this level, about half were diagnosed mucinous)". `summary.json` → `mucin.check_mucinous` (new, true / false).

## 2. "Unsure here" (`LOW_CONFIDENCE`)

From the 437 class-sample tiles labeled by eye in `overnight_v4` (blind; "unclear" left out):

| top probability | calibration: map right | held-out: map right |
|---|---|---|
| < 0.5 | 12% (33 tiles) | 12% (16) |
| 0.5–0.7 | 24% (55) | 22% (27) |
| 0.7–0.9 | 39% (70) | 29% (24) |
| 0.9–0.99 | 60% (75) | 68% (41) |
| ≥ 0.99 | 81% (62) | 88% (34) |

Below 0.9 the map is wrong most of the time; above it, right most of the time. At 0.9, **114 of 156 wrong
calibration tiles are marked unsure (52 of 69 held-out)**; at the old 0.5, only 29 (14). 0.95 adds little.

**The price: "unsure here" now covers about 30% of the tissue on a typical slide** (it was ~3%). Measured on every
usable tile of the 120 slides:

| `LOW_CONFIDENCE` | tissue marked unsure, median slide: calibration | held-out |
|---|---|---|
| 0.5 (old) | 2.4% | 2.8% |
| 0.7 | 12.5% | 13.8% |
| 0.8 | 18.6% | 20.4% |
| **0.9 (new)** | **27.9%** (max 70%) | **30.9%** (max 75%) |

The table above samples every map class equally, so hard classes (fat, necrosis, mucus) count too much. Weighted by
how many tiles of each class a real slide has, **the map is right on ~38% of tiles below 0.9 (39% calibration, 37%
held-out) and on 72–86% above it**, and 0.9 marks 47–64% of the wrong tiles. So 0.9 is about where the map goes from
"more often wrong" to "more often right": the 30% is an honest picture of how unsure the tissue model is, not a side
effect. Mostly unsure: necrosis, mucus, muscle and stroma; tumor is mostly confident (83–84% of tumor tiles ≥ 0.9). If
30% is too much on screen, 0.7 (13%) marks only the tiles the map gets right about 1 time in 5 (class sample) and leaves
the 0.7–0.9 band (right about 1 time in 3) unmarked. Per-slide shares: `unsure_share.csv`.

## 3. Pushing vs infiltrative

**Labels.** `make_pattern_sheets.py` crops each slide's thumbnail to the main tumor and dots the front tiles; the tool's
call is hidden. I labeled 88 slides with a front long enough to be judged (37 calibration + 51 held-out): pushing (broad,
rounded, well-defined deep edge), infiltrative (glands / nests scattered into the wall, ragged front), or can't tell.
**50 of 88 were "can't tell"**: tumor-only fragments, villous pieces with no wall, heavy pen ink. That leaves 19 + 19
slides (8 + 9 infiltrative). The thumbnail is ~32 µm per pixel, coarser than a pathologist's low power: these are calls
about the shape of the tumor edge, not about cells. Labels: `pattern_labels_claude.csv`; examples: `examples/pattern_*.jpg`.

**What separates them.** Solidity barely does (median 0.63 infiltrative vs 0.75 pushing, wide overlap). The number of
fingers + satellites does (median 13 vs 5).

| rule: infiltrative if solidity < S or fingers + satellites ≥ P | calibration (19) | held-out (19) |
|---|---|---|
| old: S 0.85, P 3 | 8 right (all called infiltrative), bal. acc. 0.50 | 10 right, 0.55 |
| **new: S 0.5, P 7** | **15 right, 0.80** | **15 right, 0.80** |
| S 0.5, P 6 / P 8 | 14 right / 14 right | 14 / 15 |
| per cm of front, roughness, fractal dimension | as good on calibration (0.77–0.82) | **worse**: 0.52–0.64 |

Any S up to 0.5 gives the same result (higher is worse), so solidity now only fires for very ragged outlines. On
held-out the new rule finds all 9 infiltrative slides and calls 6 of 10 pushing slides pushing.

| all slides with a judged front | infiltrative | pushing | can't judge |
|---|---|---|---|
| before (calibration + held-out) | 82 | 6 | 32 |
| **after** | **39 (44% of judged)** | **49** | 32 |

44% is closer to published rates (roughly a quarter to a third of colon cancers) than 93% was. **Caveats:** the count is
not per mm of front, so large sections lean infiltrative (per-cm versions did worse on held-out); the labels are mine
from thumbnails; and half the slides could not be judged at all, mostly because they show no wall. The summary caveat
now says this instead of "placeholder". The fake infiltrative test slide got 2 more fingers so it is still infiltrative
under the new rule (`src/fake_slide.py`).

## 4. Blood check (`BLOOD_PROB`)

96 new tiles, 24 from each p_blood band (0.3–0.5, 0.5–0.7, 0.7–0.9, 0.9–1; not fat), spread over the calibration
slides and mixed blind (`blood_labels_claude.csv`, `examples/blood_tiles_sheet1.jpg`), plus the 96 blood / border tiles
from `overnight_v4`:

| p_blood band | blood | real tissue | pen ink / glass / blur | mixed / unclear |
|---|---|---|---|---|
| 0.3–0.5 | 24% | 29% | 29% | 18% |
| 0.5–0.7 | 27% | 31% | 21% | 21% |
| 0.7–0.9 | 42% | 28% | 25% | 5% |
| ≥ 0.9 | **81%** | **4%** | 15% | 0% |

Scaled to the number of tiles in each band (`blood_bands.csv`), on the 56 calibration slides (490k non-fat tiles):

| BLOOD_PROB | tiles skipped | blood | real tissue lost | ink / glass / blur | blood kept (p ≥ 0.3) |
|---|---|---|---|---|---|
| **0.5 (kept)** | 7,307 | 3,319 | 1,732 | 1,591 | 579 |
| 0.7 | 4,978 | 2,696 | 1,010 | 1,099 | 1,202 |
| 0.9 | 1,517 | 1,225 | 58 | 233 | 2,673 |

**No clean cut-off:** between 0.3 and 0.9, MUSK's score is a real mix. 0.9 is precise but keeps about two-thirds of the
blood (which CONCH mostly calls necrosis or tumor). 0.7 vs 0.5 trades ~700 real-tissue tiles for ~600 blood tiles,
about 0.1% of the tissue either way, within the noise of these labels. 0.5 also removes the most pen ink, and nothing
else in the pipeline catches ink. So **0.5 stays**, now with these numbers instead of "placeholder". Held-out check
(24 tiles skipped as blood in `overnight_v4`): 8 blood, 5 ink / glass, 7 mixed / unclear, 4 real tissue, in line.
Most tiles in the 0.5–0.9 bands come from three ink-heavy slides (F4-6808, F5-6813, F4-6807), so the "real tissue
lost" estimate, made with labels spread evenly over slides, is probably on the high side.

## Limits

- One labeler, not a pathologist. The tile labels and pattern calls are by eye, from 128 px tiles and thumbnails.
- Small numbers: 19 + 19 slides for the pattern, 9 + 12 mucinous slides. Expect a few points of swing on new slides.
- The tile labels were sampled per class (equal numbers per map label), so rare, hard classes (fat, necrosis, mucus)
  are over-represented compared with a real slide.
- Not changed: `FRONT_MIN_UM`. My "can't tell" slides mostly had short fronts (median 8 mm vs 17–29 mm), but so did
  several clear polyps, so a longer minimum would also hide judgeable slides.
