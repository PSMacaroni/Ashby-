# Calibrating the analysis against real pT (v1, v2)

**Written:** Sat Oct 17 · **Slides:** 53 TCGA colon/rectum slides downloaded from the NCI GDC (`calibration/manifest_v1.csv`)
+ our 5 = **58 slides with known pT** (T1/Tis 18, T2 13, T3 14, T4 13; untreated, one per patient)
**Compute:** pipeline on Delta, 71 min on one A100 for 53 slides; calibration on the laptop CPU in seconds
**Code:** `scripts/get_gdc_slides.py`, `scripts/calibrate_analysis.py` · **Results:** `calibration/calibration_v1.json`,
`calibration/learning_curve_v1.png`

## What pT is, and why it's a noisy answer key for one slide

pT = how deep the primary tumor grows into the bowel wall, set by the pathologist on the **whole surgical specimen**
(Tis mucosa · T1 submucosa · T2 muscle wall · T3 through the muscle into fat · T4a through the serosa · T4b other organs).

A TCGA diagnostic slide is **one section**, so it can miss the deepest point (the slide looks shallower than pT). Many
sections have no wall layers at all. T3 vs T4a is a judgment call, and cases span hospitals and staging editions. So:
- the slide can only say "consistent with **at least** X";
- **claiming deeper than pT** is the real error;
- shallower is often just sampling.

## How the calibration works
1. Every calibratable setting (`src/cutoffs.py`: muscle-wall size, fat size, how many tumor tiles must touch, contact
   distance; serosa edge length / distance from the wall) is tried over a grid.
2. Scoring: correct +1, deeper than pT −2, shallower −0.5, can't judge 0.
3. Honest accuracy = **leave one slide out** (tune on the others, test on it).
4. A **learning curve** ("predict first, then learn", 50 random orders) shows whether more slides would help.
5. Re-run whenever more labeled slides are processed: everything is re-tuned on the larger set.

## Results

| | starting settings (all slides) | best tuned settings | **honest (leave-one-out)** |
|---|---|---|---|
| depth judged | 25 / 58 | 20 / 58 | 24 / 58 |
| correct (of judged) | 40% | 40% | **25%** |
| deeper than pT (of judged) | 20% | 20% | **33%** |
| serosal flag: pT4 found | 0 / 13 | 0 / 13 | 0 / 13 |
| serosal flag: false alarms | 7 / 45 | **1 / 45** | 1 / 45 |

| real pT | can't judge | wall not reached | ≥ T2 (muscle) | ≥ T3 (fat) |
|---|---|---|---|---|
| T1 (18) | 10 | 3 ✅ | 5 ❌ | 0 |
| T2 (13) | 6 | 3 | 4 ✅ | 0 |
| T3 (14) | 9 | 1 | 3 | 1 ✅ |
| T4 (13) | 8 | 0 | 3 | 2 ✅ |

- **The learning curve is flat** (correct ≈ 30–40%, deeper-than-pT ≈ 25–35%). More slides don't help: the errors come from
  things settings can't fix.
- **Applied:** the serosal settings (`SEROSA_MIN_RUN_UM = 1000`, `SEROSA_MAX_FROM_WALL_UM = 500`): false alarms 7 → 1,
  nothing lost.
- **Not applied:** the depth settings. The tuned ones did worse on held-out slides than the starting values.

## Why: root causes, by checking the slides themselves

| cause | evidence | type | fixable? |
|---|---|---|---|
| **57% of sections (33/58) show no muscle wall or fat** | tumor-only pieces; 19 of 27 T3/T4 sections | the slides | no; "can't judge" is the honest answer |
| **The tissue mask still drops pale fat** | DC-5337: a large fat region is in the slide but mostly missing from the map; fat found next to tumor on only 3 of 27 T3/T4 slides | pipeline | **yes**: a fat-aware mask (fat is pale but has a lacy pattern; glass is uniform), then the background guard removes glass |
| **The thin muscle inside the lining counted as the muscle wall** | AZ-5407 (pT1): surface tumor glands "touching" muscle in a polyp-like lesion | rule + model (a 112 µm tile can't tell muscularis mucosae from muscularis propria) | partly: require a thick, continuous band |
| **Possibly wrong or partial labels** | DC-5337 (pT1): tiles look like tumor glands between smooth-muscle bundles (muscle-wall invasion) | label noise | needs a pathologist |
| **Blur cut-off is a placeholder** | drops 55–65% of tiles on AG-A025 / AG-3892 | pipeline | **yes**: pick `BLUR_MIN` by looking at ~20 tiles |
| **pT4 evidence is rarely in the section** | 0/13 found at any setting | the slides + mask (fat) | partly |

## What this means for the project
- **Present the depth hint as experimental,** with three honest outputs:
  - "can't judge from this section" (most TCGA slides);
  - "reaches the muscle wall";
  - "touches fat".
  Show its accuracy plainly: **25% correct / 33% deeper-than-pT on held-out slides, with judgments on 41% of slides**.
  This is a useful negative result.
- **Next fixes, in order of value:**
  1. a fat-aware tissue mask (re-tile only the cells near tissue that the mask dropped; cheaper than a full re-run);
  2. a real `BLUR_MIN`;
  3. a muscle-wall shape rule (thick continuous band);
  4. re-run the calibration (v2) and compare.
- **Ask the pathologist** to look at DC-5337 and AZ-5407 (and the 5 T1 over-calls in `calibration/calibration_v1.json`).

## v2: better tissue map (fat-aware mask, real blur rule), same 58 slides

**Written:** Sat Oct 17, evening · **Compute:** 1 GPU-hour on Delta (`process_slides_gpu.sbatch ... --extend`: only the
new grid cells were read, +106k tiles = +22%) · **Results:** `calibration/calibration_v2.json`, `learning_curve_v2.png`;
v1 files backed up on Delta in `/projects/bfqi/hjeong6/results_v1_backup`

### What changed in the pipeline
| fix | how | checked on |
|---|---|---|
| **Fat-aware tissue mask** (`src/slides.fat_aware_mask`) | fat is nearly white like glass, but it has thin membranes, so its brightness changes locally; glass is flat. Pale pixels with local texture next to stained tissue now count, then gaps are closed and holes filled | DC-5337, G4-6321 (a big fat piece with a lymph node was almost all missing), CM-5868, G5-6641 |
| **Glass check** (`src/detectors.quality_check`) | tile blur < 0.0003 **and** < 30% colorful = glass, whatever the tissue model says | 80 sample tiles on AG-A025 / AG-3892: all glass; only 18 of 12,401 fat tiles qualify |
| **Blur rule** | blurry = sharper than only 10% of this slide's typical tile. A fixed cut-off fails: soft scans' readable tissue scores below sharp scans' junk | AG-A025: 65% → 0% dropped; what's still dropped on AG-3892 / F4-6808 is glass and pen ink |

| (all 58 slides) | v1 | v2 |
|---|---|---|
| tissue tiles kept | 461,502 | 532,926 |
| fat tiles | 12,164 | **55,365** (4.5x) |
| sections with a muscle wall / with a fat layer | 23 / 11 | 28 / 18 |
| T3/T4 slides where tumor touches fat (>= 3 tiles) | 3 / 27 | **4 / 27** |

### Calibration result: the map got better, the depth hint didn't
| depth hint, starting settings | v1 | v2 |
|---|---|---|
| judged | 25 / 58 | **30 / 58** |
| correct | 10 | 9 |
| shallower than pT ("at least" still true) | 10 | 14 |
| **deeper than pT** (the real error) | 5 (20% of judged) | 7 (23%) |
| honest tuned (leave-one-out): correct / deeper | 25% / 33% | 32% / 32% |

What moved (8 slides):
- 5 slides (4 pT4, 1 pT2) now show a muscle wall, so they go from "can't judge" to "reaches the muscle wall" = "at least
  pT2". True, but scored "shallower".
- CK-6746 (pT4b): now touches fat → correct.
- **New over-calls:** AZ-4313 (pT1a) now has 5 tumor tiles next to muscle (the thin muscularis mucosae problem again);
  G4-6628 (pT2) now has 6 tumor tiles next to fat: fat inside the bowel wall (submucosal fat) or a label issue.

**Why the fix didn't help depth:** the missing fat was a real data problem and is now fixed, but even with 4.5x more fat
only 4 of 27 T3/T4 tumors "touch" it. In a T3 tumor the cancer glands in the fat are usually wrapped in scar-like stroma
(desmoplasia), so tumor tiles rarely sit right next to fat tiles. And fat inside the wall (submucosa) looks the same as fat
outside it. **The bottleneck is now the rule, not the map.** The learning curve is still flat.

### Decisions
- **Keep** the new mask, glass check and blur rule (better tissue maps for the app and every other measurement).
- **Don't apply** the v2 depth or serosa settings: neither beats the current ones on held-out slides (serosa still finds
  0/13 pT4).
- **Next idea for depth:** judge position, not contact: "is there tumor **on the far side of the muscle wall** (away from
  the mucosa)?" That captures tumor wrapped in stroma within fat, and ignores fat inside the wall.
