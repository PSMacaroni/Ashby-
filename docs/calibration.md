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

## v3: judge position, not contact (same 58 slides, same tissue maps as v2)

**Written:** Sat Oct 17, night · **Compute:** laptop CPU, ~7 min (72,000 setting combinations) · **Code:**
`src/features.py` (`beyond_wall`, `depth_evidence`, `depth_level`), `tests/test_depth_rule.py` · **Results:**
`calibration/calibration_v3.json`, `learning_curve_v3.png`

### What's new
1. **"Beyond the muscle wall" rule** (`DEPTH_RULE = 'position'`). A raw tumor tile counts as beyond the wall when:
   - it lies at least `DEPTH_BEYOND_MIN_UM` outside the wall band (the band = muscle with gaps between bundles
     closed and pockets filled in), and it isn't in a lymph node;
   - the straight line from it to the nearest **lining** tile crosses the wall. Lining = normal mucosa, plus the
     tissue edge where mucosa or tumor dominate (the luminal surface);
   - most nearby lining tiles are behind the wall too, so one stray lining tile can't decide the side;
   - the nearest *outside* landmark (fat, or the fat/muscle/stroma tissue edge) is not behind the wall.

   With `DEPTH_BEYOND_MIN_TILES` such tiles, the hint says "consistent with at least pT3". Tumor wrapped in stroma
   inside the fat now counts, and fat inside the wall (submucosa) no longer does. `'either'` = this rule or the old
   fat-contact rule.
2. **Thick-wall check** (`WALL_MIN_WIDTH_UM`): a muscle region's average thickness (area ÷ length of its midline)
   must reach this value to count as the wall. This targets the thin muscularis mucosae.
3. The calibration also tunes these, reports each rule on its own, and counts how many T3/T4 slides reach "≥ pT3".

### Results (judged = slides where depth could be judged; % = share of judged)
| depth hint | judged | correct | **deeper than pT** | T3/T4 reached (of 27) |
|---|---|---|---|---|
| v2: starting settings, contact rule | 30 | 9 (30%) | 7 (23%) | 4 |
| v3: starting settings, position rule | 30 | 10 (33%) | 7 (23%) | 5 |
| v3: starting settings, either rule | 30 | 11 (37%) | 8 (27%) | 7 |
| v2: best tuned (all slides) | 19 | 8 (42%) | 4 (21%) | 3 |
| v3: best tuned (all slides) | 26 | 15 (58%) | 1 (4%) | 4 |
| **v2: honest (leave-one-out)** | 19 | 6 (32%) | **6 (32%)** | 3 |
| **v3: honest (leave-one-out)** | 25 | 8 (32%) | **5 (20%)** | 3 |
| v3 honest, 'contact' rule only | 20 | 5 (25%) | 7 (35%) | 3 |
| v3 honest, 'either' rule only | 23 | 12 (52%) | 6 (26%) | 7 |

The best tuned v3 settings are: position rule, muscle wall ≥ 4 mm² **and ≥ 600 µm thick**, contact = 3 tiles within
2 tiles, and beyond = 5 tiles ≥ 100 µm past the wall. Tied settings give the same result, and the script keeps the
first one, so treat 100 µm / 5 tiles as "low".

What moved, and why:
- **The thick-wall check does most of the work against over-calls.** With starting settings plus a 600 µm wall,
  over-calls drop 7 → 3 with the contact rule and 7 → 1 with the position rule. Tumor touching thin muscle strips no
  longer counts as "reaches the wall": AZ-4313, AZ-5407 and G4-6323 (pT1/Tis) become correct, and CM-6163 (pT1)
  becomes "can't judge". The position rule then also fixes G5-6641 (pT1) and G4-6628 (pT2).
- **The position rule finds 3 T3/T4 tumors that never touch fat tiles:** AZ-6606 (pT4), F5-6464 (pT4b) and
  G4-6307 (pT3). G4-6628 (pT2, the submucosal-fat over-call) is no longer called pT3. (Thick-wall check off; with
  it on, the tuned settings still reach these 3 plus CM-5868.)
- **It loses 2** that only the old rule catches: AF-2690 (pT3, fat contact, no wall in the section) and DC-6154
  (pT4a, wall too small). The position rule needs a wall. The 'either' rule keeps both (7/27 reached) but over-calls
  more.
- Still wrong: DC-5337 (pT1 → "reaches the wall"; possibly a label issue, see v1). On held-out runs, AA-3877,
  AZ-4313, G4-6323 and G5-6641 flip to over-calls depending on which near-tied settings win.

### Honest caveats
- **The gain is small and unstable.** It is 1 fewer held-out over-call than v2 (5 vs 6), on more judged slides
  (25 vs 19). An earlier run with a narrower grid got 11 correct / 5 over-calls of 27 judged. Adding more values to try
  made the honest estimate *worse* (11 → 8 correct): with 58 slides, each extra setting is a chance to overfit.
- **The fixed parts were designed on these same slides.** The lining definition, the majority vote and the
  outside-landmark check were chosen while looking at the v2 maps (CM-4746, AZ-6603, G4-6307, CM-5868). Leave-one-out
  doesn't account for that, so the real error is probably higher. The fixed numbers in `src/cutoffs.py`
  (`WALL_BAND_CLOSE_TILES`, `WALL_CROSS_MIN_UM`, `DEPTH_LINING_REACH_UM`, `DEPTH_BEYOND_MAJORITY`) are PLACEHOLDERs.
- **T3/T4 coverage is still the slides' limit:** 19 of 27 T3/T4 sections show no wall at all, so no rule can place
  tumor "beyond" it.
- **Known blind spots:**
  - Where tumor replaces the whole wall thickness, tumor below the gap "sees" the lining through it and doesn't count
    (the fake test slide shows this: 27 tiles beyond, not enough at 30).
  - Tumor reaching the outer edge counts as lining, which hides it (CK-6746 bottom edge).
  - A thick muscle fold, or tumor pockets between muscle fingers, can look like "beyond" (AZ-6603 before the
    thickness check).
- The learning curve is still noisy/flat. Held-out over-calls are slightly lower than v2 at larger pool sizes
  (≈ 0.10–0.22 vs 0.13–0.27).

### Decision
- **Not applied** (`--apply` not run; `src/cutoffs.py` still uses the contact rule with no thickness check). The
  held-out over-call rate did go down (32% → 20%), but by one slide, it moves with the grid, and the rule's fixed
  parts were tuned by eye on these slides. `--apply` would also change the serosa settings, whose held-out result
  didn't improve (still 0/13 pT4, 2 false alarms).
- **If the team wants it anyway,** the safest single change is the thick-wall check: `WALL_MIN_WIDTH_UM = 600`. It is
  one simple number with a clear anatomical reason, and it removed the most over-calls. Then
  `DEPTH_RULE = 'position'` with `WALL_MUSCLE_MIN_MM2 = 4.0`, `DEPTH_CONTACT_RADIUS_TILES = 2`,
  `DEPTH_BEYOND_MIN_UM = 100`, `DEPTH_BEYOND_MIN_TILES = 5`. Re-check on new labeled slides before showing it as more
  than experimental.
- **For the pathologist:** confirm that the muscle the rule treats as the wall on AZ-6606, F5-6464 and G4-6307 is
  the muscularis propria, and that the tumor it marks lies outside it.

### v3b: depth is judged only when the muscle wall is in the section
**Change (now the default):** without a muscle wall, the depth hint says "can't judge", even if fat is present. Before
this, a section with fat but no wall still got a depth: A6-6138 (pT2) got a false "the muscle wall is in this section
but tumor doesn't reach it", and AF-2690 (pT3) got "touches fat → at least pT3", which is right there but is the same
reasoning that over-called G4-6628. Tumor touching fat in a section without a wall now gives a **warning instead of a
depth**: the hint and the summary panel say "Tumor touches fat, but no muscle wall in this section: depth unknown (check
the fat)" (`invasion.fat_contact_without_wall`). 28 of 58 sections show a wall; the other 30 are "can't judge".

**How accurate is the hint on sections that show the wall?** These use fixed settings, not tuned slide by slide. The
hint's claim is "consistent with **at least** X", so it is true unless it says deeper than pT.

| settings (fixed) | judged | exact level | shallower (claim still true) | **deeper than pT** | claim true |
|---|---|---|---|---|---|
| old contact rule, starting settings (**the app today**) | 28 | 8 | 13 | **7** | 21 / 28 (75%) |
| position rule, starting settings | 28 | 10 | 11 | **7** | 21 / 28 (75%) |
| position rule + thick wall ≥ 600 µm | 17 | 9 | 7 | **1** | 16 / 17 (94%) |

Rows = real pT, columns = what the hint said (1 = wall not reached, 2 = reaches wall, 3 = beyond wall / fat):

| | old rule: 1 | 2 | 3 | | thick wall + position: 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| T1/Tis | 2 | 4 ❌ | 2 ❌ | | 3 | 1 ❌ | 0 |
| T2 | 3 | 3 | 1 ❌ | | 2 | 2 | 0 |
| T3 | 1 | 3 | 0 | | 1 | 1 | 1 |
| T4 | 1 | 5 | 3 | | 0 | 3 | 3 |

- The old rule's 7 errors are all pT1/Tis/pT2 tumors called deeper because tumor touched thin muscle or fat. The
  thick-wall check removes 6 of them. The 600 µm value was picked by the v3 tuning on these same slides, so 16/17 is
  optimistic.
- The cost: 11 sections with only thin muscle become "can't judge" (28 → 17 judged).
- The remaining error is DC-5337 (pT1 → "reaches the wall"), which may be a label issue (see v1).

**Held-out (leave-one-out) numbers after this change are less useful.** With fewer sections to judge, the tuner learns
that judging nothing is cheapest: the best settings judge only 9 slides. Held out: 10 judged, 5 exact, 3 deeper (30%).
The position rule alone, held out: 17 judged, 7 exact, 4 deeper (24%). With 10–17 judged slides, one slide moves these
by 6–10 points. The scoring should require a minimum number of judged slides before these numbers mean much.
`calibration/calibration_v3.json` holds this run; the first v3 table above was computed before the change.

**Applied (Sun Oct 18): thick-wall check `WALL_MIN_WIDTH_UM = 600`** is now the default, together with the "needs a
wall" gate. The depth rule itself is still `'contact'`; `'position'` stays an option. With the new defaults, on the 58
slides:

| defaults | judged | exact level | shallower (claim still true) | **deeper than pT** | claim true | T3/T4 reached | serosal false alarms |
|---|---|---|---|---|---|---|---|
| before (wall gate, no thickness check) | 28 | 8 | 13 | **7** | 21 / 28 (75%) | 3 | 2 / 45 |
| **now (wall gate + 600 µm wall)** | 17 | 7 | 9 | **1** | 16 / 17 (94%) | 2 | 1 / 45 |

- Fixed: AZ-4313, AZ-5407, G4-6323 (pT1/Tis → no longer "reaches the wall"), G4-6628 (pT2) and G5-6641 (pT1) (no
  longer "≥ pT3"), CM-6163 (pT1 → "can't judge"). Left: DC-5337 (pT1 → "≥ pT3"; possible label issue).
- Cost: 11 more sections say "can't judge" (thin muscle only), and one fewer T3/T4 reaches "≥ pT3" (3 → 2). The
  serosal flag also needs the wall (or fat), so its false alarms dropped 2 → 1; it still finds 0/13 pT4.
- **Caveat:** 600 µm was picked by looking at these same 58 slides, so "1 over-call in 17" is optimistic. Check it on
  new labeled slides.
