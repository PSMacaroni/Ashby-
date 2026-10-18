# Pipeline update report: Sun Oct 18 (overnight test → main `17bdafc`) and the showcase run

**Written:** Sun Oct 18, evening · **Covers:** the 7 commits on `main` after the overnight test (`41d6e6d`), the
showcase run of 5 slides on that code, and what's next · **Sources:** commit messages, `reports/overnight_v4`,
`reports/front_check_v1`, `reports/cutoffs_v1`, `reports/glass_check_v1`, `docs/preprocessing_test.md`,
`docs/project_timeline.md` · **Labels by eye are Claude's, not a pathologist's** (the project has none).

## 1. In one paragraph

Last night's test of the full pipeline on 125 TCGA slides showed that the clinical core worked (depth hint: no
over-calls on held-out slides; tumor tiles right ~80–88% of the time) but the map around it was noisy: glass and pen
ink passed the quality check and got called fat / necrosis / mucus / "blood", the "main tumor" was just the largest
piece, fronts were drawn on cut edges, almost every slide was called "infiltrative", the mucin measure missed mucinous
tumors, and MUSK's disagreements marked a quarter of every slide "unsure". During the day each of these was fixed or
set from evidence, re-tested on the same 120 usable slides each time (125 minus the 5 without a resolution), and
**the clinical numbers never got worse**: the depth hint kept its single over-call (14/15 claims true on the
calibration slides, 11/11 on held-out ones; one calibration slide moved to "can't judge").

## 2. What changed, in order

| time | commit | change | why (evidence) | effect on the 120 test slides |
|---|---|---|---|---|
| 12:54 | `3effa02` | **MUSK only for the blood check.** Its tissue model's second opinion is kept in `tiles.csv` (`alt_label`) but ignored; "unsure here" = low confidence only | MUSK's tissue model calls tumor / stroma "necrosis" ~4× too often; every CONCH/MUSK disagreement was marked unsure | "unsure here" 25.5% → 2.8% (calib), 28.4% → 3.1% (held-out); depth, serosa, QC unchanged |
| 13:03 | `8a20405` | **Texture-based glass / pen-ink check** (edge density + structure + ink colors) before the blood check | 980 tiles re-read at full resolution and labeled: color can't tell pale fat from tinted glass, texture can (AUROC 0.85 vs 0.58) | on those tiles: glass + ink removed 5% → 71%; real fat lost 0 / 62, tissue lost 7 / 427 |
| 13:16 | `6b660cc` | **Mucin** counts pools touching the tumor (not only enclosed holes); **slides without a resolution are skipped** (no more guessing 0.25 µm/px); by-eye front check on 20 slides | 7 of 9 mucinous slides said "no mucin inside the tumor"; 5 slides were tiled at the wrong scale | mucin vs TCGA "mucinous" AUROC 0.54 → 0.70 (calib), 0.76 → 0.83 (held-out) |
| 13:27 | `cefe820` | **Front fixes:** tumor pieces < 500 µm apart are one tumor; tiles next to glass are surface, not front; < 2 mm of front → "can't judge"; satellites must be separated by tissue; < 1 mm² tumor → no main tumor | by-eye check: main tumor right on 9 / 20, front wrong on 8, 0 of 42 "satellites" were real islands | main tumor right 9 → 14 / 20, front wrong 8 → 0, real satellites 0 / 42 → 8 / 14; depth and serosa unchanged |
| 14:10 | `b4cf680` | **Four cut-offs set from evidence** (chosen on 56 calibration slides, reported on 64 held-out) | see `reports/cutoffs_v1` | see the next table |
| 14:12 | `7d32348` | merge of the glass / ink check into `main` | | |
| 15:40 | `17bdafc` | **Glass / ink check re-run on all 120 slides** | | glass / ink tiles skipped (by eye) 21 / 98 → 53 / 98 (calib), 5 / 39 → 14 / 39 (held-out); ~3 / 4 of old "blood" calls were ink; necrosis share 5.6% → 4.9%; clinical numbers unchanged |

The four cut-offs (`b4cf680`):

| cut-off | before → now | held-out result |
|---|---|---|
| `MUCIN_FLAG_FRACTION` (new) | none → **0.015**: "check for mucinous features" | flags 15 slides, 7 truly mucinous; finds 7 of 12 |
| `LOW_CONFIDENCE` ("unsure here") | 0.5 → **0.9** | below 0.9 the map is right on only 22% of tiles, above it 77%; **unsure now covers ~30% of a typical slide** |
| growth pattern (`SOLIDITY_INFILTRATIVE`, `INFILTRATIVE_MIN_PROTRUSIONS`) | 0.85, 3 → **0.5, 7** | agrees with the by-eye call on 15 of 19 (was 10); infiltrative 93% → 44% of judged slides |
| `BLOOD_PROB` | 0.5 → 0.5 (kept, now documented) | no clean cut-off: 0.3–0.9 mixes blood, tissue and ink |

**Unchanged all day:** the depth hint (judged on only ~16% of sections, those with a thick muscle wall), the serosal
flag (still finds 0 of the pT4 tumors), and the tissue model itself (`models/tissue8_split_v1.joblib`).

## 3. The showcase run (5 slides, main `17bdafc`)

**Run:** Sun Oct 18, 15:45–16:16 on Delta · 4 GPU jobs (`showcase_gpu.sbatch` for all 5, then
`showcase_gpu_one.sbatch` to run DM-A282, DC-6160 and CK-4952 in parallel) · **0.9 GPU-hours in total** (29.9 +
7.2 + 6.6 + 11.3 min) · every slide `phrases_exit=0 analysis_exit=0` (`logs/`) · slides copied to the team folder and
md5-checked against GDC.

Lesson: the first job landed on a slow node (gpua029, ~22 tiles/s, data-loading workers waiting on the disk), while the
two parallel jobs on gpua032 read ~75 tiles/s. **One slide per job is faster** (and here cheaper too, because the slow node held its GPU longest).

| slide (TCGA pT / pN) | tissue tiles kept (skipped: glass·ink / blood / blur) | tumor | growth pattern | depth hint vs TCGA pT | immune | other | unsure | high-risk flags |
|---|---|---|---|---|---|---|---|---|
| **DM-A282** (T3 / N0, IIA) | 27,104 (3,095 / 44 / 681) | 24% | infiltrative (solidity 0.63, 9 fingers) | touches fat, "≥ pT3": **exact** | desert-like | budding hotspot; mucin 0.4% (TCGA: mucinous, **flag misses it**) | 32% | 5 |
| **DC-6154** (T4a / N2b) | 16,119 (249 / 19 / 182) | 26% | infiltrative (solidity 0.41, 47 fingers, 4 satellites) | touches fat, "≥ pT3": **exact** | desert-like | possible venous invasion (4 spots) | 27% | 5 |
| **DC-6160** (T2 / N0, I) | 25,125 (822 / 78 / 57) | 34% | **pushing** (solidity 0.87) | reaches muscle, "≥ pT2": **exact** | excluded-like (front 0.07, core 0.00) | no necrosis, no mucin | 15% | 1 |
| **AZ-4614** (T4a / N1) | 22,900 (163 / 10 / 8) | 44% | **pushing** (solidity 0.83) | reaches muscle, "≥ pT2": shallower (true) | desert-like | **central necrosis 14% of tumor**, low-oxygen rim 13% | 32% | 1 |
| **CK-4952** (T4 / N2) | 14,652 (520 / 23 / 34) | 33% | infiltrative (solidity 0.66) | reaches muscle, "≥ pT2": shallower (true) | **inflamed-like** (core 0.06, front 0.10) | **mucin 20% of tumor → "check for mucinous features"** (TCGA: mucinous) | 32% | 2 |

Compared with the same slides in the overnight test:
- **Mucin fix works where it should:** CK-4952 went from 0.4% to 20% mucin and now gets the "check for mucinous
  features" flag. DM-A282 (also TCGA-mucinous) still shows only 0.4%: the mucin may be in other blocks of that tumor.
- **Growth pattern now separates the pair we picked to contrast:** DC-6160 and AZ-4614 are "pushing", DC-6154 stays
  clearly infiltrative (overnight, nearly every slide was "infiltrative").
- **Depth hints unchanged**, all true against TCGA (3 exact, 2 shallower, none deeper).
- **"Unsure here" went from 1–3% to 15–32% of tissue** (the 0.9 confidence cut-off). Honest, but in the app's
  Certainty view it now covers about a third of the map.
- **Check:** DC-6160 reports 4 satellites although its main tumor is solid; some marked islands sit in a separate
  tissue piece across the glass. Satellites should probably be limited to the analyzed section.

Results for the app: `results/<slide_id>/` (copy into `data/results/`). Checked in the app: all 5 load in the Slide
Box with stage and flag counts, and the slide view shows the scan, tissue overlay, direction arrow and flags.

## 4. What's next

**Deadline:** abstract + ≤ 20-slide deck due **Wed Oct 21**. **GPU:** about 4 of 10 team hours left after this run; the
hackathon reservation ends **today at 18:00**, after which GPU jobs wait in the normal queue.

**For the demo (in order)**
1. **Re-run the 5 slides with the teammate's finished model.** Everything is ready: same slides, same list files, one
   `sbatch showcase/showcase_gpu_one.sbatch showcase/list_<case>.txt` per slide (7–11 min each, ~0.9 GPU-hours for
   all 5). Compare with this run (`results/`) to show what the new model changes.
2. **Load them into the app and polish it** (`app-core` branch, not on `main` yet). Decide how to show "unsure here"
   now that it covers 15–32% of each slide (e.g. only the largest unsure areas, or lighter shading).
3. **Tile previews for click-to-zoom:** the `tiles/` folders (15–31k small JPGs per slide) are on Delta only; copy the
   5 showcase folders' previews to the demo laptop if the app should show real tile pictures.
4. **Fix or explain DC-6160's satellites** across separate tissue pieces (section 3).

**Open from the record (`docs/project_timeline.md` §20, the reports above)**
5. **Normal slides → near-zero tumor** (planned validation in `CLAUDE.md`): never measured. Cheap: 5–10 TCGA `-11`
   slides, a few GPU-minutes each, one line for the deck.
6. **Pathologist:** about 30 lines of `src/cutoffs.py` are still marked PLACEHOLDER (the front fixes, immune cut-offs and others), and
   every by-eye label so far is Claude's. Say so in the deck.
7. **Known limits to state, not hide:** depth is judged on only ~16% of sections (needs a thick muscle wall); the
   serosal flag has found 0 pT4 tumors; necrosis is the weakest class; false tumor in normal mucosa is still open
   (`cefe820`); the mucinous flag finds about half the mucinous tumors.

**Parked (decided earlier):** Lumen plain-language summaries and a "drop a report file" upload in the app.
