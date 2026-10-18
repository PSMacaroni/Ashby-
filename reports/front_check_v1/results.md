# By-eye check of the tumor-front features on 20 TCGA slides

**Written:** Sun Oct 18 · **Checked by:** Claude, by eye (no pathologist on the project) · **Slides:** 20 of the 125 from
`reports/overnight_v4` (14 calibration + 6 held-out; T1 5, T2 5, T3 3, T4 7; picked to include pushing and
infiltrative calls, satellites and lymph nodes) · **Sheets:** `scripts/make_eye_sheets.py` (thumbnail with satellites /
nodes circled + tissue map + zones + 3 × 3-tile close-ups) · **Notes:** `notes_claude.csv` · **Examples:** `examples/`

These features had only been tested on fake slides. Nothing here is tuned; it's a first look at whether they make
sense on real sections. Thumbnails are 64× smaller than the scan, so judgments are about layout, not cells.

## Results

| feature | what I checked | result |
|---|---|---|
| main tumor | does the "main tumor" cover the tumor? | **9 yes, 9 partly, 2 no** |
| invasive front | is the black front line on a real tumor / non-tumor boundary? | **4 yes, 8 partly, 8 no** |
| growth pattern | the tool's pushing / infiltrative vs mine from the thumbnail | I could judge 10: **agree 8**; the 2 disagreements are polyps called "infiltrative". 9 can't be judged (tumor-only fragments), 1 n/a |
| direction of invasion | does the arrow point to the deepest invasion? | **4 yes, 1 partly, 3 no**, 12 can't tell / n/a |
| satellites | close-ups of 42 reported satellites | **34 are tumor tissue, 0 are separate tumor islands**: all 34 are part of the main tumor that the map split off. The other 8: normal mucosa (4), pen ink (3), a surface strip (1) |
| lymph nodes | 3 reported nodes | 3 are lymphoid tissue; **0 clearly a lymph node** (1 is a lymphoid follicle in the mucosa; the summary text does say "or a large lymphoid aggregate") |

Across all 120 analyzed slides the tool calls the front **infiltrative on 96%** and reports satellites on **79%**.
Published series put an infiltrative margin at roughly a quarter to a third of colon cancers, so both are inflated.

## Why: four root causes

1. **"Main tumor = the largest connected tumor piece."** The map often breaks one tumor into pieces (stroma, mucus,
   muscle or a tear between them). Only the biggest piece gets a front; every other piece becomes an "island" or
   "satellite" (9 of 20 slides, e.g. `examples/01_F5-6464_split_tumor.jpg`). On 2 slides the biggest piece is the
   wrong part entirely (`17_AA-3875_wrong_main_tumor.jpg`: the real invasion into the submucosa is split off). This is
   the main reason satellites appear on 79% of slides and the pattern is almost always "infiltrative" (satellites
   count toward it).
2. **Fronts on tissue edges.** In tumor-only fragments (6 of 20) the only "front" is a short stretch where tumor meets
   a tear or the cut edge. Pattern and direction are then computed from a few tiles of edge, but a label is still
   given.
3. **Pen ink and artifacts.** 3 of 20 slides have pathologist pen marks over the tissue (`05_F4-6808_pen_ink.jpg`);
   ink becomes "satellites", "necrosis" lines and (see overnight_v4) "muscle". One slide has scanner banding.
4. **False tumor in normal mucosa.** CA-5796 (`20_CA-5796_false_tumor_in_mucosa.jpg`) is mostly folded normal mucosa:
   scattered false "tumor" tiles become the main tumor and its satellites. Its depth hint ("wall not reached") is one
   of the 16 judged in overnight_v4 and rests on false tumor.

## What works

- **The tissue layer map** is good on full-wall sections: mucosa, submucosa, muscle wall, fat and tumor sit where
  they should (`18_CM-4746_best_case.jpg`, `12_DC-6156_good_layers_split_front.jpg`).
- **When the main tumor is right, the front usually is too** (4 of the 9 "yes" slides have a fully right front, and
  the polyp AA-A00W correctly has none), and the direction arrow pointed to the deep invasion on CM-4746, DC-6156,
  F5-6813 and AG-3878.
- Tumor-only fragments are mapped correctly as tumor.

## Suggested fixes (not done)

1. **Main tumor:** join tumor pieces that are close (e.g. within 0.5–1 mm, or through stroma) before choosing the
   main tumor, instead of taking only the largest piece.
2. **Edges:** treat borders on the tissue's cut edge or a tear as "edge", not "front"; when the real front is shorter
   than a minimum length, say "pattern / direction can't be judged" instead of giving a label.
3. **Satellites:** count an island only if it's clearly separate (beyond a distance, with non-tumor tissue between,
   not a split-off piece of the main tumor); hide satellites until that's fixed.
4. **Minimum tumor:** if tumor is only scattered tiles (CA-5796), say "no main tumor found" instead of building a front.
5. **Pen ink check** (also in overnight_v4).

## Limits

One labeler, not a pathologist; thumbnail-level layout plus 3 × 3-tile close-ups; 20 slides picked to cover cases, not
at random (T4 and slides with satellites are over-represented). Treat the numbers as "how often it's obviously right or
wrong", not as accuracy.
