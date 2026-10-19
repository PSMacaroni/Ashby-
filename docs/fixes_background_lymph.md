# Two problems found on Kather 2016, and how we fixed them

**Written:** Sat Oct 17 · **Test data:** Kather 2016 textures (5,000 non-normalized tiles, 10 Mannheim patients, never used
for training) plus CRC-VAL-HE-7K · **Code:** `src/detectors.py`, `scripts/models/train_immune.py`
**Model:** `models/immune_v1.joblib` · **Numbers:** `reports/immune_v1/metrics.json`, `reports/fixes/necrosis_guard_stats.csv`

Both problems come from the same root cause: **asking one 8-class tissue model a question it wasn't built for.** Both
fixes follow the "separate models" idea. `tissue8` answers "what tissue is this?", and small dedicated checks answer
"is this glass?" and "are lymphocytes here?".

| problem | fix | result on unseen data |
|---|---|---|
| glass tiles called "necrosis" | **background guard:** if the label isn't fat and < 10% of pixels are colorful → background | 99% of glass caught; no fat-labeled or correctly labeled tile removed |
| scattered lymphocytes missed | **immune detector v1:** a separate "lymphocytes present?" model | immune signal on missed tiles 0.04 → 0.44 (stroma 0.01, tumor 0.03) |

---

## Problem 1: empty glass is called "necrosis"

### What happens
- `tissue8` has no "background" answer, so on Kather's 625 empty-glass tiles it says **necrosis 88%**, muscle 10%.
- The pipeline's tissue mask (`src/slides.py` → `tissue_mask`) works on a **64× downsampled thumbnail** and keeps a tile
  if ≥50% of its cell looks like tissue. Partly-glass tiles (edges, tears, holes) can get through, and they would
  inflate necrosis, the "dead core" flag and the necrosis-rim features.

### The trap: fat
Fat is mostly white too (median Kather fat tile: 13% colored pixels; glass: 0%). Filters loose enough to catch all glass
also remove fat: "< 5% stained pixels" removes 99% of glass but **23% of fat**. Fat matters for the "tumor touches fat"
T3 hint.

### What we tried

| approach | glass removed | fat lost | problem |
|---|---|---|---|
| pixel rule < 5% stained | 99.2% | 22.9% | far too much fat |
| pixel rule < 1% stained | 94.7% | 2.2% | still loses fat |
| zero-shot "background" prompt | 99.0% | 12.8% | also flags 12% of debris, 8% of tumor |
| trained glass detector + pixel rule | 98.6% | 3.0% | trained on **fake** glass (see below) |
| **background guard (chosen)** | **99.0%** | **0 fat-labeled tiles** | |

**Why we dropped the trained glass detector:** its glass examples were CRC-VAL-HE-7K `BACK` tiles. These turned out to be
**purple**, not white, because NCT/CRC-VAL's color normalization tints empty glass. Real raw slides (our TCGA slides,
Kather) have white or gray glass. A detector trained on purple "glass" learns the wrong thing.

### The fix: background guard (`src/detectors.py`)
```python
frac = colorful_fraction(tile)            # share of pixels with HSV saturation > 20
if is_background(frac, tissue_label):     # frac < 0.10 and tissue_label != "fat"
    qc_ok, qc_reason = False, "background"
```
- Colorfulness (saturation) works for **white and gray** glass. Necrosis, debris and all other tissue are pink/purple
  (≥ 0.75 colorful even at the 1st percentile of real necrosis).
- **It never touches a tile the tissue model calls fat.**
- Kather: catches **619 of 625** glass tiles (99.0%; all 552 that were called necrosis). The only real tissue it removes
  is 23 fat tiles that `tissue8` had *already mislabeled* (as necrosis/mucus/muscle) and 1 debris tile. **No correctly
  labeled tile is removed.**
- CRC-VAL-HE-7K: removes **0** real necrosis tiles (its purple "glass" isn't caught, but raw slides don't have that).
- Optional extra: fill single dropped tiles that are surrounded by tissue with their neighbors' label (tears inside tissue).

Caveat: Kather "EMPTY" tiles are pure glass. Partly-glass edge tiles on real slides aren't in any test set, so check a
few on a real TCGA slide.

---

## Problem 2: scattered lymphocytes are missed

### What happens
- `tissue8` finds **dense sheets** of lymphocytes (98%) but misses **lymphocytes scattered through stroma or tumor**.
  186 of 214 misses come from patients 05 and 09, where Kather's tiles are inflamed stroma
  (`reports/kather2016/lymphocyte_examples.jpg`). In NCT / STARC-9, "lymphocytes" means *a tile that is mostly
  lymphocytes*, and inflamed stroma is labeled "stroma".
- **Why it matters:** the immune score (`src/features.py` → `immune()`) averages `tissue8`'s **P(lymphocytes)** per zone.
  On the missed tiles that's ~0.04, so **diffuse infiltrates barely count**. Diffuse infiltration at the front is what
  the Immunoscore idea is about.

### The fix: a separate immune detector (`models/immune_v1.joblib`)
A yes/no model on the same CONCH embeddings: "are lymphocytes / inflammatory cells present in this tile?" A tile can now
be **stroma or tumor AND lymphocyte-rich**, which the 8-class model can't express.

- **Training (31,010 positive / 38,905 negative tiles):**
  - Positives: NCT + STARC-9 lymphocyte tiles, plus **all 9,453 SPIDER "Inflammation" tiles from training slides**
    (colon 4,435 + skin 5,018), i.e. inflamed tissue with scattered immune cells.
  - Negatives: all other tissue.
  - Logistic regression, C = 1, picked by 5-fold cross-validation grouped by slide (balanced accuracy 0.988).
- **Compute:** SPIDER inflammation tiles streamed on Delta (CPU, 3–4 min each), embedded in one GPU job (2 min 49 s);
  trained on the laptop.

### Results

**Kather 2016** (never used; "missed" = the 214 lymphocyte tiles `tissue8` didn't call lymphocytes):

| score | mean: missed | mean: stroma / tumor / complex stroma / normal mucosa | flagged ≥ 0.5: missed | flagged: tumor / complex stroma | AUROC missed vs stroma / tumor |
|---|---|---|---|---|---|
| `tissue8` P(lymphocytes) (used now) | 0.04 | 0.002 / 0.011 / 0.015 / 0.002 | 0% | 1.0% / 0.8% | 0.892 / 0.931 |
| zero-shot CONCH | 0.70 | 0.024 / **0.120** / **0.285** / 0.084 | 72% | **11.2% / 28.5%** | 0.931 / 0.860 |
| immune detector v0 (625 inflammation tiles) | 0.37 | 0.006 / 0.029 / 0.067 / 0.018 | 36% | 2.6% / 3.7% | 0.949 / 0.939 |
| **immune detector v1** | **0.44** | **0.010 / 0.028 / 0.107 / 0.032** | **43%** | **2.4% / 7.0%** | **0.953 / 0.951** |

**SPIDER test slides** (inflamed vs all other classes): colon finds **97.0%** of inflamed tiles (5.5% false flags,
AUROC 0.991); skin **94.7%** (7.0%, AUROC 0.983). v0 found only 81% / 84%.

**Sanity checks** (mean p_immune per class): lymphocytes 0.97–0.98 everywhere; tumor ≤ 0.010; muscle, fat, mucus ≤ 0.006;
stroma 0.002–0.058 (CRC-VAL stroma is sometimes inflamed); normal mucosa 0.01–0.03. Normal mucosa naturally has immune
cells in it, so a little signal there is real.

### How to read it
- For the **immune score** (average per zone), what matters is contrast. Scattered-lymphocyte tiles now score **0.44**
  vs **0.01** for plain stroma and **0.03** for tumor, about 15–40×. With `tissue8` it was 0.04 vs 0.002–0.011.
- As a **yes/no flag** (≥ 0.5), it catches 43% of scattered-lymphocyte tiles. Some of Kather's "lymphocyte" tiles have
  only a few immune cells, so a probability makes more sense here than a hard yes/no.
- **Not solved:** counting cells. Tile scores say *where* immune cells are, not how many, and not CD8 vs CD4. A simple
  nucleus counter failed (dense lymphocytes merge); real cell counting needs a nucleus segmentation model.

### How to plug it in
1. Keep `tissue8`'s "lymphocytes" class for the tissue map. It marks lymphoid aggregates and lymph nodes
   (`features.lymph_nodes()` needs that).
2. Write `p_immune = predict_immune(load_immune("models/immune_v1.joblib"), emb)` as an extra per-tile column (e.g. in
   `structures.csv`). The tile map supports extra layers (`tm.extra`).
3. In `features.immune()`, average `p_immune` instead of P(lymphocytes) when it's present. Recalibrate `IMMUNE_LOW` /
   `IMMUNE_CORE_TO_MARGIN` in `src/cutoffs.py` (they're placeholders).

---

## Independent re-tests (data NOT used to design either fix; Kather left out)

### Lymphocyte detector

**TCGA-TILs** (Abousamra et al. 2022 / Saltz et al. 2018, Zenodo 6604094, CC-BY 4.0): **4,889 raw TCGA patches**
(rectum 4,811, colon 78) from **186 patients**, labeled TIL-positive (≥ 2 lymphocytes, 1,306) or negative.
Patches are 50 µm, smaller than our 112 µm tiles. Error bars come from resampling patients.

| score | AUROC [95% CI] | mean score: TIL+ / TIL− | flagged ≥ 0.5: TIL+ / TIL− |
|---|---|---|---|
| `tissue8` P(lymphocytes) (used now) | 0.749 [0.726–0.774] | 0.056 / 0.004 | 4.7% / 0.1% |
| zero-shot CONCH | 0.761 [0.737–0.788] | 0.511 / 0.169 | 51.3% / **16.4%** |
| immune detector v0 | 0.767 [0.744–0.792] | 0.118 / 0.010 | 9.4% / 0.2% |
| **immune detector v1** | **0.831 [0.809–0.853]** | **0.165 / 0.010** | 13.9% / 0.3% |

- **v1 is clearly the best** (its interval doesn't overlap `tissue8`'s). TIL-positive patches score **16× higher** than
  negatives, with almost no false flags (0.3%; zero-shot: 16%).
- As a hard yes/no it catches only 14% of positives, because "≥ 2 lymphocytes" in a 50 µm patch is a very low bar.
  Use the continuous score, which is how `features.immune()` averages it anyway.

**PathGen captions** (weak, AI-written; 1,908 patches after dropping 233 negated mentions like "no inflammation"):
caption mentions lymphocytes / inflammation in 876. AUROC: immune v1 **0.65** vs `tissue8` 0.56 (all organs); colorectal
0.65 vs 0.63. The signal goes the same direction but is weak, as expected from noisy captions.

### Background guard
| test | result |
|---|---|
| **CRC-VAL-HE-7K** (6,333 tissue tiles) | removes **1** tile (a fat tile already mislabeled); no correctly labeled tile |
| **NCT-CRC-HE-100K** (89,434 tissue tiles) | removes **43 (0.05%)**: 26 muscle, 11 necrosis, 5 mucus, 1 normal; 34 were correctly labeled (very pale tiles). Tumor, stroma, fat, lymphocytes: 0 |
| NCT `BACK` (glass) | normalization turned it purple (median 100% "colorful"), so the guard can't see it. Not relevant for raw slides |
| **our 5 TCGA slides, current mask** | removes **0** of 9,845 tested tiles: no glass leaks through |

### New finding: the tissue mask drops pale real tissue
On the TCGA slides, `tissue_mask()` (saturation > max(Otsu, 20)) lands at Otsu 31–70, so it throws away **pale real
tissue**: loose submucosa, edematous stroma, and probably fat (`reports/guard_check/TCGA-AA-3875-01Z-00-DX1_mask_sat12.jpg`,
green = tissue a looser mask recovers). That hides fat near the tumor ("touches fat" T3 hint).

| looser fixed mask (saturation > 12) + guard, 5 TCGA slides | result |
|---|---|
| extra tissue kept | **+12–26%** on 4 of 5 slides (+2% on the 5th) |
| what's added | tumor and tissue edges, pale stroma (e.g. 140 stroma tiles on AA-3875), mucin, a few fat tiles |
| guard on the added tiles | removed **21**, all pure glass with dust (0–6% colored) that `tissue8` had called mucus/necrosis (`..._new_flagged.jpg`) |
| side effect | added edge tiles are ~50% glass, and "mucus" goes from 3.5% to 7.6% of sampled tiles |

**Recommendation:** use the looser mask (saturation > 12) **plus** the guard. Keep part-glass edge tiles for the map
(shape, borders), but **leave tiles under ~30% colored out of area percentages** (tumor %, mucinous fraction).

## Order of work

| step | effort | where |
|---|---|---|
| looser tissue mask (saturation > 12) + background guard when tiling (`colorful_fraction` + `is_background`); skip < 30% colored tiles in area fractions | 20 min | `src/slides.py` / tiling + `src/features.py` (teammates' code on `main`; coordinate) |
| `p_immune` column + `features.immune()` uses it | 20 min + recalibrate cut-offs | `src/features.py` (teammates' code) |
| check on a real TCGA slide: edge tiles, inflamed front | 30 min | after the pipeline runs end to end |
