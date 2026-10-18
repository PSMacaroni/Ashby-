# Report: our models on the two provided datasets (NCT-CRC-HE-100K and PathGen)

**Written:** Sat Oct 17, 2026 · **Run:** `provided_v1` · **Compute:** Delta jobs 22956414 + 22956452 (A100, ~10 min, ≈0.17 GPU-hours)
**Code:** `scripts/eval_provided.py`, `scripts/eval_provided_gpu.sbatch` · **Outputs:** `data/results/eval_provided/provided_v1/`
(on Delta and copied to the laptop: `metrics.json`, confusion matrices, per-tile/per-patch prediction CSVs, example image sheets)

---

## 1. Summary

| question | answer |
|---|---|
| Does the trained tissue classifier recognize the 8 classes on data it never saw? | **Partly.** Tumor 92.6% (zero-shot 73.5%) and normal mucosa 94.7% (81.6%) are much better. Muscle 60.8% (87.8%) and stroma 25.4% (67.1%) are much worse. Overall 80.2% vs zero-shot 86.9%. |
| Why the split? | The training set (STARC-9, Stanford) and NCT (Heidelberg) define and stain muscle/stroma differently. The classifier learned STARC-9's version. |
| Does the nerve/vessel side classifier behave sensibly on unseen data? | **Yes, with one weak spot.** It never flags tumor or normal mucosa. Many "vessel" flags inside muscle/stroma tiles are real vessels. Flags agree with PathGen captions far above chance. Smooth muscle vs nerve is the weak spot. |
| Can we trust the confidence values? | Classifier on NCT: when it is ≥99% sure (58% of tiles), it's right 95.5% of the time. Below 50% confidence it's right only 28% of the time. |
| What next? | (1) Train the tissue classifier on NCT + STARC-9 together. (2) Have a pathologist review ~50 nerve flags. (3) Hand-label ~100 tiles from our own TCGA slides as the final check. |

---

## 2. The models under test (neither was trained on these datasets)

| model | file | trained on | earlier test results (held-out patients) |
|---|---|---|---|
| tissue classifier: logistic regression on CONCH embeddings, 8 classes + blood | `data/models/conch_lr_starc9_lr.joblib` | STARC-9: 90,000 tiles, ~200 Stanford patients (`docs/dataset_starc9.md`) | Stanford 97.7%, TCGA 96.9%, CRC-VAL-HE-7K 86.2% |
| nerve / vessel / other side classifier | `data/models/conch_structures_v1.joblib` | SPIDER: 20,941 tiles, 3,097 slides, colon + skin (`docs/structures_classifier.md`) | colon test 96.6%, skin test 98.6% balanced accuracy |
| baseline: zero-shot CONCH | text prompts in `src/prompts.py` (v1) | nothing (no training) | |

Both classifiers use the same frozen CONCH ViT-B/16 image embeddings (224 px input, tiles at ~0.5 µm/px).

---

## 3. Dataset 1: NCT-CRC-HE-100K

### 3.1 What it is

- 100,000 colorectal H&E tiles, 224 × 224 px at 0.5 µm/px, Macenko color-normalized, from **86 slides** (NCT Biobank /
  UMM pathology archive, Heidelberg; Kather et al. 2019). Provided on Delta at
  `/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/`.
- 9 classes. We use 8 and skip `BACK` (empty glass, which our tissue mask removes). That's **89,434 tiles** tested.
- Each tile's label is its folder. **No patient IDs**, so it can only be a test set for models never trained on it, as here.

| class (NCT folder) | tiles |
|---|---|
| tumor (TUM) | 14,317 |
| muscle (MUS) | 13,536 |
| lymphocytes (LYM) | 11,557 |
| necrosis (DEB, "debris") | 11,512 |
| stroma (STR) | 10,446 |
| fat (ADI) | 10,407 |
| mucus (MUC) | 8,896 |
| normal mucosa (NORM) | 8,763 |

### 3.2 Method

1. CONCH embedded all 89,434 tiles on a Delta A100 (211 s, 425 tiles/s). Saved as `data/embeddings/nct_100k.npz` on Delta.
2. **Zero-shot:** each tile gets the class whose averaged text prompts are closest.
3. **Classifier:** the STARC-9 logistic regression predicts 9 labels. A tile predicted "blood" counts as **wrong**, because
   the pipeline would hide it.
4. We compare each prediction with the folder label. The model never sees the label.

### 3.3 Overall results

| metric | zero-shot | classifier |
|---|---|---|
| accuracy | **86.9%** | 80.2% |
| balanced accuracy (each class weighted equally) | **87.3%** | 80.3% |
| log-loss (lower = better probabilities) | **0.83** | 0.88 |
| calibration error, ECE (0 = perfect) | **0.098** | 0.114 |
| tissue tiles wrongly hidden as "blood" | 0% | 0.8% |

### 3.4 Per class

| class | zero-shot recall | zero-shot precision | zero-shot F1 | classifier recall | classifier precision | classifier F1 |
|---|---|---|---|---|---|---|
| tumor | 73.5% | 98.1% | 0.84 | **92.6%** | 93.1% | **0.93** |
| normal mucosa | 81.6% | 79.0% | 0.80 | **94.7%** | 80.9% | **0.87** |
| lymphocytes | 99.8% | 90.9% | 0.95 | 96.9% | 99.4% | **0.98** |
| fat | 94.8% | 89.3% | 0.92 | 91.7% | 98.0% | **0.95** |
| mucus | 97.7% | 75.8% | 0.85 | 86.4% | 88.6% | **0.88** |
| necrosis | 96.4% | 91.7% | **0.94** | 94.1% | 65.1% | 0.77 |
| muscle | 87.8% | 82.7% | **0.85** | 60.8% | 73.8% | 0.67 |
| stroma | 67.1% | 87.8% | **0.76** | 25.4% | 39.5% | 0.31 |

*Recall* = of the tiles that really are this class, how many were found. *Precision* = of the tiles called this class,
how many really are. *F1* combines both.

The classifier has the better F1 for **5 of 8 classes** (tumor, normal mucosa, lymphocytes, fat, mucus). Zero-shot is
better for **necrosis, muscle and stroma**.

### 3.5 Biggest mistakes

| true → predicted | zero-shot | classifier |
|---|---|---|
| tumor → normal mucosa | **1,885** | 654 |
| tumor → mucus | 1,006 | 31 |
| normal mucosa → mucus | 935 | 111 |
| normal mucosa → lymphocytes | 617 | 19 |
| muscle → stroma | 493 | **3,406** |
| muscle → fat | 933 | 132 |
| muscle → normal mucosa / necrosis | 14 / 49 | 795 / 716 |
| stroma → muscle | 2,286 | **2,857** |
| stroma → necrosis | 546 | **3,516** |
| mucus → necrosis | 72 | 928 |
| necrosis → blood | n/a | 462 |

- The classifier **removes zero-shot's tumor/normal/mucus confusions**, which matter most for tumor % and the tumor border.
- It **adds muscle/stroma/necrosis confusions**, the same pattern as on CRC-VAL-HE-7K (also Heidelberg). STARC-9's
  "fibroconnective tissue" and NCT's "stroma" are different definitions, and the two datasets are stain-normalized
  differently. Necrosis → blood is partly expected: NCT's "debris" class includes hemorrhage.

### 3.6 Can we trust the confidence values? (classifier)

| shown confidence | share of tiles | accuracy |
|---|---|---|
| ≥ 99% | 57.8% | **95.5%** |
| ≥ 90% | 77.3% | 90.3% |
| ≥ 80% | 84.1% | 87.5% |
| ≥ 50% | 97.4% | 81.6% |
| < 50% | 2.6% | 28.0% |

High confidence is meaningful, but on this dataset the classifier is somewhat overconfident (ECE 0.114). On its own
Stanford test sets, ECE was 0.001. The app could grey out tiles below ~80% confidence.

### 3.7 Nerve / vessel flags on NCT (no labels, so these are sanity checks)

| NCT tissue | called nerve | called vessel |
|---|---|---|
| tumor | 0.0% | 0.2% |
| normal mucosa | 0.0% | 0.0% |
| lymphocytes | 0.0% | 0.6% |
| necrosis | 0.0% | 1.5% |
| fat | 0.2% | 1.3% |
| mucus | 2.0% | 2.5% |
| stroma | 1.0% | 5.9% |
| muscle | 1.9% | 13.0% |
| **all** | **0.6%** | **3.3%** |

- **Tumor and normal mucosa are essentially never flagged**, which is what a perineural/vascular-invasion check needs.
- **Vessel flags:** the 12 strongest (`nct_top_vessel.png`) mostly show thin walls around open spaces, several with
  **red blood cells inside**. These are real vessels inside tiles labeled by their main tissue, so many "muscle/stroma →
  vessel" flags are probably correct.
- **Nerve flags:** the 12 strongest (`nct_top_nerve.png`, mostly NCT "muscle") are pale, wavy fiber bundles. Some look
  nerve-like, and some may be smooth muscle. **Unresolved without a pathologist.**

---

## 4. Dataset 2: PathGen (2k sample)

### 4.1 What it is

- **2,141 patches** from **10 TCGA slides** of several cancer types, with **AI-written captions** (the PathGen pipeline).
  Provided at `/projects/bfqi/shared/datasets/pathgen_dataset/pathgen_2k_output/` (`<file_id>_<x>_<y>.png/.txt`).
- **3 slides are colorectal, and they are our own TCGA slides:** TCGA-AA-3844 (COAD, T3N2), TCGA-CM-4746 (COAD, T2N0),
  TCGA-AH-6547 (READ, T3N0). That's 578 patches.
- Patches are 672 × 672 px at **~0.25 µm/px** (40x; we measured red blood cells at ~30 px), about 168 µm across.
- **No real labels.** Captions are the only "answer key". They describe the whole patch, are machine-written, and
  sometimes negate ("no vascular pathology"). Results here measure **agreement with captions, not accuracy**.

### 4.2 Method

1. Each patch is covered with 2 × 2 overlapping windows of 448 px (112 µm), each shrunk to 224 px (0.5 µm/px) to match our
   tiles. That gives 4 tiles per patch, 8,564 in total.
2. Each tile goes through both classifiers. Per patch, we keep the **highest** nerve/vessel probability, whether **any**
   tile was flagged, the **tumor fraction** (tiles called tumor), and the most common tissue.
3. Caption keywords used as the answer key: nerve (`nerv…, perineural, neural, ganglion`), vessel (`vessel, vascular,
   capillar…, arteri…, vein, venule, endotheli…`), tumor (`carcinoma, tumor, malignan…, neoplas…, cancer`).

| caption mentions | all 10 slides | 3 colorectal slides |
|---|---|---|
| vessel | 333 patches | 40 |
| nerve | 35 | 1 |
| tumor | 1,377 | 456 |

### 4.3 Nerve / vessel vs captions

| | AUROC | flagged when caption mentions it | flagged when it doesn't |
|---|---|---|---|
| vessel, colorectal | 0.74 | 32.5% | 7.1% (4.6× less) |
| vessel, all slides | 0.65 | 34.5% | 12.1% |
| nerve, all slides | 0.82 | 14.3% | 0.6% (24× less) |
| nerve, colorectal | n/a | 1 mention only | 0.9% flagged |

AUROC = how well the probability ranks "mentioned" above "not mentioned" (0.5 = coin flip, 1.0 = perfect).

**Changing the cut-off** (patch flagged if any tile's probability ≥ threshold):

| | threshold 0.5 | 0.2 | 0.1 |
|---|---|---|---|
| vessel, colorectal: caught / clean patches kept clean | 33% / 93% | 50% / 87% | 57% / 83% |
| nerve, all slides: caught / clean patches kept clean | 14% / 99% | 26% / 99% | 29% / 98% |

- Both structures are flagged **far more often where captions mention them**, so the signal transfers to real TCGA patches.
- At the default cut-off the side classifier is **cautious**: few false alarms, but it misses many caption mentions. A
  caption may mention a vessel at the patch edge that our 4 windows barely show, so true sensitivity is uncertain.
- Most nerve captions come from **non-colon** slides (e.g. kidney). For colon nerves, the honest answer is still **not
  tested**.
- For the colorectal patches it rates most nerve-like, the captions describe **smooth muscle bundles**. That matches
  section 3.7: smooth muscle is the main nerve confusion.

### 4.4 Tissue classifier vs captions (colorectal)

| | mean tumor fraction |
|---|---|
| caption mentions tumor (456 patches) | **0.81** |
| caption doesn't (122 patches) | 0.52 |
| AUROC | 0.68 |

### 4.5 Our own three TCGA slides (anecdotal; PathGen picked these patches, not us)

| slide | stage | patches | main tissue of patches (classifier) | tumor fraction | patches with a vessel flag | with a nerve flag |
|---|---|---|---|---|---|---|
| TCGA-AA-3844 (COAD) | T3N2 | 182 | tumor 172, necrosis 6, normal 4 | 0.90 | 5.5% | 0% |
| TCGA-CM-4746 (COAD) | T2N0 | 240 | tumor 126, **muscle 40**, necrosis 33, stroma 19, normal 11, mucus 8 | 0.51 | 13.8% | 2.1% |
| TCGA-AH-6547 (READ) | T3N0 | 156 | tumor 153, necrosis 2, normal 1 | 0.94 | 5.1% | 0% |

CM-4746 is T2, meaning the tumor grows **into the muscle layer**, and it's the slide where many patches are mostly
muscle. That fits, but it's 3 slides and patches chosen by someone else, so it's **not evidence**, only a reason to look
at this slide first when we run the full pipeline.

---

## 5. Caveats

1. **NCT is color-normalized** like STARC-9. Our raw TCGA slides are not, so real-slide accuracy may differ in either direction.
2. **NCT labels are tile-level "main tissue".** A muscle tile with a vessel in it is still "muscle", so nerve/vessel
   "false alarms" there may be correct.
3. **PathGen captions are AI-written** and describe a larger area than our windows. Treat section 4 as agreement, not accuracy.
4. **Colon nerves remain untested.** No provided or public dataset labels them.
5. **Artery vs vein was not evaluated here.** There are no labels anywhere (see `docs/structures_classifier.md`).
6. **MUSK was not tested.** Access to `xiangjx/musk` is still awaiting manual approval.

## 6. Recommendations

1. **Tissue map:** don't switch the pipeline to the STARC-only classifier yet. Train on **NCT + STARC-9 together**
   (embeddings for both exist; a CPU job of seconds). Test on CRC-VAL-HE-7K, Stanford and TCGA tests, never on NCT
   itself. Expected outcome: keep the tumor/normal gains and recover muscle/stroma.
2. **Confidence in the app:** show tiles below ~80% confidence as "uncertain".
3. **Nerve flags:** have a pathologist mentor review the ~50 strongest nerve flags on NCT/colon (smooth muscle vs nerve).
   Consider requiring p_nerve ≥ 0.9 for the perineural-invasion check.
4. **Final validation:** hand-label ~100 tiles from our 5 TCGA slides (the plan in CLAUDE.md). That's the only test on
   un-normalized slides like the ones the app will show.

## 7. Files

| file | what |
|---|---|
| `metrics.json` | every number above |
| `confusion_nct_zero_shot.png/.csv`, `confusion_nct_classifier.png/.csv` | 8-class confusion matrices on NCT |
| `predictions_nct.csv.gz` | per NCT tile: true label, prediction, 8 probabilities, blood flag, p_nerve, p_vessel |
| `nct_top_nerve.png`, `nct_top_vessel.png` | the 12 NCT tiles most strongly flagged as nerve / vessel |
| `predictions_pathgen.csv` | per PathGen patch: caption flags, max p_nerve / p_vessel, tumor fraction, main tissue, caption |
| `pathgen_top_nerve.png`, `pathgen_top_vessel.png` | the 12 PathGen patches scored most nerve- / vessel-like (with whether the caption agrees) |
