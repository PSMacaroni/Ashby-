# Two separate models: tissue (8 classes) + structures (nerve / vessel / blood)

**Written:** Sat Oct 17 · **Run:** `split_v1` (trained on the laptop, CPU, a few minutes) · **Code:** `scripts/models/train_split_models.py`
**Models:** `models/tissue8_split_v1.joblib`, `models/structures_split_v1.joblib` · **Results:** `reports/split_v1/`

## TL;DR

We split the job into two models that both run on the same CONCH embeddings:

1. **`tissue8`**: **only our 8 tissue classes**, trained on **NCT-CRC-HE-100K + STARC-9 together**
   (169,434 tiles, two hospitals).
2. **`structures`**: **nerve / vessel / blood / other**. Blood moved here from the old tissue model and is still
   "identify and ignore". Trained on SPIDER + STARC-9 blood (38,941 tiles).

**Result: the overall accuracy drop is gone.** On four test sets from four sources, `tissue8` is the best or within
0.5 points of the best everywhere. That includes a hospital it never saw (SPIDER), where it's the best of all.

| | zero-shot CONCH | STARC-only (old) | NCT-only | **tissue8 (NCT + STARC)** |
|---|---|---|---|---|
| **average accuracy, 4 test sets** | 81.1% | 92.2% | 82.5% | **94.6%** |
| **worst test set** | 76.5% | 86.2% | 74.4% | **89.1%** |

## Why the split fixes it

The accuracy drop wasn't caused by nerves or blood sharing a model with the tissue classes. It came from **training on
one hospital**: the STARC-9-only model learned Stanford's way of drawing muscle vs stroma, and the NCT-only model learned
Heidelberg's. Training the tissue model on **both** fixes that. Moving blood into the structures model also makes the
tissue model a clean 8-class model whose probabilities always sum to 1 over our classes.

## Model 1: `tissue8` (8 classes only)

**Training:** logistic regression (C = 10, class-balanced) on 89,434 NCT-CRC-HE-100K tiles + 80,000 STARC-9 tiles (no
blood). C was picked on two tuning sets, STARC-9 VAL-SMALL (other Stanford slides) and a random 10% NCT hold-out. The
hold-out was only used to choose C; the final model uses all of NCT.

**Tests** (patients never used in training; accuracy / balanced accuracy):

| test set | tiles | zero-shot | STARC-only | NCT-only | **tissue8** |
|---|---|---|---|---|---|
| CRC-VAL-HE-7K (Heidelberg, 50 new patients) | 6,333 | 86.8% / 0.861 | 86.2% / 0.776 | 95.8% / 0.942 * | **95.5% / 0.939** |
| STARC-9 VAL-LARGE (Stanford, 50 new slides) | 48,000 | 82.5% / 0.825 | 97.7% / 0.977 * | 81.5% / 0.815 | **97.6% / 0.976** |
| STARC-9 TCGA (30 TCGA slides) | 18,500 | 78.7% / 0.791 | 96.9% / 0.969 * | 74.4% / 0.746 | **96.4% / 0.963** |
| SPIDER colon (third source, never trained on) | 3,500 | 76.5% / 0.809 | 88.0% / 0.892 | 78.6% / 0.792 | **89.1% / 0.903** |

\* = that model is trained on the same hospital's data (different patients), so it has a home advantage there.

**Per class, tissue8:**

| class | CRC-VAL-HE-7K | Stanford | TCGA | SPIDER colon |
|---|---|---|---|---|
| tumor | 96.5% | 98.2% | 97.2% | 81.4% |
| normal mucosa | 99.5% | 96.8% | 99.1% | 98.6% |
| muscle | 72.8% | 94.5% | 92.0% | 87.0% |
| fat | 99.9% | 98.7% | 94.8% | 98.8% |
| lymphocytes | 98.4% | 98.8% | 99.4% | – |
| mucus | 99.5% | 99.4% | 98.2% | – |
| necrosis | 93.8% | 96.8% | 98.4% | 91.0% |
| stroma | 91.0% | 97.4% | 91.7% | 85.2% |

Compared with zero-shot, **tumor** goes from 50–64% to 81–98% and **stroma** from 32–64% to 85–97%. Those were the
two weakest zero-shot classes.

**Calibration:** log-loss 0.08–0.33 and ECE 0.003–0.022 on all four tests (zero-shot: ECE 0.10–0.18). The hover
probabilities are now honest.

**Remaining weak spots:**
- **Muscle → stroma on CRC-VAL-HE-7K** (161 of 592 muscle tiles). This is the last bit of the hospital difference.
- **SPIDER tumor 81.4%**: 127 tumor tiles called necrosis and 38 called stroma. SPIDER "adenocarcinoma" tiles often
  include dirty necrosis and desmoplastic stroma, so some of this is about how the tiles were labeled.
- **Note on SPIDER "Mucus":** those tiles are **normal crypts full of goblet cells** (we checked by eye), so they're
  scored as normal mucosa. Scored as "mucus", every model gets ~0% on them.

### Extra test: Kather 2016 textures (not color-normalized, 10 Mannheim patients)

`data/datasets/kather2016/` (Zenodo 53169, CC-BY 4.0): 5,000 tiles of 150 px at 0.495 µm/px, 625 per class, **no color
normalization**. That makes it the closest of our test sets to raw slides. Its classes are coarser, so scoring is:
TUMOR → tumor, STROMA → stroma **or** muscle (their "simple stroma" includes smooth muscle), LYMPHO → lymphocytes,
DEBRIS → necrosis **or** mucus (or blood for the old model), MUCOSA → normal mucosa, ADIPOSE → fat.
COMPLEX and EMPTY are not scored. Error bars come from resampling **patients** (only 10), so they're wide.

| model | accuracy [95% CI] | strict* | tumor | stroma | lymph | debris | mucosa | fat |
|---|---|---|---|---|---|---|---|---|
| zero-shot | 76.2% [65–85] | 61.1% | 45.9% | 87.8% | **90.6%** | 55.7% | 82.7% | 94.4% |
| STARC-only | 80.6% [73–87] | 64.4% | 91.0% | 77.0% | 65.8% | 62.9% | 97.1% | 89.6% |
| NCT-only | 84.1% [75–92] | 68.9% | 92.6% | 94.2% | 63.8% | 60.2% | 98.2% | 95.7% |
| **tissue8** | **85.6% [76–94]** | **70.3%** | **94.7%** | **94.7%** | 65.8% | **64.3%** | **98.2%** | **95.7%** |

\*strict = STROMA must be "stroma" and DEBRIS must be "necrosis".

- `tissue8` is again the best overall, but **this set is harder (86%)** than the normalized ones (89–98%).
- **Lymphocytes (66%):** the errors are almost all from patients 05 and 09 (186 of 214). In those tiles Kather's
  "LYMPHO" means **lymphocytes scattered inside stroma or tumor**, and our models call them stroma / tumor / muscle.
  Dense lymphocyte sheets are recognized (`reports/kather2016/lymphocyte_examples.jpg`). **This matters for the immune
  analysis:** tile-level labels catch dense lymphoid aggregates but undercount diffuse infiltrates. Zero-shot handles
  them better (91%), so a cell-level lymphocyte count or a zero-shot "inflamed stroma" score may be needed.
- **Debris (64%):** a mixed bag (necrosis + hemorrhage + mucus). Wispy pink fibrous debris is often called stroma. Patient
  08 (54%) is mostly this kind of debris (`patient08_examples.jpg`).
- **EMPTY (glass) is called necrosis 88% of the time** by every trained model. The pipeline's tissue mask must remove
  background before classification, and stray glass tiles would inflate "necrosis".
- Caveat: Kather 2016 and NCT-CRC-HE-100K come from the same institution, so patient overlap with NCT is possible
  (not documented).

## Model 2: `structures` (nerve / vessel / blood / other)

**Training:** logistic regression (C = 10, class-balanced; C picked by 5-fold cross-validation grouped by slide, balanced
accuracy 0.981).
- SPIDER colon + skin: nerve 3,960, vessel 7,451, other 9,530 tiles.
- STARC-9: blood 10,000, plus 8,000 tissue tiles as extra "other".

**Tests:**

| test | result | old models |
|---|---|---|
| SPIDER colon test slides | vessel found 97.5% (precision 97.5%), balanced accuracy 0.967 | 96.9% / 0.966 |
| SPIDER skin test slides | nerve found **99.7%** (precision 99.4%), vessel 98.7%, balanced accuracy 0.987 | 98.8% / 0.986 |
| blood found, Stanford / TCGA | **99.5% / 94.5%** (AUROC 0.999 / 0.994) | 98.8% / 95.4% |
| NCT muscle tiles flagged vessel / nerve | **7.3% / 0.9%** | 13.0% / 1.9% |
| NCT stroma tiles flagged vessel | **4.1%** | 5.9% |
| tumor / normal mucosa flagged as anything | ≤ 0.04% | ≤ 0.2% |

**Weak spot, blood vs fat:** 7.6% (Stanford) and **16%** (TCGA) of fat tiles are called blood. In the pipeline, ignoring
those tiles would hide fat, and the "tumor touches fat" T3 hint depends on fat. **Suggested rule:** ignore a tile as
blood only if p_blood ≥ 0.9 **and** the tissue model doesn't say fat. Also, 51% of CRC-VAL-HE-7K "necrosis" (NCT's
"debris", which includes hemorrhage) is called blood. That's partly correct.

## How to use them in the pipeline

```python
from src.classifier import load_classifier, predict
from src.structures import load_structures, predict_structures
probs8, _ = predict(load_classifier("models/tissue8_split_v1.joblib"), emb)          # 8 classes, rows sum to 1
s = predict_structures(load_structures("models/structures_split_v1.joblib"), emb)    # p_other/p_nerve/p_vessel/p_blood
skip = (s["p_blood"] >= 0.9) & (probs8.argmax(1) != 3)                                # 3 = fat; tiles to ignore
```

Write the 8 tissue probabilities to `tiles.csv` and the structure probabilities to `structures.csv` (data contract in CLAUDE.md).

## Caveats

- All training and test tiles are **color-normalized** (NCT, STARC-9) or from curated datasets. Our raw TCGA slides are
  the real test: hand-label ~100 tiles.
- Nerves are only tested on skin. Artery vs vein is still zero-shot only.
- **MUSK is not included.** Our teammate's zero-shot test showed CONCH + MUSK averaged beats either alone. A `tissue8`
  trained on CONCH + MUSK embeddings could do better still. That needs a team member with MUSK access (Delta user
  `natmakur`) to embed the same tiles.

## Reproduce

Needs CONCH embeddings (from `scripts/models/embed_tiles.py`) for `nct_100k`, `starc9_train/val_small/val_large/val_tcga`,
`crc_val_7k`, `spider_{colon,skin}_{train,test}`, plus the SPIDER `manifest.csv` files:

```bash
python scripts/models/train_split_models.py --run split_v1
```
