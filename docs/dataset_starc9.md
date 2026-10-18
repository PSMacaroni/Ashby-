# Second training dataset: STARC-9

**Written:** Sat Oct 17 · **For:** the small-classifier work in `docs/handoff_small_classifier.md`
**Scripts:** `scripts/models/get_starc9.py` (download), `scripts/models/embed_tiles.py` (CONCH embeddings),
`scripts/models/train_classifier.py` (train + test), `src/classifier.py` (use in the pipeline)
**Output:** `data/datasets/starc9/`, `data/embeddings/`, `data/results/classifier/`, `data/models/` (all git-ignored)

## TL;DR

NCT-CRC-HE-100K (our current training set) is from **86 slides at one hospital**, and zero-shot CONCH on it is weakest on
**tumor → normal mucosa** and **stroma → muscle**. **STARC-9** is a newer (NeurIPS 2025) colon tile dataset from
**~200 different patients at Stanford**, with the **same tissue classes plus blood**, and three test sets from other
patients, one of them cut from **TCGA** slides like ours. We download a random subset of it at our pipeline's scale
(0.5 µm/px) and use it to train and, more importantly, to *test honestly on new patients*.

## What we downloaded

| folder | tiles | from | use it for |
|---|---|---|---|
| `train/` | 10,000 per class × 9 = 90,000 (random, seed 0) | STARC-9 training set (70,000/class, ~200 Stanford patients) | **training** |
| `val_stanford_small/` | 18,000 (2,000/class) | 20 other Stanford slides | **tuning** (pick `C`) |
| `val_stanford_large/` | 54,000 (6,000/class) | 50 other Stanford slides | **final test** |
| `val_tcga/` | 20,000 (1,500–2,500/class) | 30 TCGA colon slides | **final test**, closest to our slides |

Total 182,000 tiles, 6.1 GB. Each folder has one subfolder per class (our names) and a `manifest.csv`
(`path, set, label, source_class, source_member`). `dataset_info.json` records the settings. All tiles are
**128 × 128 px PNG at 0.5 µm/px**. We also downloaded **CRC-VAL-HE-7K** (`data/datasets/CRC-VAL-HE-7K/`, NCT-CRC's
separate-patient test set from another hospital) as a cross-dataset test.

Class mapping (`src/datasets.py` → `STARC_TO_CLASS`):
`TUM→tumor, NOR→normal_mucosa, MUS→muscle, ADI→fat, LYM→lymphocytes, MUC→mucus, NCS→necrosis, FCT→stroma, BLD→blood`.

**Blood is not a map class.** The classifier learns to recognize blood only so the pipeline can **ignore** those tiles
(`qc_ok = False, qc_reason = "blood"`) instead of calling them necrosis. The 8 tissue probabilities are re-normalized
without blood, so the data contract stays at 8 classes.

## First results (Sat Oct 17, laptop RTX 5090, run `starc9_lr`)

Logistic regression on CONCH embeddings (224 px), trained on the 90k STARC-9 `train/` tiles, C = 10 picked on
`val_stanford_small`, each test set used once. Accuracy on tissue tiles (blood tiles scored separately):

| test set (new patients) | zero-shot | classifier | tumor (zs → clf) | stroma (zs → clf) |
|---|---|---|---|---|
| `val_stanford_large` (48k) | 82.5% | **97.7%** | 63.5% → 98.2% | 32.4% → 99.3% |
| `val_tcga` (18.5k, TCGA slides) | 78.7% | **96.9%** | 52.6% → 96.6% | 31.7% → 97.9% |
| CRC-VAL-HE-7K (6.3k, other hospital) | 86.8% | 86.2% | 63.5% → 90.9% | 59.6% → **12.4%** |

- **Blood:** the classifier flags 95–99% of blood tiles as blood. Zero-shot calls them necrosis (51–65%) or
  lymphocytes/fat. It also wrongly flags 0.4–2.6% of tissue tiles as blood (mostly fat in `val_tcga`).
- **Calibration:** log-loss 0.91 → 0.06 and ECE 0.13 → 0.001 on `val_stanford_large`, so hover probabilities become
  meaningful.
- **Caveat (important):** on CRC-VAL-HE-7K (NCT-CRC's hospital and labeling), overall accuracy doesn't improve.
  NCT "stroma" is called muscle (147/421) or necrosis (167/421), and NCT "debris" is often called blood (145/339), but
  NCT's debris class includes hemorrhage. Part of this is that the two datasets define the classes differently. The
  other part is that a classifier trained on one dataset learns that dataset's look. Tumor and normal mucosa still
  improve there (63.5% → 90.9%, 88.9% → 98.1%).
- **Next step:** train on NCT-CRC-HE-100K + STARC-9 together (experiment C below) and check all three tests again,
  then check on our own TCGA slides.

Files: `data/results/classifier/starc9_lr/` (`results.md`, `metrics.json`, confusion matrices PNG + CSV) and
`data/models/conch_lr_starc9_lr.joblib`. In the pipeline: `src/classifier.py` → `predict(clf, embeddings)` returns
8 tissue probabilities plus an `is_blood` flag.

## Why it should help

1. **More patients, more variety.** ~200 patients versus 86 slides. The authors picked tiles by clustering, so that
   unusual-looking tissue is included, and pathologists checked them.
2. **Targets our two mistakes.** It has 70,000 tumor, normal mucosa, fibroconnective tissue (stroma) and muscle tiles each.
3. **Blood class.** Fixes "blood → necrosis", which NCT-CRC can't fix because it has no blood class.
4. **Honest testing.** The three test sets come from patients that are not in the training set. NCT-CRC has no patient IDs,
   so this is the only way to get a real "new patients" number. Testing on `val_tcga` is the closest thing we have to our
   real TCGA slides.

## Experiments

Train three logistic regressions and pick `C` on `val_stanford_small` only:
- **A:** NCT-CRC only (the handoff plan)
- **B:** STARC-9 `train/` only (**done**: run `starc9_lr`, above)
- **C:** NCT-CRC + STARC-9 together: `--train nct_100k starc9_train`

Test each one **once** on CRC-VAL-HE-7K, `val_stanford_large` and `val_tcga`. A model that does well on a dataset it was
*not* trained on is the strongest slide for judges.

## Caveats (be honest about these on slides)

- **Still color-normalized.** The public STARC-9 tiles are Macenko-normalized, like NCT-CRC-HE-100K. Real slides vary more,
  so real-slide accuracy will be lower. Normalization also makes **blood look purple instead of red** in these tiles, so the
  blood detector may miss bright red blood on our raw slides. Check this on our TCGA slides.
- **Smaller field of view.** STARC-9 tiles are 256 px at 0.25 µm/px = **64 µm** across. We shrink them to 0.5 µm/px so cells
  are the same size as in our pipeline, but each tile still shows less tissue than an NCT-CRC tile (112 µm) or our pipeline
  tiles.
- **No patient IDs in the training set**, so we can't split it by patient. That's why tuning and testing use the separate
  test sets, never a random split of `train/`.
- **`val_tcga` might include our 5 TCGA slides** (the slide list isn't published). Use it **only for testing, never training**,
  so it can't leak into our slide-level results.
- "Fibroconnective tissue" (STARC-9) and "stroma" (NCT-CRC) are not defined the same way. The CRC-VAL-HE-7K result shows
  this matters.
- License: **CC-BY 4.0**, so we must cite it (below).

## How to reproduce

On a laptop with an NVIDIA GPU (what we did). The download takes ~30 min at ~110 tiles/s and is resumable: re-run the
same command if it stops.

```bash
uv venv --python 3.12 .venv     # 3.12 like Delta; newer Pythons lack ready-made builds of some CONCH dependencies
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe pillow tqdm numpy pandas scikit-learn "matplotlib>=3.11.2" \
    python-dotenv huggingface_hub timm transformers "conch @ git+https://github.com/Mahmoodlab/CONCH.git"
.venv/Scripts/python scripts/models/get_starc9.py --workers 32
```

Quick test first (~30 s): `--train-per-class 5 --val-per-class 5 --out data/datasets/starc9_test`.
CONCH needs `HF_TOKEN` in `.env`.

Embed (a few minutes per set on a laptop GPU, ~250–380 tiles/s), then train (seconds, CPU):

```bash
.venv/Scripts/python scripts/models/embed_tiles.py --root data/datasets/starc9/train --name starc9_train
.venv/Scripts/python scripts/models/embed_tiles.py --root data/datasets/starc9/val_stanford_small --name starc9_val_small
.venv/Scripts/python scripts/models/embed_tiles.py --root data/datasets/starc9/val_stanford_large --name starc9_val_large
.venv/Scripts/python scripts/models/embed_tiles.py --root data/datasets/starc9/val_tcga --name starc9_val_tcga
.venv/Scripts/python scripts/models/embed_tiles.py --root data/datasets/CRC-VAL-HE-7K --name crc_val_7k
.venv/Scripts/python scripts/models/train_classifier.py --train starc9_train --tune starc9_val_small \
    --test starc9_val_large starc9_val_tcga crc_val_7k --run starc9_lr --Cs 0.1 1 10 100
```

On Delta, **re-run the same download script** instead of copying 180k small files. The dataset version is pinned and the
random picks are seeded, so you get exactly the same tiles:

```bash
cd ~/Ashby- && source scripts/delta_env.sh
nohup python scripts/models/get_starc9.py --out /work/hdd/bfqi/$USER/datasets/starc9 --workers 32 > starc9.log 2>&1 &
```

(Login nodes kill long commands, which is why this uses `nohup`. If it still gets cut off, just run it again: it skips
finished tiles.)

## Other datasets we looked at (and why not)

- **NCT-CRC-HE-100K-NONORM** (Zenodo 1214456, 11.7 GB): the same tiles *without* color normalization. It's good for stain
  robustness, but it has the same 86 slides, so there's no new patient variety and no blood. It's the next thing to try if
  real-slide colors are a problem.
- **Kather 2016 textures** (5,000 tiles): too small, and it merges muscle into stroma, which is our exact weak spot.
- **SemiCOL 2023**: colon segmentation masks from several centers, but the access terms are unclear and it would need
  tiling. Not worth it in a hackathon.

## Citation

Subramanian B, Jeyaraj R, Peterson MN, Guo T, Shah N, Langlotz C, Ng AY, Shen J. *STARC-9: A Large-scale Dataset for
Multi-Class Tissue Classification for CRC Histopathology.* NeurIPS 2025 Datasets and Benchmarks Track.
https://openreview.net/forum?id=rGWjTlK6Ev · data: https://huggingface.co/datasets/Path2AI/STARC-9 (CC-BY 4.0)
