# Trained models

Small classifiers that sit **on top of frozen CONCH image embeddings** (CONCH ViT-B/16, 224 px input, tiles at ~0.5 µm/px,
`encode_image(..., proj_contrast=True, normalize=True)`). They are scikit-learn logistic regressions, a few KB each.
This is **not** foundation-model weights; those stay on Hugging Face (gated). To use a model, load CONCH as usual,
embed the tiles, then call the matching helper in `src/`.

Saved with scikit-learn 1.9.1 (laptop) and Delta's `pytorch-conda/2.12` environment. Load them with the same major
scikit-learn version.

| file | what it predicts | trained on | code | details |
|---|---|---|---|---|
| `conch_lr_starc9_lr.joblib` | 8 tissue classes + **blood** (blood = "ignore this tile") | STARC-9, 90,000 tiles, ~200 Stanford patients | `src/classifier.py` | `docs/dataset_starc9.md`, `reports/starc9_lr/` |
| `conch_structures_v1.joblib` | **nerve / vessel / other** (+ zero-shot artery vs vein) | SPIDER, 20,941 tiles, 3,097 slides, colon + skin | `src/structures.py` | `docs/structures_classifier.md`, `reports/conch_structures_v1/` |

Both were also tested on the provided datasets they never saw (NCT-CRC-HE-100K, PathGen): `docs/eval_provided.md`, `reports/provided_v1/`.

```python
from src.classifier import load_classifier, predict               # tissue
from src.structures import load_structures, predict_structures    # nerves / vessels
emb = ...  # CONCH image embeddings, shape [n_tiles, 512] (e.g. from src.models.classify_images)

probs8, is_blood = predict(load_classifier("models/conch_lr_starc9_lr.joblib"), emb)
s = predict_structures(load_structures("models/conch_structures_v1.joblib"), emb)
# probs8: [n, 8] in CLASSES order; is_blood: skip these tiles (qc_reason="blood")
# s["p_nerve"], s["p_vessel"], s["p_other"], s["structure"], s["p_artery"] -> structures.csv
```

---

## 1. `conch_lr_starc9_lr.joblib`: tissue classifier (STARC-9)

**What it does:** gives each tile probabilities for our 8 classes (`tumor, normal_mucosa, muscle, fat, lymphocytes,
mucus, necrosis, stroma`) plus `blood`. Blood is not a map class. `predict()` returns the 8 tissue probabilities
re-normalized without blood, plus an `is_blood` flag so the pipeline can skip those tiles.

**Training:** logistic regression (C = 10, class-balanced), 10,000 tiles per class from STARC-9 (Macenko-normalized, shrunk
from 0.25 to 0.5 µm/px). C was picked on STARC-9 VAL-SMALL (20 other slides).

**How it behaves (accuracy on tissue tiles, patients never seen in training):**

| test set | zero-shot CONCH | this model | notes |
|---|---|---|---|
| STARC-9 VAL-LARGE (Stanford, 50 slides) | 82.5% | **97.7%** | tumor 98%, stroma 99%, very well calibrated |
| STARC-9 TCGA (30 TCGA slides) | 78.7% | **96.9%** | |
| CRC-VAL-HE-7K (Heidelberg, 50 patients) | **86.8%** | 86.2% | tumor 63.5% → 90.9%, but stroma 59.6% → 12.4% |
| NCT-CRC-HE-100K (Heidelberg, 86 slides) | **86.9%** | 80.2% | tumor 73.5% → 92.6%, normal 81.6% → 94.7%; muscle 87.8% → 60.8%, stroma 67.1% → 25.4% |

- **Strengths:** tumor vs normal mucosa (the most important distinction) is much better than zero-shot on every test
  set. It also catches blood (95–99%) instead of calling it necrosis.
- **Weakness:** on Heidelberg data (NCT / CRC-VAL), **muscle and stroma** get confused with each other and with necrosis.
  The two datasets define and stain these classes differently, and this model learned Stanford's version. **Overall
  accuracy drops there, so don't use it as the only tissue model yet.**
- **Confidence (NCT):** when ≥99% sure (58% of tiles), it's right 95.5% of the time. Below 50% it's right 28% of the time.

## 2. `conch_structures_v1.joblib`: nerve / vessel side classifier (SPIDER)

**What it does:** gives each tile `p_nerve`, `p_vessel`, `p_other`, kept **separate from the 8 tissue classes**
(`structures.csv`). It's meant for finding tumor next to nerves (perineural invasion) or vessels (venous /
lymphovascular invasion). The bundle also has CONCH text embeddings for a **zero-shot artery-vs-vein** score
(`p_artery`), which is experimental: no labeled artery/vein data exists to check it.

**Training:** logistic regression (C = 10, class-balanced) on SPIDER (pathologist-labeled, split by slide). Vessels come
from colon + skin, nerves from skin only, and "other" from every other SPIDER class. C was picked by 5-fold
cross-validation grouped by slide.

**How it behaves:**

| test | result |
|---|---|
| SPIDER colon test slides | vessels found 96.9% (precision 97.6%), balanced accuracy 0.966 (zero-shot 0.911) |
| SPIDER skin test slides | nerves found 98.8% (precision 99.4%), vessels 98.2%, balanced accuracy 0.986 (zero-shot 0.917) |
| colon smooth muscle wrongly called nerve | **1.4%** (zero-shot: 50%) |
| NCT-CRC-HE-100K (no labels, sanity check) | tumor / normal mucosa never flagged; 13% of "muscle" tiles flagged vessel (many show real vessels) |
| PathGen captions (weak check) | vessel flags 4.6× more common where captions mention vessels; nerve flags 24× more common |

- **Weaknesses:** nerves are only tested on skin (no colon nerve labels exist), and smooth muscle vs nerve is the main
  confusion. In a real slide vessels are rare, so a "vessel" flag is right only about 46–82% of the time depending on how
  common vessels are. Use flags to point a pathologist at spots, not as counts.

## Reproduce

| model | steps |
|---|---|
| tissue | `scripts/get_starc9.py` → `scripts/embed_tiles.py` (4 STARC-9 sets + CRC-VAL-HE-7K) → `scripts/train_classifier.py --run starc9_lr` |
| structures | Delta: `scripts/get_spider_cpu.sbatch` (colon, skin) → `scripts/embed_spider_gpu.sbatch` (embeds + trains) |
| provided-data test | Delta: `scripts/eval_provided_gpu.sbatch` |

Scripts write to `data/models/` and `data/results/` (git-ignored). Copies of the models and reports are kept here and in `reports/`.
