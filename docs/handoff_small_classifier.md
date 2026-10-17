# Handoff: train the small tissue classifier

**Owner:** Models (role 2) · **Status:** not started · **Written:** Sat Oct 17

## TL;DR

CONCH already sorts colon tissue tiles into our 8 classes **zero-shot** (text prompts only) at **90% accuracy**. It makes two
mistakes we care about: **tumor → normal mucosa** and **stroma → muscle**. Its probabilities are also overconfident (it says
1.00 even when wrong). Your job: train a **logistic regression on CONCH image embeddings** using the labeled NCT-CRC-HE-100K
tiles, test it on **different patients** (CRC-VAL-HE-7K), and compare with zero-shot. If it's better, the tissue-map pipeline
uses it instead of zero-shot.

This is a small job: one short GPU run (~10 min) to compute embeddings, then everything else is CPU (seconds).

---

## 1. Background: what we know so far

### What "embedding" and "logistic regression" mean here

- CONCH turns each image tile into a list of **512 numbers** (an *embedding*) that summarizes what the tile looks like.
- **Zero-shot** (what we do now): CONCH also turns text like "an H&E image of adenocarcinoma" into 512 numbers, and each tile
  gets the class whose text is closest. No training.
- **Small classifier** (your job): use tiles that already have correct labels, and train a simple model (logistic regression,
  from scikit-learn) that maps the 512 numbers → one of our 8 classes. CONCH itself is **not** retrained; it stays frozen.
  This usually fixes the classes that the text prompts describe badly, and gives better-calibrated probabilities.

### First-look results (job 22953479, CONCH zero-shot, 50 NCT-CRC tiles per class)

| class | zero-shot @ 224 px | @ 448 px |
|---|---|---|
| **overall** | **90.0%** | 88.8% |
| tumor | 78% | 84% |
| normal_mucosa | 90% | 82% |
| muscle | 88% | 86% |
| fat | 98% | 88% |
| lymphocytes | 100% | 100% |
| mucus | 98% | 94% |
| necrosis | 98% | 98% |
| stroma | 70% | 78% |

Most common mistakes at 224 px: **stroma → muscle 11/50**, **tumor → normal_mucosa 7/50**, muscle → fat 5/50.
Use **224 px** (`load_conch(image_size=224)`): it matches NCT-CRC's tile size, and it's what the pipeline uses now.

Things seen on real TCGA slides that the classifier should ideally help with:
- Real slides vary a lot in stain color (pale lilac to deep purple). NCT-CRC-HE-100K is **color-normalized**, so
  accuracy on real slides will likely be lower than on NCT-CRC.
- Blood (red blood cells) gets called "necrosis". NCT-CRC has **no blood class**, so a classifier trained on it can't fix
  this. See "Open questions".

### Our 8 classes, in order (this order is the column order everywhere: `p_tumor, p_normal_mucosa, ...`)

`src/prompts.py` → `CLASSES = ["tumor", "normal_mucosa", "muscle", "fat", "lymphocytes", "mucus", "necrosis", "stroma"]`

NCT-CRC folder → our class (`src/datasets.py` → `NCT_TO_CLASS`):
`TUM→tumor, NORM→normal_mucosa, MUS→muscle, ADI→fat, LYM→lymphocytes, MUC→mucus, DEB→necrosis, STR→stroma`.
`BACK` (empty glass) is skipped, because our tissue mask removes background before classification.

---

## 2. Data

| set | where | what | use it for |
|---|---|---|---|
| **NCT-CRC-HE-100K** | `/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/<CLASS>/*.tif` (on Delta) | ~89k labeled tiles (excluding BACK), 224×224 px at 0.5 µm/px, color-normalized, from 86 slides | **training** |
| **CRC-VAL-HE-7K** | not on Delta; download (below) | 7,180 labeled tiles from **50 different patients** (no overlap with the 100K set), same size/classes | **testing** |
| our TCGA slides | `/work/hdd/bfqi/natmakur/tcga_slides/` | 5 whole slides (+ `slides.csv` with stage) | stretch: real-slide check |

Download the test set once, on a Delta **login node** (free, no GPU), into your own work folder:

```bash
mkdir -p /work/hdd/bfqi/$USER/datasets && cd /work/hdd/bfqi/$USER/datasets
curl -L -o CRC-VAL-HE-7K.zip "https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content"   # 0.8 GB
unzip -q CRC-VAL-HE-7K.zip && ls CRC-VAL-HE-7K      # expect ADI BACK DEB LYM MUC MUS NORM STR TUM
```

(Source: Zenodo record 1214456, Kather et al., license CC-BY 4.0. If the long download gets cut off on the login node,
run it with `nohup ... &` and check back.)

**Why a separate test set matters:** NCT-CRC-HE-100K has no patient IDs, so a random train/test split would put tiles
from the *same patient* in both, and the score would look better than it really is. Our rule is: always split by patient.
CRC-VAL-HE-7K has no patient overlap, so testing on it gives an honest number.

---

## 3. Steps

### Step 0: Delta setup (once, in your own Delta account)

Follow README → "Saturday on Delta" (or the SSH route if Open OnDemand is slow). The short version:

```bash
# repo in your HOME folder (private). The GitHub repo is private: clone with a GitHub personal access token.
cd ~ && git clone https://github.com/PSMacaroni/Ashby-.git && cd Ashby-
source scripts/delta_env.sh            # EVERY new terminal/job: loads Delta's PyTorch + our packages
pip install --user -r requirements.txt
cp .env.example .env && chmod 600 .env
nano .env   # HF_TOKEN=<yours>, DATA_DIR=/projects/bfqi/shared/datasets, HF_HOME=/projects/bfqi/$USER/hf_cache
mkdir -p /projects/bfqi/$USER/hf_cache && chmod 700 /projects/bfqi/$USER/hf_cache
python scripts/check_setup.py          # everything OK except "No GPU" on the login node
```

Gotchas we already hit (already fixed in the repo, but good to know):
- Delta's PyTorch module hides `pip install --user` packages → that's why `delta_env.sh` exists.
- Delta's own matplotlib crashes drawing text → `requirements.txt` installs a working one.
- `transformers` 5.x broke CONCH's text tokenizer → `src/models.py` has a fixed `tokenize()`.
- Long commands on the **login node** get killed → run anything long as a job, or with `nohup`.
- `/projects/bfqi/<you>` and `/work/hdd/bfqi/<you>` are readable by **everyone in the hackathon**. Keep `.env` in `~`.

### Step 1: compute embeddings (GPU, one short job)

Write `scripts/embed_tiles.py` that, for a folder of NCT-style tiles (`<CLASS>/*.tif`):
1. loads CONCH once at 224 px: `model, preprocess = load_conch(image_size=224)` (from `src/models.py`)
2. reads tiles with a PyTorch `DataLoader` (`num_workers=8`) so file reading runs in parallel. **Don't** load all 89k
   images into a Python list first: that's ~13 GB of RAM.
3. runs `model.encode_image(batch, proj_contrast=True, normalize=True)` under `torch.inference_mode()`. This is exactly
   what `classify_images()` does, so the embeddings match the zero-shot pipeline.
4. saves one `.npz` per dataset: `embeddings` (float32, [N, 512]), `labels` (int, index into `CLASSES`),
   `paths` (str), plus `text_emb` = `build_text_classifier(model).cpu().numpy()` and `logit_scale = model.logit_scale.exp().item()`.
   Saving `text_emb` and `logit_scale` means you can recompute zero-shot probabilities later **on CPU**:
   `softmax(logit_scale * embeddings @ text_emb.T)`.

Rough shape (fill in the details):

```python
class Tiles(torch.utils.data.Dataset):
    def __init__(self, items, preprocess):           # items = [(path, class_name), ...]
        self.items, self.preprocess = items, preprocess
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, cls = self.items[i]
        return self.preprocess(load_rgb(path)), CLASSES.index(cls)

loader = DataLoader(Tiles(items, preprocess), batch_size=256, num_workers=8)
with torch.inference_mode():
    for x, y in tqdm(loader):
        emb = model.encode_image(x.to("cuda"), proj_contrast=True, normalize=True)
        ...
```

`src/datasets.py` → `sample_nct(nct_dir, n_per_class)` already lists tiles per class; for "all tiles" pass a big
`n_per_class` (it caps at what exists), or add a small helper.

**Test it small first:** run on your laptop or a login node with `--n-per-class 5` and `device="cpu"` to catch bugs
without spending GPU time. Then on GPU, run 1,000 per class (~8k tiles) to check speed, then all of NCT-CRC-HE-100K
(minus BACK) + all of CRC-VAL-HE-7K (minus BACK) in one job.

Submit with a copy of `scripts/first_look_gpu.sbatch` (change the job name and the `python ...` line). That file already
has the right account (`bfqi-delta-gpu`), partition (`gpuA100x4`, hackathon reservation), 1 GPU, 8 CPUs, 32 GB.
Bump `--time` to `01:00:00`. You're billed for the time actually used, not the limit.

Save outputs to `/projects/bfqi/$USER/embeddings/` (big, git-ignored type `.npz`; fine to share since it's from public data).
89k × 512 float32 ≈ 180 MB.

**GPU budget:** the whole team has **10 GPU-hours total** (`accounts` shows the balance). This job should take
well under 0.25 hours. **Tell the team before you submit anything on GPU.**

### Step 2: train + evaluate (CPU only, seconds; laptop or a CPU job)

Write `scripts/train_classifier.py`:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
clf.fit(X_train, y_train)          # X = embeddings from NCT-CRC-HE-100K, y = labels
probs = clf.predict_proba(X_test)  # X_test = CRC-VAL-HE-7K
```

- Columns of `predict_proba` follow `clf.classes_`. Make sure that's `[0..7]`, i.e. **CLASSES order**.
- Try a few `C` values (e.g. 0.1, 1, 10) using a validation split of the **training** set, not the test set.
- Then evaluate **once** on CRC-VAL-HE-7K and report, for both zero-shot and the classifier on the **same** test tiles:
  overall accuracy, balanced accuracy, per-class accuracy, confusion matrix (PNG + CSV; reuse `save_confusion` from
  `scripts/first_look.py`).
- **Calibration** (is "0.9" right ~90% of the time?): report log-loss for both, and ideally a reliability plot
  (bin by confidence, plot accuracy per bin). The app shows these probabilities on hover, so this matters.
- Training on 89k × 512 takes seconds to a minute on CPU. Don't run it on a login node; use your laptop
  (copy the `.npz` files down) or a CPU job (`bfqi-delta-cpu`, see `scripts/first_look_cpu.sbatch`).

Save the trained model with `joblib.dump(clf, ...)` to `/projects/bfqi/$USER/models/conch_lr_v1.joblib` and write down
the exact training settings (C, which data, date) next to it.

### Step 3: make it usable by the pipeline

Add `src/classifier.py` with:

```python
def load_classifier(path): ...            # joblib.load
def predict_proba(clf, embeddings): ...   # -> numpy [n, 8], columns in CLASSES order, rows sum to 1
```

The pipeline already gets embeddings from `classify_images()` (it returns `probs, image_embeddings`), so swapping
zero-shot for the classifier is one line. In `summary.json`, set `"model": {"name": "CONCH + logistic regression v1", ...}`.

### Stretch: check on our real TCGA slides

Accuracy on NCT-CRC isn't the same as accuracy on real slides. If there's time: hand-label ~100 tiles from our TCGA
slides (the plan in `CLAUDE.md` → "Validation we want for judges"), then compare zero-shot vs classifier on them.
Get a pathologist/mentor to check a few labels if possible.

---

## 4. What "done" looks like (report back with)

- [ ] `scripts/embed_tiles.py`, `scripts/train_classifier.py`, `src/classifier.py` pushed on a branch + PR
- [ ] Table: zero-shot vs classifier on CRC-VAL-HE-7K: overall, balanced, per-class accuracy, log-loss
- [ ] Both confusion matrices (PNG). Did **tumor→normal** and **stroma→muscle** improve?
- [ ] Path to the saved classifier + embeddings on Delta, and the settings used
- [ ] GPU time used (`sacct -j <jobid> -X -o JobID,Elapsed,State`)
- [ ] A 2–3 sentence takeaway we can put on a slide (honest: trained on color-normalized tiles from one hospital)

## 5. Open questions (decide with the team)

1. **Blood:** NCT-CRC has no blood class, so the classifier will still force blood into one of the 8 classes. Options:
   add "blood" as a zero-shot-only class, or flag tiles where the classifier's top probability is low as "uncertain".
2. **Color normalization:** train on the normalized 100K set (on Delta) or the non-normalized `NCT-CRC-HE-100K-NONORM`
   (11.7 GB on the same Zenodo page)? Non-normalized may transfer better to real TCGA slides, but it's a bigger download.
   Start with what's on Delta; try NONORM only if real-slide results are poor.
3. Should background (`BACK`) become a 9th class to catch tiles the tissue mask lets through? Probably not needed yet.

## 6. Where things are

| what | where |
|---|---|
| CONCH loading, zero-shot, embeddings | `src/models.py` (`load_conch`, `build_text_classifier`, `classify_images`) |
| class list + prompts | `src/prompts.py` |
| NCT-CRC / PathGen readers + class mapping | `src/datasets.py` |
| example GPU / CPU job files | `scripts/first_look_gpu.sbatch`, `scripts/first_look_cpu.sbatch` |
| first-look script (confusion-matrix code to reuse) | `scripts/first_look.py` |
| Delta accounts, paths, rules | `CLAUDE.md` → "Delta specifics" |
| project rules (honest limits, data contract) | `CLAUDE.md` |
