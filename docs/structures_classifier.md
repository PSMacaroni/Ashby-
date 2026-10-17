# Side classifier: nerves and vessels (for perineural / venous invasion)

**Written:** Sat Oct 17 · **Run:** `conch_structures_v1` (Delta job 22956230, A100, 4 min 39 s ≈ 0.08 GPU-hours)
**Code:** `scripts/get_spider.py` (data), `scripts/embed_spider_gpu.sbatch` (embed + train),
`scripts/train_structures.py`, `src/structures.py` (use in the pipeline)
**Results:** `data/results/structures/conch_structures_v1/` · **Model:** `data/models/conch_structures_v1.joblib`

## TL;DR

Nerves and vessels are where high-risk spread happens: tumor along a nerve is **perineural invasion**, and tumor in a
vessel is **venous / lymphovascular invasion**. Both are stage II "chemo or not" risk factors that are tedious to find
by eye. We trained a small classifier on CONCH embeddings that marks each tile as **nerve / vessel / other**. It is
**not** a 9th map class: results go to a separate `structures.csv` (see CLAUDE.md) for later "tumor next to nerve /
vessel" checks.

On slides it never saw, it finds **97–98% of vessels and 99% of nerves**. It also cuts false "nerve" calls on colon
smooth muscle from **50% (zero-shot) to 1.4%**. Artery vs vein is only a zero-shot guess (no labeled data exists).
MUSK is not used yet: access to it is still awaiting manual approval.

## Data (SPIDER, HistAI, CC-BY-NC 4.0, pathologist-labeled, train/test split by slide)

| | train tiles | test tiles | source |
|---|---|---|---|
| vessel | 5,056 colon + 2,395 skin | 1,026 colon + 605 skin | SPIDER "Vessels" |
| nerve | 3,960 skin | 775 skin | SPIDER-skin "Nerves" (no colon dataset has nerves) |
| other | 4,983 colon + 4,547 skin | 1,017 colon + 953 skin | random sample of every other SPIDER class |

That's 20,941 training tiles from 3,097 slides. Tiles are 224 px at 20x (~0.5 µm/px, the same scale as our pipeline).
They are stored on Delta in `/projects/bfqi/hjeong6/datasets/spider_{colorectal,skin}/` (`/work/hdd` gave I/O errors
on Oct 17). The model is logistic regression, C = 10, picked by 5-fold cross-validation **split by slide**
(balanced accuracy 0.974 ± 0.005).

## Results on test slides (never seen in training)

| | colon: zero-shot | colon: classifier | skin: zero-shot | skin: classifier |
|---|---|---|---|---|
| balanced accuracy | 0.911 | **0.966** | 0.917 | **0.986** |
| vessel found (recall) | 91.0% | **96.9%** | 89.8% | **98.2%** |
| nerve found (recall) | (no colon nerves) | | 99.5% | **98.8%** |
| when it says nerve, correct (precision) | | | 85.1% | **99.4%** |
| when it says vessel, correct (precision) | 98.2% | 97.6% | 93.3% | **98.5%** |
| AUROC nerve / vessel | – / 0.990 | – / **0.997** | 0.998 / 0.985 | **1.000 / 0.999** |
| log-loss (lower = better calibrated) | 0.44 | **0.09** | 0.36 | **0.04** |

**False alarms on colon tissue** (other colon tiles called nerve / vessel):

| colon tissue | zero-shot: nerve | classifier: nerve | classifier: vessel |
|---|---|---|---|
| smooth muscle | **50%** | **1.4%** | 2.9% |
| healthy stroma | 29% | 12.6% | 13.8% |
| necrosis | 6% | 1.5% | 6.2% |
| tumor, adenoma, polyps, mucus | 0–5% | **0%** | 0% |
| all colon "other" | 7.2% | **1.3%** | **2.4%** |

The "healthy stroma" false alarms may be partly real, because normal stroma often contains small vessels and nerve twigs.
Someone should look at those tiles.

## Artery vs vein (zero-shot CONCH, experimental, unvalidated)

No public dataset labels arteries and veins separately, so we can't measure accuracy. Zero-shot calls **20% of colon
and 31% of skin vessel tiles "artery"**. In the example sheets (`most_artery_like_*.png`, `most_vein_like_*.png`), the
"artery" end shows thick, ring-shaped muscular walls, and the "vein" end shows thin walls around wide or blood-filled
spaces. That is the textbook difference, but thin-walled **lymphatics** would also land on the "vein" side.
**Next step if we want it:** a pathologist labels ~150 vessel tiles.

## Honest caveats (for slides)

- **Nerves are only tested on skin.** No colon dataset labels nerves. Colon nerves (in the bowel wall, sometimes with
  ganglion cells) may look different. What we *can* say: it rarely calls colon tissue "nerve" (1.3%).
- **The test sets are about half vessels or nerves by design.** In a real slide, vessels are maybe 2–10% of tiles. At
  2% / 5% / 10% prevalence, "vessel" calls would be right about **46% / 68% / 82%** of the time (from the colon false-alarm
  rate of 2.4%). Use the flags to point a pathologist at spots, not as a count.
- **Tile level only.** A 112 µm tile can contain a vessel or nerve without being labeled that way, and the reverse.
  "Tumor touching a nerve" will need the neighboring tumor tiles too (later step).
- **CONCH only so far.** MUSK access (`xiangjx/musk`) needs manual approval from its authors. Once approved, re-run
  `embed_spider_gpu.sbatch` with MUSK embeddings and compare.

## How to use it in the pipeline

```python
from src.structures import load_structures, predict_structures
bundle = load_structures("data/models/conch_structures_v1.joblib")
s = predict_structures(bundle, image_embeddings)   # same CONCH embeddings as the tissue map
# s["p_nerve"], s["p_vessel"], s["p_other"], s["structure"], s["p_artery"]  -> write to structures.csv
```

## Reproduce on Delta

```bash
sbatch scripts/get_spider_cpu.sbatch --organ colorectal --out /projects/bfqi/$USER/datasets/spider_colorectal
sbatch scripts/get_spider_cpu.sbatch --organ skin --out /projects/bfqi/$USER/datasets/spider_skin --vessel-cap 3000 --other-per-class 250
sbatch scripts/embed_spider_gpu.sbatch     # after both finish: CONCH embeddings + training + test, ~5 min
```

The download jobs stream 89 GB / 255 GB archives and keep only the ~25k needed tiles (2–4 min each on Delta). SPIDER is
gated: accept its terms on Hugging Face with the account whose token is in `.env`.

Citation: Nechaev D. et al. *SPIDER: A Comprehensive Multi-Organ Supervised Pathology Dataset and Baseline Models.*
arXiv 2503.02876 (2025).
