# Experiment: is MUSK worth adding to our tissue and structures classifiers?

**Branch:** `musk-experiments` (built on `starc9-spider-classifiers`) · **Status:** running (results below when done)

## Question

`tissue8` (CONCH + logistic regression on NCT-CRC-HE-100K + STARC-9, `docs/split_models.md`) averages 94.6% on four
test sets. Its remaining weak spots:

- muscle → stroma on CRC-VAL-HE-7K (muscle 72.8%)
- un-normalized tiles (Kather 2016, 85.6%): lymphocytes scattered in stroma 66%, debris 64%, glass called necrosis 88%
- SPIDER colon tumor 81.4% (partly how SPIDER labels tiles)
- structures model: fat called blood (16% on TCGA)

In our zero-shot bake-off (`docs/handoff_small_classifier.md`), MUSK made **different** mistakes from CONCH and the
average of the two beat both. Does MUSK, or CONCH + MUSK together, fix any of the weak spots once a classifier is
trained on top?

## What we compare (fair test)

- **Same tiles** for every model: the exact STARC-9 tiles from `scripts/get_starc9.py` (seeded), NCT-CRC-HE-100K,
  CRC-VAL-HE-7K, Kather 2016, SPIDER colon + skin.
- **Same training recipe** as `split_v1` (`scripts/train_split_models.py`): logistic regression, class-balanced,
  C from {0.1, 1, 10} picked on STARC-9 VAL-SMALL + a 10% NCT hold-out (same random seed and hold-out for everyone).
- **Three feature sets:** `conch` (512 numbers per tile), `musk` (1024), `both` (side by side, 1536).
- Zero-shot CONCH, zero-shot MUSK and their average are shown for reference.

**Decision rule (set before seeing results):** switch to `musk` or `both` only if it beats `conch` by **at least 2
points** of average accuracy over the non-TCGA tests (CRC-VAL-HE-7K, Stanford VAL-LARGE, SPIDER colon, Kather 2016),
or clearly fixes a weak spot without hurting others. The TCGA test is reported but not used to decide, because MUSK may
have seen TCGA images during pretraining (from our features doc; not verified).

**Cost of switching:** every slide in the pipeline would then need MUSK embeddings too (MUSK is a bigger model).

## How it was run (Delta)

```bash
# data (CPU / network only)
python scripts/get_starc9.py --out data/datasets/starc9                       # same tiles as split_v1
python scripts/get_public_tests.py --out data/datasets                        # CRC-VAL-HE-7K + Kather 2016
sbatch scripts/get_spider_cpu.sbatch --organ colorectal --out data/datasets/spider_colorectal
sbatch scripts/get_spider_cpu.sbatch --organ skin --out data/datasets/spider_skin --vessel-cap 3000 --other-per-class 250
# embeddings (GPU): CONCH + MUSK, each model loaded once, finished sets skipped
sbatch scripts/embed_compare_gpu.sbatch
# train + test (CPU)
sbatch scripts/compare_cpu.sbatch --run musk_v1
```

`data/datasets` and `data/embeddings` are links to `/work/hdd/bfqi/$USER/` (big storage, not the small home folder).
The SPIDER folders are `chmod 700` (gated dataset, CC-BY-NC). Results: `data/results/compare_backbones/musk_v1/`
(`results.md`, `metrics.json`, confusion matrices, the trained classifiers).

## Results

_Pending._
