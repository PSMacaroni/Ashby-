# musk_v1: CONCH vs MUSK vs CONCH+MUSK

Written 2026-10-17 by scripts/compare_backbones.py.

## tissue8 accuracy (balanced accuracy in brackets for the normalized sets)

| test | zero-shot conch | zero-shot musk | zero-shot avg | **conch** | **musk** | **both** |
|---|---|---|---|---|---|---|
| crc_val_7k | 86.8% (0.861) | 90.2% (0.885) | 88.4% (0.873) | 95.6% (0.940) | 95.0% (0.924) | 96.0% (0.947) |
| starc9_val_large | 82.5% (0.825) | 82.1% (0.821) | 84.2% (0.842) | 97.5% (0.975) | 97.4% (0.974) | 98.2% (0.982) |
| starc9_val_tcga (not in decision) | 78.7% (0.791) | 79.5% (0.796) | 81.1% (0.814) | 96.4% (0.964) | 94.2% (0.942) | 97.4% (0.973) |
| spider_colon | 76.5% (0.809) | 82.3% (0.835) | 79.0% (0.830) | 89.3% (0.907) | 84.2% (0.844) | 89.9% (0.907) |
| kather2016 | 76.2% [64%–85%] | 72.0% [63%–80%] | 77.8% [66%–86%] | 85.8% [76%–93%] | 83.3% [72%–92%] | 85.5% [75%–94%] |
| **average of decision tests** | 80.5% | 81.6% | 82.4% | 92.1% | 90.0% | 92.4% |

## Known weak spots (higher is better unless marked)

| weak spot | conch | musk | both |
|---|---|---|---|
| CRC-VAL-7K muscle | 73.3% | 87.7% | 74.7% |
| CRC-VAL-7K stroma | 91.4% | 67.2% | 89.1% |
| SPIDER colon tumor | 80.9% | 83.3% | 85.0% |
| Kather 2016 (raw tiles) lymphocytes | 66.9% | 66.7% | 67.7% |
| Kather 2016 debris | 63.4% | 59.4% | 60.5% |
| Kather 2016 glass called necrosis (lower is better) | 86.6% | 19.8% | 88.6% |
| TCGA fat called blood (lower is better) | 16.0% | 0.1% | 4.9% |

## structures (nerve / vessel / blood / other)

| metric | conch | musk | both |
|---|---|---|---|
| SPIDER colon balanced accuracy | 96.6% | 96.2% | 97.6% |
| SPIDER colon vessel recall | 97.6% | 97.3% | 98.4% |
| SPIDER colon vessel precision | 97.2% | 96.4% | 98.0% |
| SPIDER skin balanced accuracy | 98.5% | 97.2% | 98.8% |
| SPIDER skin nerve recall | 99.6% | 98.6% | 99.7% |
| SPIDER skin nerve precision | 99.1% | 97.7% | 99.6% |
| blood found, Stanford | 99.5% | 99.9% | 99.8% |
| blood found, TCGA | 94.9% | 98.5% | 98.4% |
| TCGA fat called blood (lower is better) | 16.0% | 0.1% | 4.9% |
| NCT muscle called vessel (lower is better) | 7.3% | 5.1% | 3.4% |
| NCT muscle called nerve (lower is better) | 1.0% | 0.5% | 0.3% |

## Decision

Average over crc_val_7k, starc9_val_large, spider_colon, kather2016: zs_conch 80.5%, zs_musk 81.6%, zs_avg 82.4%, conch 92.1%, musk 90.0%, both 92.4%. Best is both (+0.3% vs CONCH): below the 2% rule, **keep CONCH** unless a weak-spot row above changes a lot.
