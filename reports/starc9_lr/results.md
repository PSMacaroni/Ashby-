# starc9_lr: zero-shot CONCH vs CONCH + logistic regression

Trained on starc9_train (90000 tiles), C = 10.0 picked on starc9_val_small.

## starc9_val_large (48000 tissue tiles)

| | zero-shot | classifier |
|---|---|---|
| accuracy | 0.825 | 0.977 |
| balanced_accuracy | 0.825 | 0.977 |
| log_loss | 0.909 | 0.063 |
| ece | 0.126 | 0.001 |
| tumor | 63.5% | 98.2% |
| normal_mucosa | 90.4% | 95.9% |
| muscle | 95.1% | 95.7% |
| fat | 90.7% | 97.8% |
| lymphocytes | 99.9% | 99.0% |
| mucus | 98.9% | 98.9% |
| necrosis | 89.3% | 96.9% |
| stroma | 32.4% | 99.3% |
| tissue wrongly flagged as blood | n/a | 0.4% |

Blood tiles (6000): classifier flags 98.8% as blood; zero-shot calls them: necrosis 65%, fat 27%, lymphocytes 6%, muscle 2%, stroma 1%, mucus 0%.

## starc9_val_tcga (18500 tissue tiles)

| | zero-shot | classifier |
|---|---|---|
| accuracy | 0.787 | 0.969 |
| balanced_accuracy | 0.791 | 0.969 |
| log_loss | 1.208 | 0.088 |
| ece | 0.156 | 0.003 |
| tumor | 52.6% | 96.6% |
| normal_mucosa | 84.4% | 99.0% |
| muscle | 94.9% | 95.6% |
| fat | 84.1% | 89.7% |
| lymphocytes | 99.8% | 99.3% |
| mucus | 95.0% | 98.1% |
| necrosis | 90.0% | 98.8% |
| stroma | 31.7% | 97.9% |
| tissue wrongly flagged as blood | n/a | 1.3% |

Blood tiles (1500): classifier flags 95.4% as blood; zero-shot calls them: necrosis 51%, lymphocytes 38%, fat 9%, muscle 1%, mucus 0%, stroma 0%.

## crc_val_7k (6333 tissue tiles)

| | zero-shot | classifier |
|---|---|---|
| accuracy | 0.868 | 0.862 |
| balanced_accuracy | 0.861 | 0.776 |
| log_loss | 0.781 | 0.553 |
| ece | 0.100 | 0.060 |
| tumor | 63.5% | 90.9% |
| normal_mucosa | 88.9% | 98.1% |
| muscle | 81.8% | 76.7% |
| fat | 98.9% | 99.7% |
| lymphocytes | 99.8% | 89.1% |
| mucus | 99.9% | 97.9% |
| necrosis | 96.8% | 56.3% |
| stroma | 59.6% | 12.4% |
| tissue wrongly flagged as blood | n/a | 2.6% |
