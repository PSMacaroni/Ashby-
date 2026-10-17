# split_v1: split models

## Model 1: tissue8 (8 classes only), NCT + STARC-9, C = 10.0

Accuracy on tissue tiles (balanced accuracy in brackets):

| test set | zero-shot | STARC-only | NCT-only | **tissue8 (NCT+STARC)** |
|---|---|---|---|---|
| crc_val_7k (6333) | 86.8% (0.861) | 86.2% (0.776) | 95.8% (0.942) | 95.5% (0.939) |
| starc9_val_large (48000) | 82.5% (0.825) | 97.7% (0.977) | 81.5% (0.815) | 97.6% (0.976) |
| starc9_val_tcga (18500) | 78.7% (0.791) | 96.9% (0.969) | 74.4% (0.746) | 96.4% (0.963) |
| spider_colon (3500) | 76.5% (0.809) | 88.0% (0.892) | 78.6% (0.792) | 89.1% (0.903) |

Per class, tissue8:

| class | crc_val_7k | starc9_val_large | starc9_val_tcga | spider_colon |
|---|---|---|---|---|
| tumor | 96.5% | 98.2% | 97.2% | 81.4% |
| normal_mucosa | 99.5% | 96.8% | 99.1% | 98.6% |
| muscle | 72.8% | 94.5% | 92.0% | 87.0% |
| fat | 99.9% | 98.7% | 94.8% | 98.8% |
| lymphocytes | 98.4% | 98.8% | 99.4% | nan% |
| mucus | 99.5% | 99.4% | 98.2% | nan% |
| necrosis | 93.8% | 96.8% | 98.4% | 91.0% |
| stroma | 91.0% | 97.4% | 91.7% | 85.2% |

## Model 2: structures (nerve / vessel / blood / other), C = 10.0

```
{
 "spider_colon_test": {
  "n": 2043,
  "balanced_accuracy": 0.9671721092307957,
  "recall_other": 0.95968534906588,
  "recall_vessel": 0.9746588693957114,
  "precision_vessel": 0.9746588693957114,
  "auroc_vessel": 0.9970923156246346,
  "other_called_blood": 0.0
 },
 "spider_skin_test": {
  "n": 2333,
  "balanced_accuracy": 0.9873868458442562,
  "recall_other": 0.9779643231899265,
  "recall_nerve": 0.9974193548387097,
  "recall_vessel": 0.9867768595041322,
  "precision_nerve": 0.993573264781491,
  "auroc_nerve": 0.9999519648846743,
  "precision_vessel": 0.9754901960784313,
  "auroc_vessel": 0.9989583333333333,
  "other_called_blood": 0.0031479538300104933
 },
 "starc9_val_large": {
  "blood_tiles": 6000,
  "blood_found": 0.9951666666666666,
  "blood_auroc": 0.9992970833333333,
  "tissue_called_blood": {
   "tumor": 0.0003333333333333333,
   "normal_mucosa": 0.0005,
   "muscle": 0.012333333333333333,
   "fat": 0.07583333333333334,
   "lymphocytes": 0.0028333333333333335,
   "mucus": 0.0035,
   "necrosis": 0.0795,
   "stroma": 0.04
  },
  "tissue_called_nerve": 0.000125,
  "tissue_called_vessel": 0.0006875
 },
 "starc9_val_tcga": {
  "blood_tiles": 1500,
  "blood_found": 0.9453333333333334,
  "blood_auroc": 0.9940300540540541,
  "tissue_called_blood": {
   "tumor": 0.0033333333333333335,
   "normal_mucosa": 0.00045454545454545455,
   "muscle": 0.005714285714285714,
   "fat": 0.1617391304347826,
   "lymphocytes": 0.002,
   "mucus": 0.0124,
   "necrosis": 0.025714285714285714,
   "stroma": 0.012083333333333333
  },
  "tissue_called_nerve": 0.0002702702702702703,
  "tissue_called_vessel": 0.0003243243243243243
 },
 "crc_val_7k": {
  "blood_tiles": 0,
  "tissue_called_blood": {
   "tumor": 0.0,
   "normal_mucosa": 0.0,
   "muscle": 0.005067567567567568,
   "fat": 0.0,
   "lymphocytes": 0.0,
   "mucus": 0.0,
   "necrosis": 0.5103244837758112,
   "stroma": 0.0
  },
  "tissue_called_nerve": 0.0,
  "tissue_called_vessel": 0.0037896731406916154
 },
 "nct_100k_flags": {
  "tumor": {
   "nerve": 0.0,
   "vessel": 0.00034923517496682265,
   "blood": 6.984703499336453e-05
  },
  "normal_mucosa": {
   "nerve": 0.0,
   "vessel": 0.0,
   "blood": 0.00022823234052265205
  },
  "muscle": {
   "nerve": 0.009308510638297872,
   "vessel": 0.07291666666666667,
   "blood": 0.0014036643026004729
  },
  "fat": {
   "nerve": 0.0002882675122513693,
   "vessel": 0.005477082732776016,
   "blood": 0.0012491592197559334
  },
  "lymphocytes": {
   "nerve": 0.0,
   "vessel": 0.0006922211646621095,
   "blood": 0.0018170805572380376
  },
  "mucus": {
   "nerve": 0.0030350719424460434,
   "vessel": 0.009217625899280575,
   "blood": 0.0004496402877697842
  },
  "necrosis": {
   "nerve": 0.0,
   "vessel": 0.0033877692842251565,
   "blood": 0.03509381514940931
  },
  "stroma": {
   "nerve": 0.0026804518475971664,
   "vessel": 0.04068542982959985,
   "blood": 0.0002871912693854107
  }
 }
}
```