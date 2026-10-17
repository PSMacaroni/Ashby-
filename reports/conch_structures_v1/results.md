# conch_structures_v1: nerve / vessel side classifier (CONCH embeddings)

Trained on 20941 SPIDER tiles from 3097 slides (colon, skin), C = 10.0.

## SPIDER-colon test (2043 tiles, slides not seen in training)

| | zero-shot | classifier |
|---|---|---|
| accuracy | 0.911 | 0.966 |
| balanced_accuracy | 0.911 | 0.966 |
| log_loss | 0.437 | 0.093 |
| recall_other | 0.912 | 0.964 |
| recall_vessel | 0.910 | 0.969 |
| flagged_nerve | 0.055 | 0.011 |
| flagged_vessel | 0.465 | 0.498 |
| precision_vessel | 0.982 | 0.976 |
| auroc_vessel | 0.990 | 0.997 |

Other tissue most often mistaken (classifier): Stroma healthy → nerve 13%, vessel 14%; Necrosis → nerve 2%, vessel 6%; Muscle → nerve 1%, vessel 3%; Inflammation → nerve 0%, vessel 4%; Fat → nerve 0%, vessel 2%

Artery vs vein (zero-shot, unvalidated): 20% of 1026 vessel tiles called artery.

## SPIDER-skin test (2333 tiles, slides not seen in training)

| | zero-shot | classifier |
|---|---|---|
| accuracy | 0.914 | 0.986 |
| balanced_accuracy | 0.917 | 0.986 |
| log_loss | 0.364 | 0.043 |
| recall_other | 0.859 | 0.987 |
| recall_nerve | 0.995 | 0.988 |
| recall_vessel | 0.898 | 0.982 |
| flagged_nerve | 0.388 | 0.330 |
| precision_nerve | 0.851 | 0.994 |
| auroc_nerve | 0.998 | 1.000 |
| flagged_vessel | 0.249 | 0.258 |
| precision_vessel | 0.933 | 0.985 |
| auroc_vessel | 0.985 | 0.999 |

Other tissue most often mistaken (classifier): Muscle → nerve 4%, vessel 5%; Solar elastosis → nerve 2%, vessel 4%; Inflammation → nerve 0%, vessel 3%; Nevus → nerve 0%, vessel 3%; Apocrine glands → nerve 3%, vessel 0%

Artery vs vein (zero-shot, unvalidated): 31% of 605 vessel tiles called artery.
