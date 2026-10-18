# blood_check_v1: why blood was mapped as necrosis, and the fix

Sat Oct 17. Delta jobs 22964695, 22964999, 22965836 (blood check parts 1–3) and 22966265 (pipeline test), ≈ 0.34 GPU-h
in total. Scripts: `scripts/models/blood_check.py` (part 1), `blood_check2.py` (part 2), `blood_check3.py` (part 3).
Write-up and the resulting pipeline change: `docs/slide_pipeline.md` → "Blood check".

## Question

Our raw TCGA slides show bright red blood; STARC-9's training blood (Macenko-normalized) is purple. Is color why blood
ends up as "necrosis", and would normalizing our tiles to the training colors fix it?

## Answer

Partly. The tissue model (`tissue8`) has no blood class, so it calls even STARC-9's own blood necrosis (78%). The
CONCH blood detector misses our blood because of **color and zoom**: STARC-9 tiles are 64 µm wide and get enlarged to
224 px, so cells look 1.75× bigger than in our 112 µm tiles. The MUSK blood detector doesn't need any of that.

## Test tiles (2,237 from our 5 slides)

376 bright-red tiles from CM-4746's clot (40 checked by eye, all blood) · 200 CM-4746 muscle tiles · 411 other tiles
`tissue8` calls necrosis · 1,250 random tiles.

## Share of tiles called blood (`versions_table.csv` has all versions)

| blood check | clot | CM-4746 muscle | other "necrosis" | random |
|---|---|---|---|---|
| CONCH structures model, as scanned | 42% | 0% | 9% | 0.7% |
| same, colors normalized per slide | 41% | 0% | 3% | 0.4% |
| same, colors normalized per tile (STARC-9 style) | 70% | 0% | 5% | 0.6% |
| same, STARC-9 zoom | 82% | 3.5% | 27% | 5.4% |
| same, per-tile colors + zoom | 85% | 2% | 13% | 2.3% |
| zero-shot CONCH, "blood" prompt added | 99% | 3.5% | 31% | 5.4% |
| CONCH 9-class (STARC-9) model | 81% | 2.5% | 15% | 1.2% |
| **MUSK structures model (now in the pipeline)** | **98%** | **0%** | **7%** | **0.5%** |

`tissue8` calls the clot necrosis 61–96% of the time under every color / zoom version.

## Pipeline test (AA-3844, CM-4746)

- CM-4746: 558 tiles skipped as blood (368 of the 376 clot tiles); "necrosis" 718 → 255 tiles.
- AA-3844: 16 tiles skipped (the CONCH model would have flagged 123, mostly pale tumor edges).
- Of CM-4746's remaining necrosis tiles, MUSK calls 140 muscle; 16 of 20 checked were plainly muscle, so the analysis
  now overrules them (`cutoffs.ALT_OVERRULES`).

## Figures

| file | what |
|---|---|
| `blood_versions.jpg` | clot tiles as scanned / slide-normalized / tile-normalized / STARC-9 zoom, vs STARC-9 training blood |
| `starc9_training_tiles.jpg` | STARC-9 blood, necrosis and muscle tiles (what the CONCH models learned) |
| `raw_necrosis_sample.jpg` | tiles `tissue8` called necrosis before the fix (as scanned | normalized) |
| `cm4746_before_after.jpg` | CM-4746 tissue map before / after the blood check (red = skipped as blood) |
| `cm4746_necrosis_by_musk_opinion.jpg` | remaining necrosis tiles where MUSK says muscle (top) or necrosis (bottom) |

## Limits

Blood "truth" is one clot on one slide, judged by eye (not a pathologist). Magenta blood on AA-3844 / AA-3875 wasn't
measured. `cutoffs.BLOOD_PROB` stays a placeholder until the pathologist labels ~50 blood / not-blood tiles.
