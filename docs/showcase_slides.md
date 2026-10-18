# Showcase slides: 5 TCGA slides for the app demo

Picked on Oct 18 from the 125 slides of the overnight test (`reports/overnight_v4/`), using the current pipeline's
results (`per_slide_after_fixes.csv`, the summaries and `analysis_overview.png` on Delta). The goal: **one slide per
story, together covering every part of the app**. Re-run them with the finished model, then load them into the app.

**Re-run on main `17bdafc` (Oct 18, 16:15):** results, slide record and report in `showcase/` (`showcase/REPORT.md`).
The numbers below are from the overnight run; the stories still hold.

| # | Slide | Set | TCGA pT / pN / stage | Story it tells | What the app shows on it (current pipeline) |
|---|---|---|---|---|---|
| 1 | `TCGA-DM-A282-01Z-00-DX1` | held-out | T3 / N0 / IIA | **The stage II "chemo or not" case** | Textbook wall layers in one section (tumor → mucosa → muscle → fat). Depth hint "touches fat: ≥ pT3" = **exact**. Infiltrative front, budding hotspot, perineural / lymphatic "look here" spots, possible leftover adenoma |
| 2 | `TCGA-DC-6154-01Z-00-DX1` | calibration | T4a / N2b / IVA | **Ragged, high-risk front** | Most dramatic infiltrative front (solidity 0.41, 47 finger protrusions, 4 satellites), front arrows everywhere, direction 5 o'clock, touches fat (exact), 5 high-risk flags incl. venous, risk regions up to L4 |
| 3 | `TCGA-DC-6160-01Z-00-DX1` | held-out | T2 / N0 / I | **Low-risk contrast** | Smooth front (solidity 0.87) pushing into a thick muscle wall (front faces 90% muscle). Depth "reaches the muscle wall: ≥ pT2" = **exact**. Almost no risk points |
| 4 | `TCGA-AZ-4614-01Z-00-DX1` | held-out | T4a / N1 / IVA | **Dead core** | Central necrosis 14% of the tumor + low-oxygen rim 13%. Cleanest QC of the shortlist (0.5% glass / blur). 16 necrosis tiles checked by eye: dead-cell debris and red cells, no glass |
| 5 | `TCGA-CK-4952-01Z-00-DX1` | held-out | T4 / N2 / IIIC | **Mucinous + immune** | TCGA "mucinous" tumor: mucus pools (13% of tissue) next to tumor, normal mucosa and muscle (7 of 8 map colors). Lymphocytes reach inside the tumor ("inflamed-like"), the only inflamed slide among the five |

Backup: `TCGA-CK-6746-01Z-00-DX1` (calibration, T4b / N0 / IIB, **MSI-high**): lymphocytes at the front 0.18 vs core
0.08; good for the immune story and for the stage II "high-risk T4" point, but no muscle wall to judge depth.

## Why these

- **Four of five are held-out** (patients never used to set any cut-off), so what judges see wasn't tuned on them.
  DC-6154 is the exception, kept because no held-out slide has a front as clearly infiltrative.
- **Contrast in pairs:** ragged (2) vs smooth (3); "≥ pT3" (1, 2) vs "≥ pT2" (3); desert-like (1–4) vs inflamed (5).
- Every depth hint on these slides is true against TCGA's pT: exact on 1–3, shallower than the real stage on 4–5
  ("reaches the muscle wall" on T4 tumors). None is deeper than the real stage.
- Clean slides only: glass + blur < 8% of tiles (except DM-A282 at 10.8%; check its thumbnail for pen ink),
  "unsure" < 4%, > 5,000 tissue tiles, no missing resolution, none of the known problem slides (DC-5337 label issue,
  F4-6808 / F5-6813 pen ink, the 5 slides tiled at the wrong scale).

## Known weak spots to keep honest about in the demo

- DC-6160 has two tissue pieces; only the larger tumor piece is analyzed as the "main tumor" (the top piece stays grey
  in the zones view).
- CK-4952: the "mucin inside the tumor" measure misses most mucin (known issue, `results.md` section 4); the tissue map
  and mucus share do show it.
- Necrosis is the weakest class overall (glass gets called necrosis), so AZ-4614's necrosis was checked by eye, by
  Claude, not a pathologist.
- Stage IV slides (DC-6154, AZ-4614) are fine for showing features, but don't frame them as stage II decisions.

## Where the files are

On Delta: `/work/hdd/bfqi/natmakur/results_v4/` (DC-6154, CK-6746) and `/work/hdd/bfqi/natmakur/results_holdout_v1/`
(the other four). Slide files and md5s: `calibration/manifest_v1.csv` and `calibration/manifest_holdout_v1.csv`.
