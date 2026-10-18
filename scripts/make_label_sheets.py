"""
Contact sheets for labeling tiles by eye (there is no pathologist on this project, so labels are ours, by eye).
Tiles are sampled from processed slides, then SHUFFLED together so the sheet doesn't reveal what the model said
(blind labeling). The model's answers are only in index.csv, to compare afterwards.

Samples (each spread over slides: at most --per-slide tiles from one slide per set):
  class_<name>   tiles the tissue model labels <name> (8 sets): how often each map label is right
  blood          tiles skipped as blood (qc_reason = blood): were they blood?
  blood_border   0.3 <= p_blood < 0.7, not fat: for choosing cutoffs.BLOOD_PROB later
  overrule       CONCH necrosis + MUSK muscle (the analysis uses muscle): which one was right?
  blood_bins     --n tiles in each p_blood band 0.3-0.5 / 0.5-0.7 / 0.7-0.9 / 0.9-1.0, not fat: for choosing
                 cutoffs.BLOOD_PROB

Writes <out>/sheet_NN.jpg (8 x 6 tiles of 128 px, numbered 1-48) and <out>/index.csv.

    python scripts/make_label_sheets.py --results /work/hdd/bfqi/$USER/results_v4 --out <dir> --n 48
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from_cols = ["row", "col", "label", "confidence", "qc_ok", "qc_reason", "p_blood", "alt_label", "p_immune"]
CLASSES = ["tumor", "normal_mucosa", "muscle", "fat", "lymphocytes", "mucus", "necrosis", "stroma"]
COLS, ROWS, PX = 8, 6, 128


def load(results):
    frames = []
    for folder in sorted(Path(results).iterdir()):
        if (folder / "tiles.csv").exists() and (folder / "tiles").is_dir():
            df = pd.read_csv(folder / "tiles.csv", usecols=lambda c: c in from_cols)
            df["qc_reason"] = df.qc_reason.fillna("").astype(str)
            df["slide"] = folder.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def spread(df, n, per_slide, rng):
    """n random tiles, at most per_slide from any one slide (so big slides don't dominate)."""
    if df.empty:
        return df
    df = df.iloc[rng.permutation(len(df))]
    df = df.groupby("slide", group_keys=False).head(per_slide)
    return df.iloc[rng.permutation(len(df))[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48, help="tiles per set")
    ap.add_argument("--per-slide", type=int, default=4)
    ap.add_argument("--sets", default="classes,blood,blood_border,overrule")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tiles = load(args.results)
    ok = tiles.qc_ok.astype(bool)
    picks = []
    wanted = args.sets.split(",")
    if "classes" in wanted:
        for c in CLASSES:
            picks.append(spread(tiles[ok & (tiles.label == c)], args.n, args.per_slide, rng).assign(set=f"class_{c}"))
    if "blood" in wanted:
        picks.append(spread(tiles[tiles.qc_reason == "blood"], args.n, args.per_slide, rng).assign(set="blood"))
    if "blood_border" in wanted and "p_blood" in tiles:
        m = tiles.p_blood.between(0.3, 0.7, inclusive="left") & (tiles.label != "fat")
        picks.append(spread(tiles[m], args.n, args.per_slide, rng).assign(set="blood_border"))
    if "overrule" in wanted and "alt_label" in tiles:
        m = ok & (tiles.label == "necrosis") & (tiles.alt_label == "muscle")
        picks.append(spread(tiles[m], args.n, args.per_slide, rng).assign(set="overrule"))
    if "blood_bins" in wanted and "p_blood" in tiles:
        for lo, hi in ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
            m = (tiles.p_blood >= lo) & (tiles.p_blood < hi) & (tiles.label != "fat") & (tiles.qc_reason != "background")
            picks.append(spread(tiles[m], args.n, args.per_slide, rng).assign(set=f"blood_{lo:.1f}"))
    index = pd.concat(picks, ignore_index=True)
    index = index.iloc[rng.permutation(len(index))].reset_index(drop=True)   # blind: sets mixed together

    per = COLS * ROWS
    index["sheet"] = index.index // per + 1
    index["cell"] = index.index % per + 1
    for s, g in index.groupby("sheet"):
        sheet = Image.new("RGB", (COLS * (PX + 4), ROWS * (PX + 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for r in g.itertuples():
            x, y = ((r.cell - 1) % COLS) * (PX + 4) + 2, ((r.cell - 1) // COLS) * (PX + 4) + 2
            path = Path(args.results) / r.slide / "tiles" / f"{int(r.row)}_{int(r.col)}.jpg"
            if path.exists():
                sheet.paste(Image.open(path).convert("RGB").resize((PX, PX)), (x, y))
            draw.rectangle([x, y, x + 22, y + 14], fill="black")
            draw.text((x + 2, y + 1), str(r.cell), fill="white")
        sheet.save(out / f"sheet_{s:02d}.jpg", quality=88)
    index.to_csv(out / "index.csv", index=False)
    print(f"{len(index)} tiles on {index.sheet.max()} sheets; per set: {index.set.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
