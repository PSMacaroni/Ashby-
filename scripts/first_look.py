"""
First look at CONCH, without Jupyter (same steps as notebooks/01_conch_first_look.ipynb).

Run on a Delta GPU node, from the project folder:

    python scripts/first_look.py --slide-dir /path/to/tcga/slides \
        --pathgen /path/to/PathGen-1.6M.json --conch /path/to/conch/pytorch_model.bin

--pathgen and --conch are optional (without --conch it downloads CONCH using HF_TOKEN).
Saves PNG grids + CSVs to data/results/first_look/ and prints CONCH's guesses.
"""
import argparse
import json
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # first, so HF_HOME from .env is applied

import matplotlib
matplotlib.use("Agg")  # no screen on Delta: draw straight to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import build_text_classifier, classify_images, load_conch
from src.prompts import CLASSES, PROMPTS_VERSION
from src.slides import get_mpp, get_thumbnail, open_slide, read_patch, sample_tissue_coords, tissue_mask


def top_k(prob_row, k=3):
    order = np.argsort(prob_row)[::-1][:k]
    return [(CLASSES[i], float(prob_row[i])) for i in order]


def save_grid(images, probs, path, captions=None, ncols=4):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (img, p) in enumerate(zip(images, probs)):
        ax = axes.flat[i]
        ax.imshow(img)
        title = f"#{i}\n" + "\n".join(f"{c}: {v:.2f}" for c, v in top_k(p))
        if captions is not None:
            title += "\n" + captions[i][:60] + ("..." if len(captions[i]) > 60 else "")
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def prob_table(probs):
    df = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES]).round(4)
    df["label"] = [CLASSES[i] for i in probs.argmax(1)]
    df["confidence"] = probs.max(1).round(4)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide-dir", required=True, help="folder with TCGA .svs slides (searched recursively)")
    ap.add_argument("--slide-index", type=int, default=0, help="which slide to sample random tiles from")
    ap.add_argument("--pathgen", help="path to PathGen-1.6M.json (optional)")
    ap.add_argument("--conch", help="path to preloaded CONCH pytorch_model.bin (optional)")
    ap.add_argument("--n", type=int, default=16, help="patches per test")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "first_look"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("device:", config.get_device())

    slides = sorted(islice(Path(args.slide_dir).rglob("*.svs"), 20000))
    print(f"found {len(slides)} .svs slides in {args.slide_dir}")
    if not slides:
        sys.exit("No slides found - check --slide-dir")

    print("loading CONCH ...")
    model, preprocess = load_conch(args.conch)
    text_emb = build_text_classifier(model)
    print(f"text classifier {tuple(text_emb.shape)}, prompts {PROMPTS_VERSION}")

    # --- Test 1: random tissue patches from one slide ---
    slide_path = slides[args.slide_index]
    slide = open_slide(slide_path)
    print(f"\n== Random tissue patches: {slide_path.name}")
    print("size (px):", slide.dimensions, "| microns/pixel:", get_mpp(slide))
    thumb = get_thumbnail(slide, 64)
    thumb.save(out / "thumbnail.png")
    plt.imsave(out / "tissue_mask.png", tissue_mask(thumb))

    coords = sample_tissue_coords(slide, n=args.n, seed=0)
    patches = [read_patch(slide, x, y) for x, y in coords]
    probs, _ = classify_images(model, preprocess, patches, text_emb)
    save_grid(patches, probs, out / "random_tiles_grid.png")
    df = prob_table(probs)
    df.insert(0, "y", [y for _, y in coords])
    df.insert(0, "x", [x for x, _ in coords])
    df.insert(0, "slide", slide_path.name)
    df.to_csv(out / "random_tiles.csv", index=False)
    for i, p in enumerate(probs):
        print(f"  #{i:2d} " + ", ".join(f"{c} {v:.2f}" for c, v in top_k(p)))

    # --- Test 2: PathGen patches with captions ---
    if args.pathgen:
        print(f"\n== PathGen: loading {args.pathgen} (big file, takes a minute) ...")
        with open(args.pathgen) as f:
            pathgen = json.load(f)
        slide_by_key = {s.name.split(".")[0]: s for s in slides}
        matched = [r for r in pathgen if str(r.get("wsi_id", "")).split(".")[0] in slide_by_key]
        print(f"{len(pathgen):,} records, {len(matched):,} point at slides we have")
        if matched:
            rng = np.random.default_rng(0)
            picked = [matched[i] for i in rng.choice(len(matched), min(args.n, len(matched)), replace=False)]
            pg_patches = []
            for r in picked:
                s = open_slide(slide_by_key[str(r["wsi_id"]).split(".")[0]])
                x, y = (int(v) for v in r["position"])
                pg_patches.append(s.read_region((x, y), 0, (672, 672)).convert("RGB"))
            captions = [r["caption"] for r in picked]
            pg_probs, _ = classify_images(model, preprocess, pg_patches, text_emb)
            save_grid(pg_patches, pg_probs, out / "pathgen_grid.png", captions=captions)
            pg = prob_table(pg_probs)
            pg.insert(0, "caption", captions)
            pg.insert(0, "position", [r["position"] for r in picked])
            pg.insert(0, "wsi_id", [r["wsi_id"] for r in picked])
            pg.to_csv(out / "pathgen_tiles.csv", index=False)
            for i, (cap, p) in enumerate(zip(captions, pg_probs)):
                guess = ", ".join(f"{c} {v:.2f}" for c, v in top_k(p, 2))
                print(f"  #{i:2d} CONCH: {guess}\n      CAPTION: {cap}")

    print(f"\nsaved to {out}: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
