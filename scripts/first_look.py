"""
First look at CONCH on the Delta datasets. Three tests:

  1. NCT-CRC-HE-100K (labeled tiles): zero-shot accuracy per tissue class + confusion matrix,
     at CONCH's 224 px and native 448 px input sizes, to pick the better one.
  2. PathGen 2k patches: CONCH's guess next to each patch's caption.
  3. (optional, --slide) One TCGA whole slide: random tissue patches.

Run on a Delta GPU node, from the project folder:

    source scripts/delta_env.sh
    python scripts/first_look.py --slide /work/hdd/bfqi/natmakur/tcga_slides/<name>.svs

Paths default to the shared datasets under DATA_DIR. Results go to data/results/first_look/.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # first, so HF_HOME from .env is applied

import matplotlib
matplotlib.use("Agg")  # no screen on Delta: draw straight to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.datasets import load_rgb, sample_nct, sample_pathgen
from src.models import build_text_classifier, classify_images, load_conch
from src.prompts import CLASSES, PROMPTS_VERSION


def top_k(prob_row, k=3):
    order = np.argsort(prob_row)[::-1][:k]
    return [(CLASSES[i], float(prob_row[i])) for i in order]


def save_grid(images, probs, path, titles=None, ncols=4):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (img, p) in enumerate(zip(images, probs)):
        ax = axes.flat[i]
        ax.imshow(img)
        title = f"#{i}\n" + "\n".join(f"{c}: {v:.2f}" for c, v in top_k(p))
        if titles is not None:
            title = titles[i] + "\n" + title
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def prob_table(probs):
    df = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES]).round(4)
    df["label"] = [CLASSES[i] for i in probs.argmax(1)]
    df["confidence"] = probs.max(1).round(4)
    return df


def save_confusion(true_idx, pred_idx, path_png, path_csv, title):
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        cm[t, p] += 1
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(path_csv)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(n), CLASSES)
    ax.set_xlabel("CONCH says")
    ax.set_ylabel("true label")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path_png, dpi=100)
    plt.close(fig)
    return cm


def nct_test(args, out, items, images):
    """Zero-shot accuracy on NCT-CRC at each image size. Returns (best_size, metrics)."""
    true_idx = np.array([CLASSES.index(c) for _, c in items])
    metrics = {}
    for size in args.sizes:
        t0 = time.time()
        model, preprocess = load_conch(args.conch, image_size=size)
        text_emb = build_text_classifier(model)
        probs, _ = classify_images(model, preprocess, images, text_emb)
        pred_idx = probs.argmax(1)
        acc = float((pred_idx == true_idx).mean())
        per_class = {c: float((pred_idx[true_idx == i] == i).mean()) for i, c in enumerate(CLASSES)}
        metrics[size] = {"accuracy": acc, "per_class_accuracy": per_class,
                         "n_tiles": len(items), "seconds": round(time.time() - t0, 1)}

        tag = f"nct_{size}px"
        save_confusion(true_idx, pred_idx, out / f"{tag}_confusion.png", out / f"{tag}_confusion.csv",
                       f"NCT-CRC zero-shot, CONCH @ {size}px: accuracy {acc:.1%}")
        df = prob_table(probs)
        df.insert(0, "true_label", [c for _, c in items])
        df.insert(0, "file", [p.name for p, _ in items])
        df.to_csv(out / f"{tag}_predictions.csv", index=False)

        # 2 examples per class, so you can see what each class looks like and what CONCH said
        pick = [i for k in range(len(CLASSES)) for i in np.flatnonzero(true_idx == k)[:2]]
        save_grid([images[i] for i in pick], probs[pick], out / f"{tag}_examples.png",
                  titles=[f"TRUE: {items[i][1]}" for i in pick])

        print(f"\n== NCT-CRC, CONCH @ {size}px: accuracy {acc:.1%} on {len(items)} tiles "
              f"({metrics[size]['seconds']}s)")
        for c, a in per_class.items():
            print(f"   {c:14s} {a:6.1%}")

        del model, preprocess, text_emb
        gc.collect()
        torch.cuda.empty_cache()
    best = max(metrics, key=lambda s: metrics[s]["accuracy"])
    return best, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = config.DATA_DIR or Path(".")
    ap.add_argument("--nct-dir", default=str(data / "NCT-CRC-HE-100K" / "data-samples"))
    ap.add_argument("--pathgen-dir", default=str(data / "pathgen_dataset" / "pathgen_2k_output"))
    ap.add_argument("--slide", help="optional: one TCGA .svs slide for the random-patch test")
    ap.add_argument("--conch", help="optional: path to a CONCH pytorch_model.bin (else Hugging Face)")
    ap.add_argument("--n-per-class", type=int, default=50, help="NCT tiles per class")
    ap.add_argument("--n", type=int, default=16, help="patches for the PathGen and slide tests")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 448], help="CONCH input sizes to compare")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "first_look"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("device:", config.get_device(), "| prompts", PROMPTS_VERSION)

    # --- Test 1: NCT-CRC accuracy ---
    items = sample_nct(args.nct_dir, args.n_per_class)
    images = [load_rgb(p) for p, _ in items]
    best, metrics = nct_test(args, out, items, images)
    print(f"\nbest size on NCT-CRC: {best}px -> using it for the rest")

    model, preprocess = load_conch(args.conch, image_size=best)
    text_emb = build_text_classifier(model)

    # --- Test 2: PathGen captions ---
    pg = sample_pathgen(args.pathgen_dir, args.n)
    pg_images = [load_rgb(r["path"]) for r in pg]
    pg_probs, _ = classify_images(model, preprocess, pg_images, text_emb)
    save_grid(pg_images, pg_probs, out / "pathgen_grid.png",
              titles=[r["caption"][:50] + "..." for r in pg])
    df = prob_table(pg_probs)
    df.insert(0, "caption", [r["caption"] for r in pg])
    df.insert(0, "file", [r["path"].name for r in pg])
    df.to_csv(out / "pathgen_tiles.csv", index=False)
    print(f"\n== PathGen: {len(pg)} patches")
    for i, (r, p) in enumerate(zip(pg, pg_probs)):
        guess = ", ".join(f"{c} {v:.2f}" for c, v in top_k(p, 2))
        print(f"  #{i:2d} CONCH: {guess}\n      CAPTION: {r['caption']}")

    # --- Test 3 (optional): one whole slide ---
    if args.slide:
        from src.slides import get_mpp, get_thumbnail, open_slide, read_patch, sample_tissue_coords, tissue_mask

        slide = open_slide(args.slide)
        name = Path(args.slide).name
        print(f"\n== Slide {name}: size {slide.dimensions}, microns/px {get_mpp(slide)}")
        thumb = get_thumbnail(slide, 64)
        thumb.save(out / "slide_thumbnail.png")
        plt.imsave(out / "slide_tissue_mask.png", tissue_mask(thumb))
        coords = sample_tissue_coords(slide, n=args.n, out_px=best, seed=0)
        patches = [read_patch(slide, x, y, out_px=best) for x, y in coords]
        probs, _ = classify_images(model, preprocess, patches, text_emb)
        save_grid(patches, probs, out / "slide_tiles_grid.png")
        df = prob_table(probs)
        df.insert(0, "y", [y for _, y in coords])
        df.insert(0, "x", [x for x, _ in coords])
        df.insert(0, "slide", name)
        df.to_csv(out / "slide_tiles.csv", index=False)
        for i, p in enumerate(probs):
            print(f"  #{i:2d} " + ", ".join(f"{c} {v:.2f}" for c, v in top_k(p)))

    summary = {"prompts_version": PROMPTS_VERSION, "best_size": best,
               "nct": {str(k): v for k, v in metrics.items()}}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved to {out}: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
