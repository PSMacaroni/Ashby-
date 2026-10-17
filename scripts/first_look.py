"""
First look at a zero-shot model (CONCH or MUSK) on the Delta datasets. Three tests:

  1. NCT-CRC-HE-100K (labeled tiles): zero-shot accuracy per tissue class + confusion matrix.
     CONCH runs at 224 px and its native 448 px to pick the better one; MUSK always uses 384 px.
  2. PathGen 2k patches: the model's guess next to each patch's caption.
  3. (optional, --slide) One TCGA whole slide: random tissue patches.

The same tiles are used for every model (fixed random seed), so results can be compared directly.

Run on a Delta GPU node, from the project folder:

    source scripts/delta_env.sh
    python scripts/first_look.py --model musk --slide /work/hdd/bfqi/natmakur/tcga_slides/<name>.svs

Paths default to the shared datasets under DATA_DIR. Results go to data/results/first_look/ (CONCH)
or data/results/first_look_<model>/ (other models).
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
from src.models import build_text_classifier, classify_images, load_conch, load_musk
from src.prompts import CLASSES, PROMPTS_VERSION


def load_model(args, size):
    """Load the chosen model. `size` only matters for CONCH (MUSK is fixed at 384 px)."""
    if args.model == "musk":
        return load_musk()
    return load_conch(args.conch, image_size=size)


def try_plot(fn, *args, **kwargs):
    """Draw a figure, but don't let a drawing error throw away the (expensive) results."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  [warning] could not draw {fn.__name__}: {type(e).__name__}: {e}")
        plt.close("all")


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


def confusion_matrix(true_idx, pred_idx, path_csv):
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        cm[t, p] += 1
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(path_csv)
    return cm


def save_confusion(cm, path_png, title):
    n = len(CLASSES)
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


def nct_test(args, out, items, images):
    """
    Zero-shot accuracy on NCT-CRC at each image size.
    Returns (best_size, metrics, (model, preprocess, text_emb) for the best size, still loaded).
    """
    name = args.model.upper()
    sizes = [384] if args.model == "musk" else args.sizes
    true_idx = np.array([CLASSES.index(c) for _, c in items])
    metrics, kept = {}, None
    for size in sizes:
        t0 = time.time()
        model, preprocess = load_model(args, size)
        text_emb = build_text_classifier(model)
        probs, _ = classify_images(model, preprocess, images, text_emb)
        pred_idx = probs.argmax(1)
        acc = float((pred_idx == true_idx).mean())
        per_class = {c: float((pred_idx[true_idx == i] == i).mean()) for i, c in enumerate(CLASSES)}
        metrics[size] = {"accuracy": acc, "per_class_accuracy": per_class,
                         "n_tiles": len(items), "seconds": round(time.time() - t0, 1)}

        # Save the numbers first; pictures last (a drawing error must not lose results)
        tag = f"nct_{size}px"
        df = prob_table(probs)
        df.insert(0, "true_label", [c for _, c in items])
        df.insert(0, "file", [p.name for p, _ in items])
        df.to_csv(out / f"{tag}_predictions.csv", index=False)
        cm = confusion_matrix(true_idx, pred_idx, out / f"{tag}_confusion.csv")

        print(f"\n== NCT-CRC, {name} @ {size}px: accuracy {acc:.1%} on {len(items)} tiles "
              f"({metrics[size]['seconds']}s incl. loading)")
        for c, a in per_class.items():
            print(f"   {c:14s} {a:6.1%}")
        print(f"   most common mix-ups (true -> {name} says: count):")
        off = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(len(CLASSES))
               for j in range(len(CLASSES)) if i != j and cm[i, j]]
        for count, t, p in sorted(off, reverse=True)[:5]:
            print(f"     {t} -> {p}: {count}")

        try_plot(save_confusion, cm, out / f"{tag}_confusion.png",
                 f"NCT-CRC zero-shot, {name} @ {size}px: accuracy {acc:.1%}")
        # 2 examples per class, so you can see what each class looks like and what the model said
        pick = [i for k in range(len(CLASSES)) for i in np.flatnonzero(true_idx == k)[:2]]
        try_plot(save_grid, [images[i] for i in pick], probs[pick], out / f"{tag}_examples.png",
                 titles=[f"TRUE: {items[i][1]}" for i in pick])

        # keep only the best model so far in memory (so we don't reload it for tests 2 and 3)
        if kept is None or acc > metrics[kept[0]]["accuracy"]:
            kept = (size, model, preprocess, text_emb)
        del model, preprocess, text_emb
        gc.collect()
        torch.cuda.empty_cache()
    best = kept[0]
    return best, metrics, kept[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = config.DATA_DIR or Path(".")
    ap.add_argument("--nct-dir", default=str(data / "NCT-CRC-HE-100K" / "data-samples"))
    ap.add_argument("--pathgen-dir", default=str(data / "pathgen_dataset" / "pathgen_2k_output"))
    ap.add_argument("--model", choices=["conch", "musk"], default="conch")
    ap.add_argument("--slide", help="optional: one TCGA .svs slide for the random-patch test")
    ap.add_argument("--slide-px", type=int, default=224,
                    help="slide patch size in px at 0.5 microns/px (same patches for every model)")
    ap.add_argument("--conch", help="optional: path to a CONCH pytorch_model.bin (else Hugging Face)")
    ap.add_argument("--n-per-class", type=int, default=50, help="NCT tiles per class")
    ap.add_argument("--n", type=int, default=16, help="patches for the PathGen and slide tests")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 448], help="CONCH input sizes to compare")
    ap.add_argument("--out", help="output folder (default: data/results/first_look[_<model>])")
    args = ap.parse_args()

    name = args.model.upper()
    out = Path(args.out or config.RESULTS_DIR / ("first_look" if args.model == "conch"
                                                   else f"first_look_{args.model}"))
    out.mkdir(parents=True, exist_ok=True)
    print("model:", name, "| device:", config.get_device(), "| prompts", PROMPTS_VERSION)

    # --- Test 1: NCT-CRC accuracy ---
    items = sample_nct(args.nct_dir, args.n_per_class)
    images = [load_rgb(p) for p, _ in items]
    best, metrics, (model, preprocess, text_emb) = nct_test(args, out, items, images)
    summary = {"model": name, "prompts_version": PROMPTS_VERSION, "best_size": best,
               "nct": {str(k): v for k, v in metrics.items()}}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest size on NCT-CRC: {best}px -> using it for the rest")

    # --- Test 2: PathGen captions ---
    pg = sample_pathgen(args.pathgen_dir, args.n)
    pg_images = [load_rgb(r["path"]) for r in pg]
    pg_probs, _ = classify_images(model, preprocess, pg_images, text_emb)
    df = prob_table(pg_probs)
    df.insert(0, "caption", [r["caption"] for r in pg])
    df.insert(0, "file", [r["path"].name for r in pg])
    df.to_csv(out / "pathgen_tiles.csv", index=False)
    print(f"\n== PathGen: {len(pg)} patches")
    for i, (r, p) in enumerate(zip(pg, pg_probs)):
        guess = ", ".join(f"{c} {v:.2f}" for c, v in top_k(p, 2))
        print(f"  #{i:2d} {name}: {guess}\n      CAPTION: {r['caption']}")
    try_plot(save_grid, pg_images, pg_probs, out / "pathgen_grid.png",
             titles=[r["caption"][:50] + "..." for r in pg])

    # --- Test 3 (optional): one whole slide ---
    if args.slide:
        from src.slides import get_mpp, get_thumbnail, open_slide, read_patch, sample_tissue_coords, tissue_mask

        slide = open_slide(args.slide)
        name = Path(args.slide).name
        print(f"\n== Slide {name}: size {slide.dimensions}, microns/px {get_mpp(slide)}")
        thumb = get_thumbnail(slide, 64)
        thumb.save(out / "slide_thumbnail.png")
        try_plot(plt.imsave, out / "slide_tissue_mask.png", tissue_mask(thumb))
        coords = sample_tissue_coords(slide, n=args.n, out_px=args.slide_px, seed=0)
        patches = [read_patch(slide, x, y, out_px=args.slide_px) for x, y in coords]
        probs, _ = classify_images(model, preprocess, patches, text_emb)
        df = prob_table(probs)
        df.insert(0, "y", [y for _, y in coords])
        df.insert(0, "x", [x for x, _ in coords])
        df.insert(0, "slide", name)
        df.to_csv(out / "slide_tiles.csv", index=False)
        for i, p in enumerate(probs):
            print(f"  #{i:2d} " + ", ".join(f"{c} {v:.2f}" for c, v in top_k(p)))
        try_plot(save_grid, patches, probs, out / "slide_tiles_grid.png")

    print(f"\nsaved to {out}: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
