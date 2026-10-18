"""
Would better preprocessing fix the map errors? Test on tiles that were labeled by eye (reports/overnight_v4 labels +
extra fat / pale tiles), BEFORE changing the pipeline.

For every tile in --tiles (columns: tile_id, slide, row, col), read at full resolution (224 px, 0.5 µm/px) exactly
like scripts/process_slide.py, then:

1. Glass / fat / ink measures (cheap, no model):
   - colorful, blur: the pipeline's current measures (src/detectors.colorful_fraction, src/slides.blur_score)
   - tissue_px: share of pixels that differ from THIS slide's glass color (glass is often tinted; the glass color is
     the median color of thumbnail pixels outside the tissue mask)
   - edges: share of pixels on a sharp edge (Sobel). Fat has a net of thin membranes; glass is flat.
   - ink_bluegreen, ink_black, red_sat: shares of pen-like colors (blue/green, near-black, very saturated red)
2. Color normalization (Macenko): stain colors estimated once per slide from 16 tissue tiles, then every tile is
   mapped to the standard H&E colors. CONCH + the tissue model, and MUSK + blood / second-opinion models, are run on
   the raw and the normalized tile, to compare with the labels by eye.

Writes <out>/features.csv and <out>/tiles/<tile_id>.jpg (224 px raw, for labeling at full resolution) and
<out>/tiles_norm/<tile_id>.jpg.

    python scripts/eval_preprocessing.py --tiles preproc_tiles.csv --out data/preproc_test \
        --slide-dirs /projects/bfqi/$USER/tcga_calib /projects/bfqi/$USER/tcga_slides /work/hdd/bfqi/natmakur/tcga_holdout
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # first, so HF_HOME from .env is applied

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from skimage.filters import sobel
from tqdm import tqdm

from src import cutoffs
from src.classifier import load_classifier, predict
from src.detectors import class_probs, colorful_fraction, load_bundle, predict_blood
from src.models import load_conch, load_musk
from src.prompts import CLASSES
from src.slides import blur_score, fat_aware_mask, get_mpp, get_thumbnail, open_slide

TILE_PX, MPP, DOWNSAMPLE = 224, 0.5, 64
TISSUE_MODEL, BLOOD_MODEL, ALT_MODEL = ("models/tissue8_split_v1.joblib", "models/structures_musk_v1.joblib",
                                        "models/tissue8_musk_v1.joblib")
GLASS_DIST = 30          # a pixel is tissue if its RGB distance from the slide's glass color is above this
EDGE_MIN = 0.06          # Sobel magnitude (0-1 gray) above this = a sharp edge


# --- Macenko stain normalization (Macenko et al. 2009; reference colors as in torchstain / StainTools) ---------------
HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])
IO, BETA, ALPHA = 240, 0.15, 1


def stain_matrix(rgb):
    """H&E stain vectors + 99th-percentile concentrations from an RGB array [n, 3] of tissue pixels."""
    od = -np.log((rgb.astype(float) + 1) / IO)
    od = od[(od > BETA).all(1)]
    if len(od) < 500:
        return None
    _, vecs = np.linalg.eigh(np.cov(od.T))
    v = vecs[:, 1:3]
    proj = od @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    a = v @ np.array([np.cos(lo), np.sin(lo)])
    b = v @ np.array([np.cos(hi), np.sin(hi)])
    he = np.array([a, b]).T if a[0] > b[0] else np.array([b, a]).T
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    return he, np.percentile(conc, 99, axis=1)


def normalize(tile, he, maxc):
    rgb = np.asarray(tile).reshape(-1, 3).astype(float)
    od = -np.log((rgb + 1) / IO)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    conc *= (MAXC_REF / maxc)[:, None]
    out = np.clip(IO * np.exp(-HE_REF @ conc), 0, 255).T.reshape(TILE_PX, TILE_PX, 3)
    return Image.fromarray(out.astype(np.uint8))


# --- cheap measures --------------------------------------------------------------------------------------------------
def measures(tile, glass_rgb):
    a = np.asarray(tile).astype(float)
    hsv = np.asarray(tile.convert("HSV")).astype(float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = a.mean(-1) / 255
    return {"colorful": colorful_fraction(tile), "blur": blur_score(tile),
            "tissue_px": float((np.linalg.norm(a - glass_rgb, axis=-1) > GLASS_DIST).mean()),
            "edges": float((sobel(gray) > EDGE_MIN).mean()),
            "ink_bluegreen": float(((h >= 60) & (h <= 175) & (s > 80)).mean()),
            "ink_black": float((v < 40).mean()),
            "red_sat": float((((h < 12) | (h > 240)) & (s > 200)).mean())}


def find_slide(slide_id, dirs):
    for d in dirs:
        hits = sorted(Path(d).glob(f"{slide_id}*.svs"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slide-dirs", nargs="+", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    (out / "tiles_norm").mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(args.tiles)

    raw, norm, rows = [], [], []
    for slide_id, group in tqdm(tiles.groupby("slide"), desc="slides"):
        path = find_slide(slide_id, args.slide_dirs)
        if path is None:
            print(f"missing slide {slide_id}", flush=True)
            continue
        slide = open_slide(path)
        mpp = get_mpp(slide) or 0.25
        read_px = int(round(TILE_PX * MPP / mpp))
        thumb = get_thumbnail(slide, DOWNSAMPLE)
        mask = fat_aware_mask(thumb, min_sat=cutoffs.TISSUE_MASK_SAT, texture_sd=cutoffs.TISSUE_MASK_TEXTURE_SD,
                              gray_max=cutoffs.TISSUE_MASK_GRAY_MAX, near_px=cutoffs.TISSUE_MASK_NEAR_PX)
        far = ~ndimage.binary_dilation(mask, iterations=10)
        th = np.asarray(thumb).reshape(-1, 3)
        glass_rgb = np.median(th[far.ravel()] if far.sum() > 100 else th[th.mean(1) > np.percentile(th.mean(1), 95)], 0)

        # stain colors for this slide, from 16 tiles well inside the tissue
        inner = ndimage.binary_erosion(mask, iterations=3)
        ys, xs = np.nonzero(inner)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(ys), min(16, len(ys)), replace=False) if len(ys) else []
        sx, sy = slide.dimensions[0] / thumb.width, slide.dimensions[1] / thumb.height
        mosaic = [np.asarray(slide.read_region((int(xs[i] * sx), int(ys[i] * sy)), 0, (read_px, read_px))
                             .convert("RGB").resize((TILE_PX, TILE_PX))).reshape(-1, 3) for i in pick]
        stains = stain_matrix(np.concatenate(mosaic)) if mosaic else None

        for t in group.itertuples():
            tile = slide.read_region((int(t.col * read_px), int(t.row * read_px)), 0, (read_px, read_px)).convert("RGB")
            tile = tile.resize((TILE_PX, TILE_PX), Image.BILINEAR)
            tile.save(out / "tiles" / f"{t.tile_id}.jpg", quality=90)
            ntile = normalize(tile, *stains) if stains is not None else tile
            ntile.save(out / "tiles_norm" / f"{t.tile_id}.jpg", quality=90)
            raw.append(tile)
            norm.append(ntile)
            rows.append({"tile_id": t.tile_id, "slide_mpp": mpp, "glass_r": glass_rgb[0], "glass_g": glass_rgb[1],
                         "glass_b": glass_rgb[2], "normalized": stains is not None, **measures(tile, glass_rgb)})
    feats = pd.DataFrame(rows)

    device = config.get_device()
    conch, prep = load_conch(device=device, image_size=TILE_PX)
    musk, prep_m = load_musk(device=device)
    tissue, blood, alt = load_classifier(TISSUE_MODEL), load_bundle(BLOOD_MODEL), load_bundle(ALT_MODEL)

    def run(images, tag):
        e, m = [], []
        with torch.inference_mode():
            for i in tqdm(range(0, len(images), 64), desc=tag):
                batch = images[i:i + 64]
                e.append(conch.encode_image(torch.stack([prep(x) for x in batch]).to(device),
                                            proj_contrast=True, normalize=True).float().cpu().numpy())
                m.append(musk.encode_image(torch.stack([prep_m(x) for x in batch]).to(device)).float().cpu().numpy())
        E, M = np.concatenate(e), np.concatenate(m)
        probs, _ = predict(tissue, E)
        feats[f"label_{tag}"] = np.array(CLASSES)[probs.argmax(1)]
        feats[f"conf_{tag}"] = probs.max(1).round(4)
        feats[f"alt_{tag}"] = np.array(CLASSES)[class_probs(alt, M).argmax(1)]
        feats[f"p_blood_{tag}"] = predict_blood(blood, M).round(4)
        np.save(out / f"conch_{tag}.npy", E.astype(np.float32))

    run(raw, "raw")
    run(norm, "norm")
    res = tiles.merge(feats, on="tile_id", how="left")
    res.to_csv(out / "features.csv", index=False)
    print(f"saved {out / 'features.csv'}: {len(feats)} tiles from {feats.tile_id.size and tiles.slide.nunique()} slides")


if __name__ == "__main__":
    main()
