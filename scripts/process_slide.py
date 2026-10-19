"""
Whole-slide pipeline (Delta, GPU): one .svs slide -> the result folder the analysis and the app read
(data contract in CLAUDE.md). Afterwards run scripts/analyze_slide.py (CPU) on the folder.

Steps:
  1. thumbnail (64x smaller) + fat-aware tissue mask (src/slides.fat_aware_mask: stained pixels + textured pale
     pixels next to them, i.e. fat; cutoffs.TISSUE_MASK_*)
  2. grid of 224 px tiles at 0.5 µm/px (112 µm), keep grid cells that are >= 50% tissue in the thumbnail
  3. per tile: read at full resolution, colorful fraction, blur score, texture measures (edges, structure_cov,
     ink_*: src/detectors.texture_measures), 128 px preview
  4. CONCH image embeddings (224 px) -> embeddings.npy (rows = tiles.csv rows); MUSK image embeddings of the same
     tiles (MUSK enlarges them to 384 px) -> embeddings_musk.npy
  5. tissue model (models/tissue8_split_v1.joblib, CONCH) -> p_<class>, label (raw argmax), confidence
  6. lymphocyte detector (models/immune_v1.joblib, CONCH) -> p_immune (used by features.immune when present)
  7. blood check (models/structures_musk_v1.joblib, MUSK) -> p_blood; MUSK's own tissue model
     (models/tissue8_musk_v1.joblib) -> alt_label, a second opinion kept for reference (the analysis doesn't use it)
  8. quality check (src/detectors.quality_check + is_blood): qc_reason "background" (colorless and not fat, or
     glass: very flat and pale, or glass / pen ink by texture: docs/preprocessing_test.md), "blood" (p_blood >= cutoffs.BLOOD_PROB and not labeled fat) or "blur"
     (< cutoffs.BLUR_REL x the slide's typical sharpness). scripts/recheck_quality.py re-applies this to saved tiles

Writes data/results/<slide_id>/: thumbnail.png, tiles.csv, embeddings.npy, embeddings_musk.npy, tiles/<row>_<col>.jpg,
summary.json

    python scripts/process_slide.py --slide /projects/bfqi/$USER/tcga_slides/TCGA-AA-3844-....svs \
        --slides-csv /projects/bfqi/$USER/tcga_slides/slides.csv

--extend: the slide was processed before with a smaller mask. Only the grid cells the current mask adds are read and
embedded; they're appended to tiles.csv / embeddings.npy / embeddings_musk.npy, and the models and quality check are
re-run on all tiles (cheap, from the saved embeddings). A folder made before the MUSK blood check has no
embeddings_musk.npy: then every tile is read again.
"""
import argparse
import json
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # first, so HF_HOME from .env is applied

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import cutoffs
from src.classifier import load_classifier, predict
from src.detectors import (TEXTURE_COLUMNS, class_probs, colorful_fraction, glass_color, is_blood, load_bundle,
                           load_immune, predict_blood, predict_immune, quality_check, texture_measures)
from src.models import load_conch, load_musk
from src.prompts import CLASSES, PROMPTS_VERSION
from src.slides import blur_score, fat_aware_mask, get_mpp, get_thumbnail, open_slide

TILE_PX, MPP, DOWNSAMPLE, PREVIEW_PX = 224, 0.5, 64, 128
TISSUE_MODEL = "models/tissue8_split_v1.joblib"
IMMUNE_MODEL = "models/immune_v1.joblib"
BLOOD_MODEL = "models/structures_musk_v1.joblib"   # MUSK: other / nerve / vessel / blood (MUSK experiment, musk_v1)
ALT_MODEL = "models/tissue8_musk_v1.joblib"        # MUSK's tissue8 (same recipe as tissue8_split_v1): alt_label
MUSK_CHUNK = 128                                   # MUSK works at 384 px: embed at most this many tiles at once


def tile_grid(slide, assume_mpp=None):
    """Tile positions on an exact grid (tiles touch, no overlap) + tissue coverage of each grid cell."""
    mpp = get_mpp(slide) or assume_mpp
    if not mpp:   # guessing 0.25 tiled 5 small TCGA scans at the wrong scale (reports/overnight_v4, section 7)
        raise ValueError("the slide file has no resolution (microns per pixel): skipped. Pass --assume-mpp only if "
                         "you know the real value")
    read_px = int(round(TILE_PX * MPP / mpp))             # level-0 pixels per tile (e.g. 444 at 0.252 µm/px)
    thumb = get_thumbnail(slide, DOWNSAMPLE)
    W, H = slide.dimensions
    sx, sy = W / thumb.width, H / thumb.height             # exact level-0 px per thumbnail px
    mask = fat_aware_mask(thumb, min_sat=cutoffs.TISSUE_MASK_SAT, texture_sd=cutoffs.TISSUE_MASK_TEXTURE_SD,
                          gray_max=cutoffs.TISSUE_MASK_GRAY_MAX, near_px=cutoffs.TISSUE_MASK_NEAR_PX)
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    R, C = H // read_px, W // read_px
    rows = []
    for r in range(R):
        y0, y1 = int(r * read_px / sy), max(int((r + 1) * read_px / sy), int(r * read_px / sy) + 1)
        for c in range(C):
            x0, x1 = int(c * read_px / sx), max(int((c + 1) * read_px / sx), int(c * read_px / sx) + 1)
            area = (y1 - y0) * (x1 - x0)
            cov = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / area
            if cov >= 0.5:
                rows.append((r, c, c * read_px, r * read_px, cov))
    tiles = pd.DataFrame(rows, columns=["row", "col", "x", "y", "mask_coverage"])
    return tiles, thumb, mask, read_px, mpp, (sx + sy) / 2, (R, C)


class Tiles(Dataset):
    def __init__(self, slide_path, tiles, read_px, preprocess, preprocess_musk, preview_dir, glass_rgb):
        self.path, self.tiles, self.read_px, self.glass_rgb = str(slide_path), tiles, read_px, glass_rgb
        self.preprocess, self.preprocess_musk = preprocess, preprocess_musk
        self.preview_dir, self.slide = preview_dir, None

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        if self.slide is None:
            self.slide = open_slide(self.path)                 # one handle per DataLoader worker
        t = self.tiles.iloc[i]
        tile = self.slide.read_region((int(t.x), int(t.y)), 0, (self.read_px, self.read_px)).convert("RGB")
        tile = tile.resize((TILE_PX, TILE_PX), Image.BILINEAR)
        tile.resize((PREVIEW_PX, PREVIEW_PX), Image.BILINEAR).save(
            self.preview_dir / f"{int(t.row)}_{int(t.col)}.jpg", quality=85)
        return (self.preprocess(tile), self.preprocess_musk(tile), colorful_fraction(tile), blur_score(tile),
                np.array(texture_measures(tile, self.glass_rgb), dtype=np.float32))


def slide_info(slide_path, slides_csv):
    """Project / pT / pN / stage from the slides table, if the slide is listed."""
    if not slides_csv or not Path(slides_csv).exists():
        return {}
    df = pd.read_csv(slides_csv)
    row = df[df.file_name == Path(slide_path).name]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {k: (None if pd.isna(r[k]) or r[k] == "None" else str(r[k])) for k in ("project", "pT", "pN", "stage") if k in r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide", required=True, nargs="+", help="one or more .svs files (CONCH + MUSK are loaded once)")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--slides-csv", default=None, help="table with file_name, project, pT, pN, stage")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=128, help="128: each tile also carries a 384 px MUSK copy")
    ap.add_argument("--extend", action="store_true", help="only read cells the current mask adds (see above)")
    ap.add_argument("--skip-done", action="store_true", help="skip slides whose summary.json already exists")
    ap.add_argument("--assume-mpp", type=float, default=None,
                    help="µm/px to use for slides whose file has none (default: skip those slides)")
    ap.add_argument("--stop-at", default=None, help="don't start another slide after this time, e.g. 2026-10-18T07:10")
    args = ap.parse_args()

    stop_at = datetime.fromisoformat(args.stop_at) if args.stop_at else None
    models, failed, n = {}, [], len(args.slide)
    for i, path in enumerate(args.slide, 1):
        slide_id = Path(path).name.split(".")[0]
        if stop_at and datetime.now() >= stop_at:
            print(f"stop-at {args.stop_at} reached: {n - i + 1} slides not started", flush=True)
            break
        if args.skip_done and (Path(args.out) / slide_id / "summary.json").exists():
            print(f"[{i}/{n}] {slide_id}: already done, skipped", flush=True)
            continue
        print(f"[{i}/{n}] {datetime.now():%H:%M:%S} {slide_id}", flush=True)
        try:
            process_one(Path(path), args, models)
        except Exception:                                      # one bad slide shouldn't stop the rest
            traceback.print_exc()
            failed.append(slide_id)
    print(f"finished {datetime.now():%H:%M:%S}; failed: {', '.join(failed) or 'none'}", flush=True)


def process_one(slide_path, args, models):
    """One slide -> its result folder. `models` caches CONCH + MUSK across slides (filled on first use)."""
    slide_id = slide_path.name.split(".")[0]                   # TCGA-AA-3844-01Z-00-DX1
    out = Path(args.out) / slide_id
    (out / "tiles").mkdir(parents=True, exist_ok=True)

    slide = open_slide(slide_path)
    tiles, thumb, mask, read_px, slide_mpp, downsample, grid = tile_grid(slide, args.assume_mpp)
    thumb.save(out / "thumbnail.png")
    glass_rgb = glass_color(thumb, mask)
    print(f"{slide_id}: slide {slide.dimensions}, {slide_mpp} µm/px, tile = {read_px} px at level 0, "
          f"grid {grid}, {len(tiles)} tissue tiles", flush=True)

    n_added = None
    saved = [out / f for f in ("tiles.csv", "embeddings.npy", "embeddings_musk.npy")]
    if args.extend and all(f.exists() for f in saved):
        old = pd.read_csv(saved[0])
        E_old, M_old = np.load(saved[1]), np.load(saved[2])
        assert len(old) == len(E_old) == len(M_old), "tiles.csv and the embeddings don't match"
        seen = set(zip(old.row, old.col))
        todo = tiles[[(r, c) not in seen for r, c in zip(tiles.row, tiles.col)]].reset_index(drop=True)
        cov = dict(zip(zip(tiles.row, tiles.col), tiles.mask_coverage))
        old["mask_coverage"] = [cov.get(k, m) for k, m in zip(zip(old.row, old.col), old.mask_coverage)]
        old = old[["row", "col", "x", "y", "mask_coverage", "colorful", "blur",
                   *[c for c in TEXTURE_COLUMNS if c in old]]]       # folders made before the texture measures lack them
        n_added = len(todo)
        print(f"extend: {len(old)} tiles already done, {n_added} new cells to read", flush=True)
    else:
        if args.extend:
            print("extend: no earlier tiles.csv + embeddings.npy + embeddings_musk.npy here, reading every tile",
                  flush=True)
        old, todo = None, tiles
        E_old, M_old = np.zeros((0, 512), np.float32), np.zeros((0, 1024), np.float32)

    E_new, M_new = np.zeros((0, E_old.shape[1]), np.float32), np.zeros((0, M_old.shape[1]), np.float32)
    if len(todo):
        if not models:
            device = config.get_device()
            models.update(device=device, conch=load_conch(device=device, image_size=TILE_PX),
                          musk=load_musk(device=device))
        device = models["device"]
        (model, preprocess), (musk, preprocess_musk) = models["conch"], models["musk"]
        loader = DataLoader(Tiles(slide_path, todo, read_px, preprocess, preprocess_musk, out / "tiles", glass_rgb),
                            batch_size=args.batch_size, num_workers=args.workers)
        embs, embs_musk, colorful, blur, texture = [], [], [], [], []
        with torch.inference_mode():
            for x, xm, c, b, tx in tqdm(loader, desc=slide_id):
                embs.append(model.encode_image(x.to(device), proj_contrast=True, normalize=True).float().cpu().numpy())
                for chunk in xm.split(MUSK_CHUNK):
                    embs_musk.append(musk.encode_image(chunk.to(device)).float().cpu().numpy())
                colorful.append(c.numpy())
                blur.append(b.numpy())
                texture.append(tx.numpy())
        E_new = np.concatenate(embs).astype(np.float32)
        M_new = np.concatenate(embs_musk).astype(np.float32)
        todo = todo.assign(colorful=np.concatenate(colorful), blur=np.concatenate(blur),
                           **dict(zip(TEXTURE_COLUMNS, np.concatenate(texture).T.round(5))))
    tiles = pd.concat([d for d in (old, todo) if d is not None], ignore_index=True)
    E = np.concatenate([E_old, E_new]).astype(np.float32)
    E_musk = np.concatenate([M_old, M_new]).astype(np.float32)

    probs, _ = predict(load_classifier(TISSUE_MODEL), E)      # 8 classes, rows sum to 1
    for i, c in enumerate(CLASSES):
        tiles[f"p_{c}"] = probs[:, i].round(5)
    tiles["label"] = np.array(CLASSES)[probs.argmax(1)]          # raw argmax: the app's hover must match the numbers
    tiles["confidence"] = probs.max(1).round(5)
    tiles["p_immune"] = predict_immune(load_immune(IMMUNE_MODEL), E).round(5)
    tiles["p_blood"] = predict_blood(load_bundle(BLOOD_MODEL), E_musk).round(5)
    tiles["alt_label"] = np.array(CLASSES)[class_probs(load_bundle(ALT_MODEL), E_musk).argmax(1)]

    for c in TEXTURE_COLUMNS:                                   # extend on an old folder: old tiles have none
        if c not in tiles:
            tiles[c] = np.nan
    background, blurry, blur_ref = quality_check(tiles.colorful, tiles.blur, tiles.label, texture=tiles)
    blood = is_blood(tiles.p_blood.to_numpy(), tiles.label.to_numpy()) & ~background
    blurry = blurry & ~blood
    tiles["qc_ok"] = ~(background | blood | blurry)
    tiles["qc_reason"] = np.where(background, "background", np.where(blood, "blood", np.where(blurry, "blur", "")))
    tiles["tile_size_px"] = TILE_PX

    cols = ["row", "col", "x", "y", "tile_size_px", "qc_ok", "qc_reason", *[f"p_{c}" for c in CLASSES],
            "label", "confidence", "p_immune", "p_blood", "alt_label", "colorful", "blur", "mask_coverage",
            *TEXTURE_COLUMNS]
    tiles[cols].to_csv(out / "tiles.csv", index=False)
    np.save(out / "embeddings.npy", E)
    np.save(out / "embeddings_musk.npy", E_musk)

    ok = tiles.qc_ok
    summary = {
        "slide_id": slide_id, "case_id": slide_id[:12], "processed_at": str(date.today()),
        "mpp": MPP, "slide_mpp": slide_mpp, "tile_size_px": TILE_PX, "thumbnail_downsample": round(downsample, 4),
        "grid_shape": list(grid), "n_tiles_read": int(len(tiles)), "n_tissue_tiles": int(ok.sum()),
        "qc": {"background": int(background.sum()), "blood": int(blood.sum()), "blur": int(blurry.sum()),
               "blur_reference": round(blur_ref, 6),
               "blur_rule": f"blur < {cutoffs.BLUR_REL} x blur_reference (median blur of clearly stained tiles)",
               "blood_cutoff_is_placeholder": True},
        "tissue_mask": {"rule": f"fat-aware: saturation > {cutoffs.TISSUE_MASK_SAT} plus textured pale pixels "
                                f"(local SD > {cutoffs.TISSUE_MASK_TEXTURE_SD}) within {cutoffs.TISSUE_MASK_NEAR_PX} px "
                                f"of stained tissue, on the {DOWNSAMPLE}x thumbnail; cell >= 50% tissue",
                        "background_guard": f"< 10% colorful pixels and not labeled fat, or glass (blur < "
                                            f"{cutoffs.GLASS_BLUR_MAX} and < {cutoffs.GLASS_COLORFUL_MAX:.0%} colorful), "
                                            f"or glass / ink by texture (edges < {cutoffs.GLASS_EDGES_MAX} or "
                                            f"structure_cov < {cutoffs.GLASS_STRUCTURE_MIN} or pen-ink colors)",
                        "glass_rgb": [round(float(v), 1) for v in glass_rgb],
                        "tiles_added_by_extend": n_added},
        "blood_check": {"rule": f"MUSK p_blood >= {cutoffs.BLOOD_PROB} and not labeled {' or '.join(cutoffs.BLOOD_SPARE_LABELS)}", "file": BLOOD_MODEL},
        "model": {"name": "CONCH ViT-B/16 + logistic regression (tissue8_split_v1: NCT-CRC-HE-100K + STARC-9)",
                  "file": TISSUE_MODEL, "prompts_version": PROMPTS_VERSION},
        "alt_model": {"name": "MUSK + logistic regression (tissue8_musk_v1: NCT-CRC-HE-100K + STARC-9)",
                      "file": ALT_MODEL, "column": "alt_label"},
        "embeddings_model": "CONCH", "embeddings": "CONCH ViT-B/16, 224 px, proj_contrast + normalize",
        "embeddings_musk": "MUSK large, 384 px (embeddings_musk.npy)",
        "detectors": {"p_immune": IMMUNE_MODEL, "p_blood": BLOOD_MODEL},
        **slide_info(slide_path, args.slides_csv),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out}: {int(ok.sum())} tissue tiles kept ({int(background.sum())} background, "
          f"{int(blood.sum())} blood, {int(blurry.sum())} blurry); labels {tiles[ok].label.value_counts().to_dict()}",
          flush=True)


if __name__ == "__main__":
    main()
