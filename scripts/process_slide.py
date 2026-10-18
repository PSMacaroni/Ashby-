"""
Whole-slide pipeline (Delta, GPU): one .svs slide -> the result folder the analysis and the app read
(data contract in CLAUDE.md). Afterwards run scripts/analyze_slide.py (CPU) on the folder.

Steps:
  1. thumbnail (64x smaller) + tissue mask (saturation > cutoffs.TISSUE_MASK_SAT: keeps pale submucosa / fat)
  2. grid of 224 px tiles at 0.5 µm/px (112 µm), keep grid cells that are >= 50% tissue in the thumbnail
  3. per tile: read at full resolution, colorful fraction, blur score, 128 px preview
  4. CONCH image embeddings (224 px) -> embeddings.npy (rows = tiles.csv rows)
  5. tissue model (models/tissue8_split_v1.joblib) -> p_<class>, label (raw argmax), confidence
  6. lymphocyte detector (models/immune_v1.joblib) -> p_immune (used by features.immune when present)
  7. quality check: qc_reason "background" (src/detectors.is_background: < 10% colorful and not fat) or "blur"

Writes data/results/<slide_id>/: thumbnail.png, tiles.csv, embeddings.npy, tiles/<row>_<col>.jpg, summary.json

    python scripts/process_slide.py --slide /projects/bfqi/$USER/tcga_slides/TCGA-AA-3844-....svs \
        --slides-csv /projects/bfqi/$USER/tcga_slides/slides.csv
"""
import argparse
import json
import sys
from datetime import date
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
from src.detectors import colorful_fraction, is_background, load_immune, predict_immune
from src.models import load_conch
from src.prompts import CLASSES, PROMPTS_VERSION
from src.slides import blur_score, get_mpp, get_thumbnail, open_slide, tissue_mask

TILE_PX, MPP, DOWNSAMPLE, PREVIEW_PX = 224, 0.5, 64, 128
TISSUE_MODEL = "models/tissue8_split_v1.joblib"
IMMUNE_MODEL = "models/immune_v1.joblib"


def tile_grid(slide):
    """Tile positions on an exact grid (tiles touch, no overlap) + tissue coverage of each grid cell."""
    mpp = get_mpp(slide) or 0.25
    read_px = int(round(TILE_PX * MPP / mpp))             # level-0 pixels per tile (e.g. 444 at 0.252 µm/px)
    thumb = get_thumbnail(slide, DOWNSAMPLE)
    W, H = slide.dimensions
    sx, sy = W / thumb.width, H / thumb.height             # exact level-0 px per thumbnail px
    mask = tissue_mask(thumb, min_sat=cutoffs.TISSUE_MASK_SAT)
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
    def __init__(self, slide_path, tiles, read_px, preprocess, preview_dir):
        self.path, self.tiles, self.read_px = str(slide_path), tiles, read_px
        self.preprocess, self.preview_dir, self.slide = preprocess, preview_dir, None

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
        return self.preprocess(tile), colorful_fraction(tile), blur_score(tile)


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
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--slides-csv", default=None, help="table with file_name, project, pT, pN, stage")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    slide_path = Path(args.slide)
    slide_id = slide_path.name.split(".")[0]                   # TCGA-AA-3844-01Z-00-DX1
    out = Path(args.out) / slide_id
    (out / "tiles").mkdir(parents=True, exist_ok=True)

    slide = open_slide(slide_path)
    tiles, thumb, mask, read_px, slide_mpp, downsample, grid = tile_grid(slide)
    thumb.save(out / "thumbnail.png")
    print(f"{slide_id}: slide {slide.dimensions}, {slide_mpp} µm/px, tile = {read_px} px at level 0, "
          f"grid {grid}, {len(tiles)} tissue tiles", flush=True)

    device = config.get_device()
    model, preprocess = load_conch(device=device, image_size=TILE_PX)
    loader = DataLoader(Tiles(slide_path, tiles, read_px, preprocess, out / "tiles"),
                        batch_size=args.batch_size, num_workers=args.workers)
    embs, colorful, blur = [], [], []
    with torch.inference_mode():
        for x, c, b in tqdm(loader, desc=slide_id):
            embs.append(model.encode_image(x.to(device), proj_contrast=True, normalize=True).float().cpu().numpy())
            colorful.append(c.numpy())
            blur.append(b.numpy())
    E = np.concatenate(embs).astype(np.float32)
    tiles["colorful"], tiles["blur"] = np.concatenate(colorful), np.concatenate(blur)

    probs, _ = predict(load_classifier(TISSUE_MODEL), E)      # 8 classes, rows sum to 1
    for i, c in enumerate(CLASSES):
        tiles[f"p_{c}"] = probs[:, i].round(5)
    tiles["label"] = np.array(CLASSES)[probs.argmax(1)]          # raw argmax: the app's hover must match the numbers
    tiles["confidence"] = probs.max(1).round(5)
    tiles["p_immune"] = predict_immune(load_immune(IMMUNE_MODEL), E).round(5)
    background = is_background(tiles.colorful.to_numpy(), tiles.label.to_numpy())
    blurry = tiles.blur.to_numpy() < cutoffs.BLUR_MIN
    tiles["qc_ok"] = ~(background | blurry)
    tiles["qc_reason"] = np.where(background, "background", np.where(blurry, "blur", ""))
    tiles["tile_size_px"] = TILE_PX

    cols = ["row", "col", "x", "y", "tile_size_px", "qc_ok", "qc_reason", *[f"p_{c}" for c in CLASSES],
            "label", "confidence", "p_immune", "colorful", "blur", "mask_coverage"]
    tiles[cols].to_csv(out / "tiles.csv", index=False)
    np.save(out / "embeddings.npy", E)

    ok = tiles.qc_ok
    summary = {
        "slide_id": slide_id, "case_id": slide_id[:12], "processed_at": str(date.today()),
        "mpp": MPP, "slide_mpp": slide_mpp, "tile_size_px": TILE_PX, "thumbnail_downsample": round(downsample, 4),
        "grid_shape": list(grid), "n_tiles_read": int(len(tiles)), "n_tissue_tiles": int(ok.sum()),
        "qc": {"background": int(background.sum()), "blur": int((blurry & ~background).sum()),
               "blur_cutoff_is_placeholder": True},
        "tissue_mask": {"rule": f"saturation > {cutoffs.TISSUE_MASK_SAT} on the {DOWNSAMPLE}x thumbnail, cell >= 50% tissue",
                        "background_guard": "< 10% colorful pixels and not labeled fat"},
        "model": {"name": "CONCH ViT-B/16 + logistic regression (tissue8_split_v1: NCT-CRC-HE-100K + STARC-9)",
                  "file": TISSUE_MODEL, "prompts_version": PROMPTS_VERSION},
        "embeddings_model": "CONCH", "embeddings": "CONCH ViT-B/16, 224 px, proj_contrast + normalize",
        "detectors": {"p_immune": IMMUNE_MODEL},
        **slide_info(slide_path, args.slides_csv),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out}: {int(ok.sum())} tissue tiles kept ({int(background.sum())} background, "
          f"{int((blurry & ~background).sum())} blurry); labels {tiles[ok].label.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
