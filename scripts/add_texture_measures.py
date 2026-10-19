"""
Add the texture measures (edges, structure_cov, ink_bluegreen, ink_black, ink_red; src/detectors.texture_measures) to
slides processed before they existed, so the glass / pen-ink check can be applied without re-running the models.
CPU only: reads each tile from the slide again (no GPU, no CONCH / MUSK). Then re-apply the quality check and the
analysis:

    python scripts/add_texture_measures.py data/results/TCGA-* --slide-dirs /projects/bfqi/$USER/tcga_calib
    python scripts/recheck_quality.py data/results/TCGA-*
    python scripts/analyze_slide.py data/results/<slide_id>          # for each folder

Folders that already have the columns are skipped (--force redoes them).
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from src import cutoffs
from src.detectors import TEXTURE_COLUMNS, glass_color, has_texture, texture_measures
from src.slides import fat_aware_mask, get_mpp, get_thumbnail, open_slide

TILE_PX, MPP, DOWNSAMPLE = 224, 0.5, 64          # same as scripts/process_slide.py


def find_slide(slide_id, dirs):
    for d in dirs:
        hits = sorted(Path(d).glob(f"{slide_id}*.svs"))
        if hits:
            return hits[0]
    return None


def measure_chunk(job):
    path, coords, read_px, glass_rgb = job
    slide = open_slide(path)
    out = []
    for x, y in coords:
        tile = slide.read_region((int(x), int(y)), 0, (read_px, read_px)).convert("RGB")
        out.append(texture_measures(tile.resize((TILE_PX, TILE_PX), Image.BILINEAR), glass_rgb))
    return out


def add(folder, slide_dirs, workers, force):
    tiles = pd.read_csv(folder / "tiles.csv")
    if has_texture(tiles) and tiles[list(TEXTURE_COLUMNS)].notna().all().all() and not force:
        print(f"{folder.name}: already has the texture measures, skipped", flush=True)
        return
    path = find_slide(folder.name, slide_dirs)
    if path is None:
        print(f"{folder.name}: slide file not found in {slide_dirs}, skipped", flush=True)
        return
    slide = open_slide(path)
    mpp = get_mpp(slide) or 0.25
    read_px = int(round(TILE_PX * MPP / mpp))
    thumb = get_thumbnail(slide, DOWNSAMPLE)
    mask = fat_aware_mask(thumb, min_sat=cutoffs.TISSUE_MASK_SAT, texture_sd=cutoffs.TISSUE_MASK_TEXTURE_SD,
                          gray_max=cutoffs.TISSUE_MASK_GRAY_MAX, near_px=cutoffs.TISSUE_MASK_NEAR_PX)
    glass_rgb = glass_color(thumb, mask)
    coords = list(zip(tiles.x, tiles.y))
    chunks = [coords[i:i + 500] for i in range(0, len(coords), 500)]
    with Pool(workers) as pool:
        values = [v for part in pool.map(measure_chunk, [(str(path), c, read_px, glass_rgb) for c in chunks]) for v in part]
    for c, col in zip(TEXTURE_COLUMNS, np.asarray(values).T):
        tiles[c] = col.round(5)
    tiles.to_csv(folder / "tiles.csv", index=False)
    print(f"{folder.name}: texture measures added to {len(tiles)} tiles", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--slide-dirs", nargs="+", required=True, help="folders with the .svs files")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for f in args.folders:
        add(Path(f), args.slide_dirs, args.workers, args.force)


if __name__ == "__main__":
    main()
