"""
Reading whole-slide images (.svs) with openslide.

Slides are huge (often 100,000 x 80,000 pixels), so we never load the whole thing.
We read a small thumbnail to find tissue, then read individual patches at full resolution.
"""
import numpy as np
import openslide
from PIL import Image
from skimage.filters import threshold_otsu


def open_slide(path):
    return openslide.OpenSlide(str(path))


def get_mpp(slide):
    """Microns per pixel at full resolution (~0.25 at 40x, ~0.5 at 20x). None if unknown."""
    value = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    return float(value) if value else None


def get_thumbnail(slide, downsample=64):
    """Small image of the whole slide, shrunk by `downsample` in each direction."""
    w, h = slide.dimensions
    return slide.get_thumbnail((w // downsample, h // downsample)).convert("RGB")


def tissue_mask(thumbnail):
    """
    True where there is tissue. Tissue is pink/purple (colorful); background glass is
    white/grey, so we threshold the color saturation. Rough, but good enough to skip glass.
    """
    hsv = np.asarray(thumbnail.convert("HSV"))
    sat = hsv[..., 1].astype(float)
    if sat.max() == sat.min():
        return np.zeros(sat.shape, dtype=bool)
    return sat > max(threshold_otsu(sat), 20)


def read_patch(slide, x, y, out_px=448, target_mpp=0.5):
    """
    Read a square patch whose top-left is (x, y) in full-resolution pixels, covering
    out_px * target_mpp microns, resized to out_px x out_px. If the slide's mpp is unknown,
    reads out_px pixels at full resolution with no rescaling.
    """
    mpp = get_mpp(slide)
    read_px = int(round(out_px * target_mpp / mpp)) if mpp else out_px
    patch = slide.read_region((int(x), int(y)), 0, (read_px, read_px)).convert("RGB")
    if read_px != out_px:
        patch = patch.resize((out_px, out_px), Image.BILINEAR)
    return patch


def sample_tissue_coords(slide, n=16, out_px=448, target_mpp=0.5, downsample=64, seed=0,
                         min_tissue=0.5):
    """
    Pick n random patch positions (full-resolution x, y) that are mostly tissue.
    For a quick look only; the real pipeline tiles the whole slide on a grid.
    """
    thumb = get_thumbnail(slide, downsample)
    mask = tissue_mask(thumb)
    mpp = get_mpp(slide)
    read_px = int(round(out_px * target_mpp / mpp)) if mpp else out_px
    cell = max(1, read_px // downsample)          # patch size measured in thumbnail pixels
    rows, cols = mask.shape[0] // cell, mask.shape[1] // cell
    candidates = []
    for r in range(rows):
        for c in range(cols):
            if mask[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].mean() >= min_tissue:
                candidates.append((c * cell * downsample, r * cell * downsample))
    rng = np.random.default_rng(seed)
    if len(candidates) > n:
        candidates = [candidates[i] for i in rng.choice(len(candidates), n, replace=False)]
    return candidates
