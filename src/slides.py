"""
Reading whole-slide images (.svs) with openslide.

Slides are huge (often 100,000 x 80,000 pixels), so we never load the whole thing.
We read a small thumbnail to find tissue, then read individual patches at full resolution.
"""
import numpy as np
import openslide
from PIL import Image
from scipy import ndimage
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


def tissue_mask(thumbnail, min_sat=None):
    """
    True where there is tissue. Tissue is pink/purple (colorful); background glass is
    white/grey, so we threshold the color saturation. Rough, but good enough to skip glass.
    min_sat=None: automatic cut-off (Otsu, at least 20). A fixed min_sat (cutoffs.TISSUE_MASK_SAT = 12)
    also keeps pale tissue (submucosa, loose stroma, fat) that the automatic cut-off drops; the
    whole-slide pipeline uses that plus a per-tile background check.
    """
    hsv = np.asarray(thumbnail.convert("HSV"))
    sat = hsv[..., 1].astype(float)
    if min_sat is not None:
        return sat > min_sat
    if sat.max() == sat.min():
        return np.zeros(sat.shape, dtype=bool)
    return sat > max(threshold_otsu(sat), 20)


def fat_aware_mask(thumbnail, min_sat=12, texture_sd=4.0, gray_max=235, near_px=40, close_px=3, min_px=200):
    """
    Tissue mask that also keeps pale fat. The saturation rule (sat > min_sat) misses fat: it is nearly white, but
    unlike glass it has thin membranes, so its brightness changes locally. Adds pale pixels with local SD of
    brightness > texture_sd (5 px window) that lie within near_px of stained tissue, closes small gaps, fills holes,
    and drops pieces smaller than min_px or without stained tissue (dust, smudges). Glass that still slips in is
    removed per tile by src.detectors.quality_check.
    """
    rgb = np.asarray(thumbnail.convert("RGB")).astype(float)
    stained = np.asarray(thumbnail.convert("HSV"))[..., 1].astype(float) > min_sat
    if not stained.any():
        return stained
    gray = rgb.mean(-1)
    mean = ndimage.uniform_filter(gray, 5)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(gray * gray, 5) - mean * mean, 0))
    near = ndimage.distance_transform_edt(~stained) <= near_px
    mask = stained | ((sd > texture_sd) & (gray < gray_max) & near)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((2 * close_px + 1,) * 2, bool)))
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    has_stain = np.bincount(lab[stained & mask].ravel(), minlength=n + 1) >= 50
    keep = (sizes >= min_px) & has_stain
    keep[0] = False
    return keep[lab] | stained          # never lose what the saturation rule kept


def read_patch(slide, x, y, out_px=224, target_mpp=0.5):
    """
    Read a square patch whose top-left is (x, y) in full-resolution pixels, covering
    out_px * target_mpp microns, resized to out_px x out_px. If the slide's mpp is unknown,
    reads out_px pixels at full resolution with no rescaling.
    Default 224 px at 0.5 µm/px = 112 µm, the same as the NCT-CRC tiles (our tile size).
    """
    mpp = get_mpp(slide)
    read_px = int(round(out_px * target_mpp / mpp)) if mpp else out_px
    patch = slide.read_region((int(x), int(y)), 0, (read_px, read_px)).convert("RGB")
    if read_px != out_px:
        patch = patch.resize((out_px, out_px), Image.BILINEAR)
    return patch


def sample_tissue_coords(slide, n=16, out_px=224, target_mpp=0.5, downsample=64, seed=0,
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


def blur_score(patch):
    """
    Sharpness of a patch: variance of the Laplacian (an edge filter) on the grayscale image.
    Blurry patches have few sharp edges, so a low score = blurry. See src/detectors.quality_check: blurry = below
    cutoffs.BLUR_REL x the slide's typical score (scans differ a lot in overall sharpness).
    """
    from skimage.color import rgb2gray
    from skimage.filters import laplace

    gray = rgb2gray(np.asarray(patch.convert("RGB"), dtype=float) / 255.0)
    return float(laplace(gray).var())
