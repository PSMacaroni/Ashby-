"""
Small helpers that sit next to the 8-class tissue model (see docs/fixes_background_lymph.md):

1. Background guard: glass that slips past the tissue mask gets forced into a tissue class (mostly "necrosis").
   Glass is colorless (white on raw slides, gray elsewhere); necrosis and all other tissue are pink/purple. So a tile
   with < 10% colorful pixels is background UNLESS the tissue model calls it fat (fat is mostly white, keep it).
   Kather 2016: catches 99% of glass tiles, never removes a tile labeled fat, removes no correctly labeled tissue.

2. Immune detector: "are lymphocytes / inflammatory cells present in this tile?", separate from the tissue class,
   so a tile can be stroma or tumor AND lymphocyte-rich (scattered infiltrates, which the tissue model misses).
   Use p_immune for the immune score instead of the tissue model's P(lymphocytes).

    from src.detectors import colorful_fraction, is_background, load_immune, predict_immune
    frac = colorful_fraction(tile_rgb)                     # when tiling: 0..1
    bg = is_background(frac, tissue_label)                 # True -> qc_ok=False, qc_reason="background"
    p_immune = predict_immune(load_immune("models/immune_v1.joblib"), embeddings)   # [n] 0..1
"""
import joblib
import numpy as np
from PIL import Image

COLORFUL_SAT = 20          # HSV saturation (0-255) above which a pixel counts as stained tissue
BACKGROUND_MAX_COLORFUL = 0.10


def colorful_fraction(tile):
    """Share of pixels with real color (stain). ~0 for glass (white or gray), ~0.13 median for fat, > 0.6 for other tissue."""
    img = tile if isinstance(tile, Image.Image) else Image.fromarray(np.asarray(tile, dtype=np.uint8))
    sat = np.asarray(img.convert("RGB").convert("HSV"))[..., 1]
    return float((sat > COLORFUL_SAT).mean())


def is_background(colorful, tissue_label, max_colorful=BACKGROUND_MAX_COLORFUL):
    """True if the tile is (almost) colorless and the tissue model did not call it fat. Works on scalars or arrays."""
    return (np.asarray(colorful) < max_colorful) & (np.asarray(tissue_label) != "fat")


def load_immune(path):
    return joblib.load(path)


def predict_immune(model, embeddings):
    """P(lymphocytes / inflammatory cells present) per tile from CONCH embeddings [n, 512]."""
    return model.predict_proba(embeddings)[:, 1]
