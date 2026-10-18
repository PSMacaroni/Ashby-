"""
Small helpers that sit next to the 8-class tissue model (see docs/fixes_background_lymph.md):

1. Background guard: glass that slips past the tissue mask gets forced into a tissue class (mostly "necrosis").
   Glass is colorless (white on raw slides, gray elsewhere); necrosis and all other tissue are pink/purple. So a tile
   with < 10% colorful pixels is background UNLESS the tissue model calls it fat (fat is mostly white, keep it).
   Kather 2016: catches 99% of glass tiles, never removes a tile labeled fat, removes no correctly labeled tissue.

2. Immune detector: "are lymphocytes / inflammatory cells present in this tile?", separate from the tissue class,
   so a tile can be stroma or tumor AND lymphocyte-rich (scattered infiltrates, which the tissue model misses).
   Use p_immune for the immune score instead of the tissue model's P(lymphocytes).

3. Blood check: the tissue model has no blood class, so blood comes out as "necrosis". The MUSK "structures" model
   (other / nerve / vessel / blood, from the MUSK experiment) recognizes blood on our raw slides: on CM-4746's clot it
   found 98% of 376 blood tiles and flagged 0.5% of random tissue tiles. The CONCH version found 39% (it learned blood
   from color-normalized, enlarged STARC-9 tiles). Details: docs/slide_pipeline.md.

    from src.detectors import colorful_fraction, is_background, load_immune, predict_immune
    frac = colorful_fraction(tile_rgb)                     # when tiling: 0..1
    bg = is_background(frac, tissue_label)                 # True -> qc_ok=False, qc_reason="background"
    p_immune = predict_immune(load_immune("models/immune_v1.joblib"), embeddings)   # [n] 0..1
    p_blood = predict_blood(load_bundle("models/structures_musk_v1.joblib"), musk_embeddings)   # [n] 0..1
    blood = is_blood(p_blood, tissue_label)                # True -> qc_ok=False, qc_reason="blood"
"""
import joblib
import numpy as np
from PIL import Image

from src import cutoffs

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


def quality_check(colorful, blur, tissue_label):
    """
    Per-tile quality check for a whole slide -> (background, blurry, blur_reference), boolean arrays + a number.
    background = the colorless guard above OR glass (very flat and pale; cutoffs.GLASS_*).
    blurry = blur score below cutoffs.BLUR_REL x this slide's typical sharpness (median blur of tiles with >= 50%
    colorful pixels), so soft but readable scans aren't thrown away.
    """
    from src import cutoffs
    colorful, blur = np.asarray(colorful, float), np.asarray(blur, float)
    glass = (blur < cutoffs.GLASS_BLUR_MAX) & (colorful < cutoffs.GLASS_COLORFUL_MAX)
    background = is_background(colorful, tissue_label) | glass
    stained = colorful >= 0.5
    reference = float(np.median(blur[stained] if stained.sum() >= 20 else blur)) if len(blur) else 0.0
    blurry = (blur < cutoffs.BLUR_REL * reference) & ~background
    return background, blurry, reference


def load_immune(path):
    return joblib.load(path)


def predict_immune(model, embeddings):
    """P(lymphocytes / inflammatory cells present) per tile from CONCH embeddings [n, 512]."""
    return model.predict_proba(embeddings)[:, 1]


def load_bundle(path):
    """A saved model: a dict with "clf" (+ "labels" or "classes"), or a bare scikit-learn classifier."""
    obj = joblib.load(path)
    return obj if isinstance(obj, dict) else {"clf": obj}


def class_probs(bundle, embeddings):
    """[n, n_labels] probabilities in the bundle's label order; a label the model never saw gets 0."""
    clf = bundle["clf"]
    names = bundle.get("labels", bundle.get("classes"))
    if names is None:
        names = list(clf.classes_)
    p = np.zeros((len(embeddings), len(names)))
    p[:, clf.classes_] = clf.predict_proba(embeddings)
    return p


def predict_blood(bundle, embeddings):
    """P(blood) per tile from MUSK embeddings [n, 1024] with the MUSK structures model."""
    return class_probs(bundle, embeddings)[:, list(bundle["labels"]).index("blood")]


def is_blood(p_blood, tissue_label, min_prob=cutoffs.BLOOD_PROB):
    """True if the blood model is sure enough and the tissue model did not call the tile fat (blood models confuse
    the two, and fat matters for the "tumor touches fat" T3 hint). Works on scalars or arrays."""
    return (np.asarray(p_blood) >= min_prob) & (np.asarray(tissue_label) != "fat")
