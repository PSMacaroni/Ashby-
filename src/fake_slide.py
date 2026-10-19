"""
Fake slides for testing the analysis (and building the app) with no Delta, no GPU and no model.

The fake slide is a strip of colon wall: lumen (glass) on top, then normal mucosa, stroma
(submucosa), muscle, fat, and glass below. A tumor grows down from the mucosa into the fat.

Variants:
  "infiltrative"  tumor with finger protrusions and satellite islands toward the lower left, plus a
                  round island in fat next to an artery (venous-invasion look) and a few planted
                  flag hotspots
  "pushing"       the same tumor with a smooth edge and no fingers or satellites
  "normal"        no tumor at all (like a TCGA "-11" normal-tissue slide)

Every variant also has a lymph node, a mucin pool in the fat, a nerve touching the tumor (if any), a
necrotic core, a band of lymphocytes along the right side of the front, and an "unsure" patch.

    tiles, phrase_scores, meta, truth = make_fake_slide("infiltrative")
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from src.phrases import FLAG_PHRASES
from src.prompts import CLASSES

R, C = 64, 96
TILE_PX = 224               # tile size at 0.5 µm/px (112 µm)
LEVEL0_PER_TILE = 448       # level-0 pixels per tile on a 40x (0.25 µm/px) slide
K = {name: i for i, name in enumerate(CLASSES)}
BG = -1
VARIANTS = ("infiltrative", "pushing", "normal")


def _grid():
    rr, cc = np.mgrid[0:R, 0:C]
    return rr, cc


def _ellipse(r0, c0, ry, rx):
    rr, cc = _grid()
    return ((rr - r0) / ry) ** 2 + ((cc - c0) / rx) ** 2 <= 1


def _box(r0, r1, c0, c1):
    m = np.zeros((R, C), bool)
    m[r0:r1 + 1, c0:c1 + 1] = True
    return m


def layout(variant):
    """True class per tile (BG = glass), plus where the planted structures and hotspots are."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    rr, cc = _grid()
    label = np.full((R, C), BG)
    label[6:12] = K["normal_mucosa"]
    label[12:18] = K["stroma"]
    label[18:28] = K["muscle"]
    label[28:56] = K["fat"]
    label[:, :2] = BG                     # keep the tissue off the grid edges
    label[:, -2:] = BG
    marks = {}

    tumor = np.zeros((R, C), bool)
    if variant != "normal":
        tumor = _ellipse(18, 48, 16, 18) & (rr >= 6)
    if variant == "infiltrative":
        fingers = _box(30, 37, 35, 37) | _box(33, 39, 40, 42) | _box(26, 28, 27, 33)
        more_fingers = _box(34, 41, 50, 52) | _box(18, 20, 24, 29)   # fingers + satellites reach cutoffs' 7; kept out
                                                                         # of marks so the budding hotspot stays put
        satellites = _box(40, 44, 30, 34) | _box(34, 38, 22, 26) | _box(42, 46, 40, 44)
        island = _box(44, 48, 62, 66)       # ~2 mm from the main tumor, in fat, next to an "artery"
        tumor |= fingers | more_fingers | satellites | island
        marks.update(fingers=fingers, satellites=satellites, island=island,
                     artery=_box(45, 47, 68, 68))
    label[tumor] = K["tumor"]

    if variant != "normal":
        label[_box(13, 19, 44, 52)] = K["necrosis"]          # necrotic core
        label[_box(24, 26, 54, 56)] = K["mucus"]             # a little mucin inside the tumor
        outside = ndimage.distance_transform_edt(label != K["tumor"])
        band = (outside >= 1) & (outside <= 2) & (cc > 56) & (rr >= 8) & (rr <= 28) & (label != BG)
        label[band] = K["lymphocytes"]                        # lymphocytes along the right of the front
        marks["nerve"] = _box(32, 32, 56, 57) | _box(33, 33, 57, 57)   # in fat, touching the tumor
        marks["lymphatic"] = _box(20, 20, 68, 68) | _box(22, 22, 68, 68)

    node = _ellipse(46, 12, 6, 6)
    label[node] = K["lymphocytes"]                            # a lymph node in the fat...
    if variant != "normal":
        label[_box(44, 47, 10, 13)] = K["tumor"]              # ...with tumor inside it
    label[_box(40, 43, 76, 80)] = K["mucus"]                  # mucin pool in the fat
    marks["node"] = node
    marks["unsure"] = _box(50, 53, 84, 88)
    marks["tumor"] = label == K["tumor"]
    return label, marks


def make_fake_slide(variant="infiltrative", seed=0):
    """Returns (tiles DataFrame, phrase-score DataFrame, meta dict, truth dict) for one fake slide."""
    rng = np.random.default_rng(seed)
    label, marks = layout(variant)
    rows, cols = np.nonzero(label != BG)
    n = len(rows)
    true = label[rows, cols]

    # model-like probabilities: mostly the true class, a little noise, 3% of tiles plainly wrong
    probs = 0.15 * rng.dirichlet(np.ones(len(CLASSES)), size=n)
    shown = true.copy()
    wrong = rng.random(n) < 0.03
    shown[wrong] = (true[wrong] + rng.integers(1, len(CLASSES), wrong.sum())) % len(CLASSES)
    probs[np.arange(n), shown] += 0.85
    unsure = marks["unsure"][rows, cols]
    probs[unsure] = 1 / len(CLASSES) + 0.01 * rng.random((unsure.sum(), len(CLASSES)))
    probs /= probs.sum(axis=1, keepdims=True)

    tiles = pd.DataFrame({"row": rows, "col": cols, "x": cols * LEVEL0_PER_TILE, "y": rows * LEVEL0_PER_TILE,
                          "tile_size_px": TILE_PX, "qc_ok": True, "qc_reason": ""})
    for i, name in enumerate(CLASSES):
        tiles[f"p_{name}"] = probs[:, i].round(4)
    tiles["label"] = [CLASSES[k] for k in probs.argmax(axis=1)]
    tiles["confidence"] = probs.max(axis=1).round(4)

    # optional detector columns (what the Models teammate's nerve/artery classifiers will add)
    p_nerve = 0.02 + 0.05 * rng.random(n)
    p_artery = 0.02 + 0.05 * rng.random(n)
    if "nerve" in marks:
        p_nerve[marks["nerve"][rows, cols]] = 0.9
    if "artery" in marks:
        p_artery[marks["artery"][rows, cols]] = 0.9
    tiles["p_nerve"], tiles["p_artery"] = p_nerve.round(4), p_artery.round(4)

    # a second model's label: agrees except on a patch of tumor next to the mucosa
    alt = np.array(tiles["label"])
    disagree = (rows >= 8) & (rows <= 11) & (cols >= 36) & (cols <= 40) & (true == K["tumor"])
    alt[disagree] = "normal_mucosa"
    tiles["alt_label"] = alt

    # text-phrase scores: small noise everywhere, bumps where the planted things are
    scores = pd.DataFrame({"row": rows, "col": cols})
    for key in FLAG_PHRASES:
        scores[key] = 0.01 * rng.standard_normal(n)
    for key in ("stroma_mature", "stroma_keloid", "stroma_immature"):
        scores[key] += 0.2

    def bump(key, mask, amount=0.05):
        scores.loc[mask[rows, cols], key] += amount

    if "fingers" in marks:
        near_fingers = ndimage.binary_dilation(marks["fingers"], iterations=3)
        bump("budding", near_fingers)
        bump("vein", marks["island"])
        bump("artery", marks["artery"])
    if "nerve" in marks:
        bump("nerve", marks["nerve"])
        bump("pni", marks["nerve"])
        bump("lymphatic", marks["lymphatic"])
        left_stroma = (true == K["stroma"]) & (cols < 48)
        scores.loc[left_stroma, "stroma_immature"] += 0.03
        scores.loc[(true == K["stroma"]) & (cols >= 48), "stroma_mature"] += 0.03
        scores.loc[(rows <= 11) & (cols < 40) & (true == K["tumor"]), "adenoma"] += 0.05
    for key in FLAG_PHRASES:
        scores[key] = scores[key].round(5)

    meta = {"slide_id": f"fake_{variant}", "mpp": 0.5, "tile_size_px": TILE_PX,
            "thumbnail_downsample": LEVEL0_PER_TILE // 4, "grid_shape": [R, C],
            "model": {"name": "fake (no model)", "prompts_version": "v1"}}
    truth = {"label": label, **marks}
    return tiles, scores, meta, truth


def thumbnail(label, px_per_tile=4):
    """A picture of the true layout (class colors), px_per_tile pixels per tile."""
    from src.viz import label_image

    rgb = label_image(label)
    return np.kron(rgb, np.ones((px_per_tile, px_per_tile, 1)))
