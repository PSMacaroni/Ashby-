"""
Makes FAKE slides in app/demo_data/ so we can build and show the app before real results come
back from Delta. Every number in them is made up, and the app marks them "DEMO".

    python app/make_demo_data.py

The app runs this by itself the first time (when app/demo_data/ is empty). The files follow the
data contract in CLAUDE.md, plus the extra columns and fields listed in app/README.md.

How a fake slide is made:
  1. Draw a colon wall on a grid of tiles: mucosa, stroma (submucosa), muscle, fat.
  2. Add a tumor (smooth = "pushing", ragged with fingers = "infiltrative"), a necrotic core,
     lymphocytes, mucus and satellite islands.
  3. Fake CONCH and MUSK probabilities from the true labels: CONCH over-confident, MUSK softer,
     each with the mix-ups we measured on NCT-CRC.
  4. Run simple versions of the analysis (border, depth, immune, necrosis, risk points) on the
     PREDICTED labels, the way the real pipeline would.
  5. Paint a fake H&E-looking thumbnail from the true labels.
"""
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from scipy.spatial import ConvexHull

from common import CAVEATS, CLASSES, DEMO_DIR

ROWS, COLS = 90, 140      # grid of tiles
TILE_PX = 224             # tile size in pixels at MPP (what the model sees)
MPP = 0.5                 # microns per pixel of a tile, so one tile is 112 µm wide
SLIDE_MPP = 0.25          # microns per pixel of the full-resolution slide (40x)
DOWNSAMPLE = 64           # the thumbnail is the full slide shrunk 64x in each direction
STEP = int(TILE_PX * MPP / SLIDE_MPP)    # one tile in full-resolution pixels (448)
THUMB_TILE = STEP // DOWNSAMPLE          # one tile in thumbnail pixels (7)
TILE_UM = TILE_PX * MPP                  # one tile's width in microns (112)
K = {c: i for i, c in enumerate(CLASSES)}
GLASS = -1

SCENARIOS = [
    dict(slide_id="DEMO-01", name="Demo 1: infiltrative tumor reaching fat", case_id="DEMO-0001",
         entered="2026-10-16", seed=1,
         tumor=dict(center=(34, 60), radius=(28, 30), rough=0.06,
                    fingers=[(0.55, 0.50), (0.85, 0.60), (1.15, 0.45)]),   # (angle, length)
         necrosis=0.40, immune="excluded",
         satellites=[(69, 95, 1.6), (73, 104, 1.8), (64, 110, 1.3)],         # (row, col, radius)
         venous_satellite=1, pen=True),
    dict(slide_id="DEMO-02", name="Demo 2: pushing tumor into muscle", case_id="DEMO-0002",
         entered="2026-10-16", seed=2,
         tumor=dict(center=(30, 78), radius=(20, 27), rough=0.015, fingers=[]),
         necrosis=0.0, immune="inflamed", satellites=[], venous_satellite=None, pen=False),
    dict(slide_id="DEMO-03", name="Demo 3: normal colon wall", case_id="DEMO-0003",
         entered="2026-10-17", seed=3,
         tumor=None, necrosis=0.0, immune=None, satellites=[], venous_satellite=None, pen=False),
]

# Fake model behavior: (true class, class it gets mistaken for, how often), from our NCT-CRC test
CONCH_MIXUPS = [("stroma", "muscle", 0.18), ("tumor", "normal_mucosa", 0.10),
                ("muscle", "fat", 0.06), ("stroma", "necrosis", 0.04)]
MUSK_MIXUPS = [("normal_mucosa", "mucus", 0.12), ("muscle", "stroma", 0.10),
               ("stroma", "muscle", 0.06), ("fat", "mucus", 0.06)]

# Risk points per feature near a tile. PLACEHOLDER until the team's points table
# (features doc, section 7) is in the repo.
# Each entry: (points, card title, short hover text, suggested action).
RISK_FEATURES = {
    "venous_candidate": (2, "Possible venous invasion candidate", "venous invasion candidate",
                         "Look for a vein wall around this tumor island; consider an elastin stain."),
    "infiltrative_front": (1, "Infiltrative (ragged) part of the tumor front", "infiltrative front",
                           "Look for tumor budding and small clusters along this edge."),
    "front_fat": (1, "Tumor front touches fat", "front touches fat",
                  "Confirm tumor in pericolic fat (supports at least pT3)."),
    "satellite": (1, "Tumor island separate from the main mass", "satellite island",
                  "Check whether this island is a tumor deposit, a vessel or a lymph node."),
    "budding_hotspot": (1, "Possible budding hotspot (experimental)", "budding hotspot",
                        "Count tumor buds in this field (ITBCC field: 0.785 mm²)."),
    "stroma_high": (1, "Stroma-high field (more than 50% stroma)", "stroma-high field",
                    "Check the tumor-stroma ratio in this field."),
    "necrosis_rim": (1, "Next to necrosis (likely low oxygen)", "next to necrosis",
                     "Context only: tissue next to necrosis is likely low in oxygen."),
}
NEAR_TILES = 4   # "near" = within 4 tiles (~0.45 mm), so a 1 mm-wide neighborhood

# Rough H&E colors (RGB) per true class, for the fake thumbnail and tile pictures
HE_COLORS = {
    "tumor": (118, 58, 140), "normal_mucosa": (165, 95, 165), "muscle": (206, 96, 138),
    "fat": (240, 228, 234), "lymphocytes": (78, 48, 128), "mucus": (192, 192, 218),
    "necrosis": (202, 150, 172), "stroma": (228, 166, 190),
}
GLASS_RGB = (244, 243, 246)
NUCLEI = {"tumor": 0.30, "normal_mucosa": 0.18, "muscle": 0.08, "fat": 0.01, "lymphocytes": 0.55,
          "mucus": 0.02, "necrosis": 0.15, "stroma": 0.05}   # share of pixels that are dark nuclei

CROSS = ndimage.generate_binary_structure(2, 1)   # 4 neighbors (up, down, left, right)
SQUARE = np.ones((3, 3), bool)                    # 8 neighbors


# ---------------------------------------------------------------- 1-2. true tissue labels

def disk(center, radius):
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    return np.hypot(rr - center[0], cc - center[1]) <= radius


def patches(mask, fraction, rng, sigma=1.2):
    """About `fraction` of the tiles in `mask`, in small clumps rather than scattered dots."""
    if fraction <= 0 or not mask.any():
        return np.zeros_like(mask)
    field = ndimage.gaussian_filter(rng.normal(size=mask.shape), sigma)
    return mask & (field >= np.quantile(field[mask], 1 - fraction))


def colon_wall(rng):
    """True labels for a normal colon wall seen from the side, plus each tile's depth (rows)."""
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    phase = rng.uniform(0, 2 * np.pi)
    surface = 12 + 5 * np.sin(cc / 22 + phase)       # where the gut lumen ends
    depth = rr - surface                             # rows below the surface
    outer = 66 + 4 * np.sin(cc / 13 + phase)         # outer edge of the fat
    labels = np.full((ROWS, COLS), GLASS)
    labels[(depth >= 0) & (depth < 12)] = K["normal_mucosa"]
    labels[(depth >= 12) & (depth < 26)] = K["stroma"]          # submucosa
    labels[(depth >= 26) & (depth < 44)] = K["muscle"]          # muscularis propria
    labels[(depth >= 44) & (depth < outer)] = K["fat"]          # fat around the bowel
    labels[(cc < 4 + 3 * np.sin(rr / 7)) | (cc > COLS - 5 - 3 * np.sin(rr / 9))] = GLASS
    return labels, depth


def tumor_blob(center, radius, rough, fingers, rng):
    """Tumor mask, each tile's distance from the center (1 = the border) and its angle."""
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    dr, dc = (rr - center[0]) / radius[0], (cc - center[1]) / radius[1]
    dist, angle = np.hypot(dr, dc), np.arctan2(dr, dc)     # angle 0 = right, pi/2 = down
    edge = np.ones_like(dist)
    for k in range(2, 15):          # wobble the outline: more wobble = more ragged border
        edge += rough * rng.normal() * np.sin(k * angle + rng.uniform(0, 2 * np.pi))
    for a, length in fingers:       # narrow protrusions ("fingers") at the infiltrative front
        gap = np.angle(np.exp(1j * (angle - a)))
        edge += length * np.exp(-gap ** 2 / (2 * 0.06 ** 2))
    return dist < edge, dist / edge, angle


def true_labels(sc, rng):
    labels, depth = colon_wall(rng)
    tissue = labels != GLASS

    # a few lymphoid aggregates in the submucosa, and mucus on the surface
    for c in rng.choice(np.arange(15, COLS - 15), 3, replace=False):
        r = int(np.argmax(depth[:, c] >= 16))
        labels[disk((r, c), 2.2) & tissue] = K["lymphocytes"]
    for c in rng.choice(np.arange(12, COLS - 12), 4, replace=False):
        r = int(np.argmax(depth[:, c] >= 0))
        labels[disk((r - 1, c), 2.0) & (depth >= -2) & (depth < 1)] = K["mucus"]

    t = sc["tumor"]
    if t:
        tumor, sdist, angle = tumor_blob(t["center"], t["radius"], t["rough"], t["fingers"], rng)
        # grows from the lining; may bulge into the lumen, but not out past the outer edge
        tumor &= (depth >= -3) & ((labels != GLASS) | (depth < 0))
        labels[tumor] = K["tumor"]
        if sc["necrosis"]:                       # dead core in the middle
            core = tumor & (sdist < sc["necrosis"] * (1 + 0.3 * ndimage.gaussian_filter(
                rng.normal(size=tumor.shape), 2)))
            labels[core] = K["necrosis"]
        if t["fingers"]:                         # stroma reaction between the fingers
            lo, hi = min(a for a, _ in t["fingers"]) - 0.3, max(a for a, _ in t["fingers"]) + 0.3
            front = tumor & (sdist > 0.7) & (angle > lo) & (angle < hi)
            labels[patches(front, 0.35, rng, sigma=0.8)] = K["stroma"]
        for r, c, rad in sc["satellites"]:       # small separate tumor islands
            labels[disk((r, c), rad) & (labels != GLASS)] = K["tumor"]

        # lymphocytes: "excluded" = mostly at the edge; "inflamed" = inside the tumor too
        outside = ndimage.distance_transform_edt(~tumor)
        band = ~tumor & (outside <= 3) & (labels != GLASS) & (labels != K["mucus"])
        living = tumor & (labels == K["tumor"])
        if sc["immune"] == "excluded":
            infiltrative_side = (angle > 0.2) & (angle < 1.4)
            labels[patches(band & ~infiltrative_side, 0.45, rng)] = K["lymphocytes"]
            labels[patches(band & infiltrative_side, 0.12, rng)] = K["lymphocytes"]
            labels[patches(living, 0.02, rng)] = K["lymphocytes"]
        elif sc["immune"] == "inflamed":
            labels[patches(band, 0.50, rng)] = K["lymphocytes"]
            labels[patches(living, 0.18, rng)] = K["lymphocytes"]
    return labels


def qc_marks(labels, sc, rng):
    """Tiles that fail quality checks: a pen mark (some slides) and a few blurry tiles."""
    reason = np.full(labels.shape, "", dtype=object)
    if sc["pen"]:
        for i in range(40):                       # a diagonal pen stroke
            r, c = int(10 + i * 0.75), int(116 + i * 0.3)
            reason[r, c] = reason[r, min(c + 1, COLS - 1)] = "pen mark"
    blurry = (rng.random(labels.shape) < 0.004) & (reason == "")
    reason[blurry] = "blurry"
    reason[labels == GLASS] = ""
    return reason


# ---------------------------------------------------------------- 3. fake model outputs

def fake_model(labels, rng, sharpness, mixups):
    """One model's probabilities [rows, cols, 8] for every tile, faked from the true labels."""
    n = len(CLASSES)
    onehot = np.stack([labels == k for k in range(n)], -1).astype(float)
    # share of the 3x3 neighborhood in each class: at borders, the neighbors' classes leak in
    neigh = np.stack([ndimage.uniform_filter(onehot[..., k], 3) for k in range(n)], -1)
    logits = rng.normal(0, 1, onehot.shape) + sharpness * onehot
    leak = rng.uniform(0.5, 1.2, labels.shape)[..., None]
    logits += sharpness * 0.9 * neigh * (1 - onehot) * leak
    for true_c, wrong_c, rate in mixups:
        hit = (labels == K[true_c]) & (rng.random(labels.shape) < rate)
        logits[hit, K[wrong_c]] += sharpness * rng.uniform(1.0, 1.3, hit.sum())
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- 4. simple analysis

def touches(a, b):
    """Tiles of `a` that have a 4-neighbor in `b`."""
    return a & ndimage.binary_dilation(b, structure=CROSS)


def near(mask, radius=NEAR_TILES):
    if not mask.any():
        return mask
    return ndimage.distance_transform_edt(~mask) <= radius


def rc_list(rc):
    return [int(round(rc[0])), int(round(rc[1]))]


def side_and_clock(vec):
    """(row, col) direction -> ('bottom-right', 4) etc. 12 o'clock = top of the image."""
    dr, dc = vec
    names = ["right", "top-right", "top", "top-left", "left", "bottom-left", "bottom", "bottom-right"]
    side = names[int(np.round(np.arctan2(-dr, dc) / (np.pi / 4))) % 8]
    hour = int(np.round(np.arctan2(dc, -dr) / (2 * np.pi) * 12)) % 12
    return side, hour or 12


def analyze(pred, smooth_pred, ok, sc):
    """
    Simple stand-ins for the real analysis step. Returns (summary fields, risk points per tile,
    risk feature masks). Shapes use the smoothed labels; counts (tissue %, lymphocytes) the raw ones.
    """
    raw = {c: (pred == K[c]) & ok for c in CLASSES}
    is_ = {c: (smooth_pred == K[c]) & ok for c in CLASSES}
    n_tissue = int(ok.sum())
    out = {"tissue_fractions": {c: round(float(raw[c].sum() / n_tissue), 4) for c in CLASSES}}
    points = np.zeros(pred.shape, int)
    found = {}                                       # risk feature name -> boolean mask

    lab, n = ndimage.label(is_["tumor"], structure=SQUARE)
    sizes = ndimage.sum(is_["tumor"], lab, range(1, n + 1)) if n else []
    if n == 0 or max(sizes) < 20:
        out.update(tumor={"fraction_of_tissue": 0, "area_mm2": 0, "centroid_rc": None},
                   border={"n_border_tiles": 0, "solidity": None, "irregularity": None, "pattern": None},
                   invasion={"border_touch_fraction": {}, "deepest_layer": None,
                             "t_stage_hint": "No tumor found in this section.", "direction": None},
                   immune={"core_fraction": None, "margin_fraction": None, "margin_by_side": {},
                           "pattern": None},
                   necrosis={"fraction_in_tumor": None, "central": False},
                   stroma=None, satellites=[], high_risk_flags=[])
        return out, points, found

    main_id = int(np.argmax(sizes)) + 1
    main = lab == main_id
    region = ndimage.binary_fill_holes(main)          # the tumor including dead/immune spots inside
    cen = np.argwhere(region).mean(0)
    tile_mm2 = (TILE_UM / 1000) ** 2
    out["tumor"] = {"fraction_of_tissue": round(float(raw["tumor"].sum() / n_tissue), 4),
                    "area_mm2": round(float(raw["tumor"].sum() * tile_mm2), 2),
                    "centroid_rc": rc_list(cen)}

    # border shape, measured on a cleaned-up outline (small gaps and holes filled)
    non_tumor_tissue = ok & ~region
    edge = touches(main, non_tumor_tissue)
    front = touches(main, is_["stroma"] | is_["muscle"] | is_["fat"])   # the invasive front
    shape = ndimage.binary_fill_holes(ndimage.binary_closing(main, SQUARE) | main)
    pts = np.argwhere(shape).astype(float)
    corners = np.vstack([pts + d for d in [(-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)]])
    solidity = float(shape.sum() / ConvexHull(corners).volume)
    perimeter = sum(int((shape & ~np.roll(shape, s, axis=a)).sum()) for a in (0, 1) for s in (1, -1))
    irregularity = perimeter / (2 * np.sqrt(np.pi * shape.sum()))
    pattern = "infiltrative" if solidity < 0.85 else "pushing"
    out["border"] = {"n_border_tiles": int(edge.sum()), "solidity": round(solidity, 3),
                     "irregularity": round(float(irregularity), 2), "pattern": pattern}

    # satellites: other tumor pieces of 2+ tiles
    sats, sat_mask = [], np.zeros_like(main)
    main_dist = ndimage.distance_transform_edt(~main)
    for i in range(1, n + 1):
        piece = lab == i
        if i == main_id or piece.sum() < 2:
            continue
        sat_mask |= piece
        ring = ndimage.binary_dilation(piece, structure=SQUARE, iterations=2) & ~piece & ok
        around = CLASSES[int(np.bincount(pred[ring], minlength=len(CLASSES)).argmax())]
        sats.append({"center_rc": rc_list(np.argwhere(piece).mean(0)), "n_tiles": int(piece.sum()),
                     "distance_mm": round(float(main_dist[piece].min() * TILE_UM / 1000), 2),
                     "surrounded_by": around})
    out["satellites"] = sats

    # depth and direction
    touch = {c: round(float(touches(main, is_[c] & ~region).sum() / max(edge.sum(), 1)), 3)
             for c in ["muscle", "fat", "normal_mucosa", "stroma"]}
    sat_in_fat = any(s["surrounded_by"] == "fat" for s in sats)
    if touch["fat"] >= 0.03 or sat_in_fat:
        deepest, hint = "fat", "Consistent with at least pT3 (tumor reaches fat in this section)."
    elif touch["muscle"] >= 0.03:
        deepest, hint = "muscle", "Reaches muscle: consistent with pT2 or deeper."
    else:
        deepest, hint = "mucosa", "No muscle or fat contact in this section."
    deep = touches(main, is_["muscle"] | is_["fat"]) | sat_mask
    target = np.argwhere(deep if deep.any() else front).mean(0)
    vec = (target - cen) / (np.linalg.norm(target - cen) + 1e-9)
    side, hour = side_and_clock(vec)
    out["invasion"] = {"border_touch_fraction": touch, "deepest_layer": deepest, "t_stage_hint": hint,
                       "direction": {"side": side, "clock": hour, "vector_rc": [round(float(v), 2) for v in vec]}}

    # immune: lymphocytes in the tumor core vs the margin band (500 µm either side of the border)
    band = 500 / TILE_UM
    inside = ndimage.distance_transform_edt(region)
    outside = ndimage.distance_transform_edt(~region)
    core = region & (inside > band) & ok & ~is_["necrosis"]
    margin = ok & ((region & (inside <= band)) | (~region & (outside <= band)))
    lym = raw["lymphocytes"]
    core_f = float(lym[core].mean()) if core.any() else 0.0
    margin_f = float(lym[margin].mean()) if margin.any() else 0.0
    rr, cc = np.mgrid[0:pred.shape[0], 0:pred.shape[1]]
    ang = np.degrees(np.arctan2(-(rr - cen[0]), cc - cen[1]))     # 90 = up
    sides = {"top": (ang >= 45) & (ang < 135), "left": (ang >= 135) | (ang < -135),
             "bottom": (ang >= -135) & (ang < -45), "right": (ang >= -45) & (ang < 45)}
    by_side = {s: round(float(lym[margin & m].mean()), 3) if (margin & m).any() else None
               for s, m in sides.items()}
    if core_f >= 0.08:
        immune_pattern = "inflamed"
    elif margin_f >= 0.12 and core_f < margin_f / 3:
        immune_pattern = "excluded"
    else:
        immune_pattern = "desert"
    out["immune"] = {"core_fraction": round(core_f, 3), "margin_fraction": round(margin_f, 3),
                     "margin_by_side": by_side, "pattern": immune_pattern}

    # necrosis inside the tumor
    nec_in = is_["necrosis"] & region
    central = False
    if nec_in.any():
        gap = np.linalg.norm(np.argwhere(nec_in).mean(0) - cen)
        central = bool(gap < 0.35 * np.sqrt(region.sum() / np.pi))
    out["necrosis"] = {"fraction_in_tumor": round(float(nec_in.sum() / region.sum()), 3),
                       "central": central}

    # tumor-stroma ratio: most stroma-rich ~2 mm field along the front
    rad = 1000 / TILE_UM
    kernel = disk((rad, rad), rad)[: int(2 * rad) + 1, : int(2 * rad) + 1].astype(float)
    bed = ndimage.binary_dilation(region, iterations=2)
    s_count = ndimage.convolve((is_["stroma"] & bed).astype(float), kernel, mode="constant")
    t_count = ndimage.convolve((is_["tumor"] & bed).astype(float), kernel, mode="constant")
    ratio = np.where((t_count >= 10) & front, s_count / np.maximum(s_count + t_count, 1), 0)
    worst = np.unravel_index(int(ratio.argmax()), ratio.shape)
    out["stroma"] = {"worst_field_stroma_fraction": round(float(ratio[worst]), 3),
                     "worst_field_center_rc": [int(worst[0]), int(worst[1])],
                     "stroma_high": bool(ratio[worst] > 0.5)}

    # risk features -> points per tile
    front_density = ndimage.uniform_filter(front.astype(float), 7)
    found["infiltrative_front"] = front & (front_density > 0.2)
    found["front_fat"] = touches(main, is_["fat"])
    found["satellite"] = sat_mask
    found["necrosis_rim"] = main & near(nec_in, 2) if nec_in.any() else np.zeros_like(main)
    found["stroma_high"] = disk(worst, NEAR_TILES) & ok if out["stroma"]["stroma_high"] else np.zeros_like(main)
    if pattern == "infiltrative" and found["infiltrative_front"].any():
        hot = np.unravel_index(int(np.where(front, front_density, 0).argmax()), front.shape)
        found["budding_hotspot"] = disk(hot, 4.5) & ok
    if sc["venous_satellite"] is not None and sc["venous_satellite"] < len(sats):
        r, c = sats[sc["venous_satellite"]]["center_rc"]
        found["venous_candidate"] = (lab == lab[r, c]) & (lab > 0)
    for name, mask in found.items():
        points += RISK_FEATURES[name][0] * near(mask)
    points[~ok] = 0

    flags = []
    if pattern == "infiltrative":
        flags.append(f"Infiltrative border (solidity {solidity:.2f})")
    if deepest == "fat":
        flags.append("Tumor reaches fat: consistent with at least pT3")
    if "venous_candidate" in found:
        flags.append("Possible venous invasion: 1 spot to check (consider an elastin stain)")
    if sats:
        flags.append(f"{len(sats)} tumor island(s) near the main mass: check for tumor deposits")
    if out["stroma"]["stroma_high"]:
        flags.append(f"Stroma-high front ({ratio[worst]:.0%} stroma in the worst field)")
    if "budding_hotspot" in found:
        flags.append("Possible budding hotspot: count buds there (experimental)")
    out["high_risk_flags"] = flags
    return out, points, found


def risk_regions(points, found, ok, min_tiles=3):
    """
    Group tiles into regions, one card each: connected patches of the same level, highest
    level first (so a level-4 spot inside a level-2 area gets its own card).
    """
    level = np.clip(points, 0, 4)
    regions = []
    for top in (4, 3, 2, 1):
        lab, n = ndimage.label(level == top, structure=SQUARE)
        for i in range(1, n + 1):
            core = lab == i
            if core.sum() < min_tiles:
                continue
            cells = np.argwhere(core)
            nearest = cells[np.argmin(np.hypot(*(cells - cells.mean(0)).T))]
            reasons = [name for name in RISK_FEATURES if name in found and (near(found[name]) & core).any()]
            (r0, c0), (r1, c1) = cells.min(0), cells.max(0)
            regions.append({
                "level": top, "points": int(points[core].max()), "n_tiles": int(core.sum()),
                "title": RISK_FEATURES[reasons[0]][1] if reasons else "Risk features nearby",
                "reasons": [RISK_FEATURES[r][1] for r in reasons],
                "suggested_actions": [RISK_FEATURES[r][3] for r in reasons],
                "center_rc": [int(nearest[0]), int(nearest[1])],
                "bbox_rc": [int(r0), int(c0), int(r1), int(c1)],
            })
    regions.sort(key=lambda g: (-g["level"], -g["n_tiles"]))
    for i, g in enumerate(regions, 1):
        g["id"] = i
    return regions


# ---------------------------------------------------------------- 5. fake pictures

def paint_thumbnail(labels, reason, rng):
    """A fake H&E-looking thumbnail, one tile = THUMB_TILE x THUMB_TILE pixels."""
    lab = np.repeat(np.repeat(labels, THUMB_TILE, 0), THUMB_TILE, 1)
    rgb = np.empty(lab.shape + (3,))
    rgb[:] = GLASS_RGB
    dens = np.zeros(lab.shape)
    for c, color in HE_COLORS.items():
        rgb[lab == K[c]] = color
        dens[lab == K[c]] = NUCLEI[c]
    for ch in range(3):                                  # soft edges, not squares
        rgb[..., ch] = ndimage.gaussian_filter(rgb[..., ch], 2.5)
    tissue = ndimage.gaussian_filter((lab != GLASS).astype(float), 2.5)
    stain = 1 + 0.06 * ndimage.gaussian_filter(rng.normal(size=lab.shape), 8) * 8
    rgb *= np.clip(stain, 0.85, 1.15)[..., None] * tissue[..., None] + (1 - tissue[..., None])
    dots = rng.random(lab.shape) < ndimage.gaussian_filter(dens, 2) * 0.6
    rgb[dots] = rgb[dots] * 0.45 + np.array([60, 30, 100]) * 0.55     # nuclei
    for ch in range(3):
        rgb[..., ch] = ndimage.gaussian_filter(rgb[..., ch], 0.6)
    rgb += rng.normal(0, 3, rgb.shape)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    pen = np.argwhere(reason == "pen mark")
    if len(pen):
        d = ImageDraw.Draw(img)
        xy = [(c * THUMB_TILE + THUMB_TILE // 2, r * THUMB_TILE + THUMB_TILE // 2) for r, c in pen[::2]]
        d.line(xy, fill=(40, 110, 90), width=6)
    return img


def fake_preview(label, seed, size=128):
    """A fake 'microscope' picture of one tile, for DEMO slides only (real ones come from Delta)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), HE_COLORS.get(label, GLASS_RGB))
    d = ImageDraw.Draw(img)
    nucleus = (70, 35, 110)

    def dot(x, y, r, color=nucleus):
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    if label in ("tumor", "normal_mucosa"):
        normal = label == "normal_mucosa"
        for _ in range(3 if normal else 5):
            cx, cy = rng.uniform(10, size - 10, 2)
            rx, ry = rng.uniform(12, 18, 2) if normal else rng.uniform(6, 20, 2)
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(236, 220, 232))   # gland opening
            for t in np.linspace(0, 2 * np.pi, 18 if normal else 28, endpoint=False):
                wobble = 0 if normal else rng.normal(0, 2.5)
                dot(cx + (rx + 3 + wobble) * np.cos(t), cy + (ry + 3 + wobble) * np.sin(t),
                    2 if normal else rng.uniform(2, 3.6))
    elif label == "lymphocytes":
        for _ in range(260):
            dot(*rng.uniform(0, size, 2), rng.uniform(1.6, 2.4), (60, 35, 115))
    elif label == "fat":
        for _ in range(9):
            cx, cy = rng.uniform(0, size, 2)
            r = rng.uniform(14, 26)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(248, 244, 247), outline=(214, 140, 170), width=2)
    elif label in ("muscle", "stroma"):
        base = rng.uniform(0, np.pi)
        for _ in range(40):                                   # fibers
            x, y = rng.uniform(0, size, 2)
            a = base + (rng.normal(0, 0.1) if label == "muscle" else rng.normal(0, 0.6))
            d.line([x - 30 * np.cos(a), y - 30 * np.sin(a), x + 30 * np.cos(a), y + 30 * np.sin(a)],
                   fill=(235, 175, 198) if label == "stroma" else (190, 80, 125), width=2)
        for _ in range(30 if label == "muscle" else 14):      # nuclei
            x, y = rng.uniform(0, size, 2)
            a = base + (rng.normal(0, 0.1) if label == "muscle" else rng.normal(0, 0.6))
            d.line([x - 5 * np.cos(a), y - 5 * np.sin(a), x + 5 * np.cos(a), y + 5 * np.sin(a)],
                   fill=nucleus, width=3 if label == "muscle" else 2)
    elif label == "mucus":
        for _ in range(12):
            x, y = rng.uniform(0, size, 2)
            d.arc([x - 20, y - 10, x + 20, y + 10], rng.uniform(0, 180), rng.uniform(180, 360),
                  fill=(170, 175, 210), width=2)
        for _ in range(4):
            dot(*rng.uniform(0, size, 2), 2)
    elif label == "necrosis":
        for _ in range(25):
            dot(*rng.uniform(0, size, 2), rng.uniform(4, 9), (215, 160, 180))
        for _ in range(90):
            dot(*rng.uniform(0, size, 2), rng.uniform(0.6, 1.6))
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    arr = np.asarray(img, float) + rng.normal(0, 5, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- put it together

def make_slide(sc, out_dir):
    rng = np.random.default_rng(sc["seed"])
    truth = true_labels(sc, rng)
    reason = qc_marks(truth, sc, rng)
    tissue = truth != GLASS
    ok = tissue & (reason == "")

    conch = fake_model(truth, rng, sharpness=9.0, mixups=CONCH_MIXUPS)
    musk = fake_model(truth, rng, sharpness=3.5, mixups=MUSK_MIXUPS)
    probs = (conch + musk) / 2                     # stand-in for the final classifier
    pred = np.where(tissue, probs.argmax(-1), GLASS)    # the map label shown on hover

    # the analysis uses lightly smoothed labels (3x3 average of the probabilities), like the
    # real pipeline, so single noisy tiles don't count as "tumor touches fat"
    weight = ndimage.uniform_filter(tissue.astype(float), 3)[..., None]
    smooth = ndimage.uniform_filter(probs * tissue[..., None], (3, 3, 1)) / np.maximum(weight, 1e-9)
    fields, points, found = analyze(pred, np.where(tissue, smooth.argmax(-1), GLASS), ok, sc)
    regions = risk_regions(points, found, ok)
    level = np.clip(points, 0, 4)

    rows, cols = np.nonzero(tissue)
    table = {"row": rows, "col": cols, "x": cols * STEP, "y": rows * STEP, "tile_size_px": TILE_PX,
             "qc_ok": ok[rows, cols], "qc_reason": reason[rows, cols]}
    for k, c in enumerate(CLASSES):
        table[f"p_{c}"] = probs[rows, cols, k].round(4)
    table["label"] = [CLASSES[k] for k in pred[rows, cols]]
    table["confidence"] = probs[rows, cols].max(-1).round(4)
    for name, p in (("conch", conch), ("musk", musk)):
        for k, c in enumerate(CLASSES):
            table[f"p_{name}_{c}"] = p[rows, cols, k].round(4)
    table["risk_points"] = points[rows, cols]
    table["risk_level"] = level[rows, cols]
    nearby = {name: near(mask) for name, mask in found.items()}
    table["risk_flags"] = ["; ".join(RISK_FEATURES[name][2] for name in RISK_FEATURES
                                     if name in nearby and nearby[name][r, c] and ok[r, c])
                           for r, c in zip(rows, cols)]

    summary = {
        "slide_id": sc["slide_id"], "display_name": sc["name"], "case_id": sc["case_id"],
        "processed_at": sc["entered"], "is_demo": True,
        "mpp": MPP, "slide_mpp": SLIDE_MPP, "tile_size_px": TILE_PX, "thumbnail_downsample": DOWNSAMPLE,
        "grid_shape": [ROWS, COLS], "n_tissue_tiles": int(tissue.sum()),
        **{k: fields[k] for k in ["tissue_fractions", "tumor", "border", "invasion", "immune", "necrosis"]},
        "stroma": fields["stroma"], "satellites": fields["satellites"],
        "high_risk_flags": fields["high_risk_flags"],
        "risk_regions": regions, "risk_points_version": "demo-placeholder",
        "caveats": ["DEMO DATA: every number on this slide is made up, to build the app."] + CAVEATS,
        "model": {"name": "DEMO: fake CONCH + MUSK average", "prompts_version": "v1"},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paint_thumbnail(truth, reason, rng).save(out_dir / "thumbnail.png")
    pd.DataFrame(table).to_csv(out_dir / "tiles.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_all():
    for sc in SCENARIOS:
        s = make_slide(sc, DEMO_DIR / sc["slide_id"])
        print(f"{s['slide_id']}: {s['n_tissue_tiles']} tissue tiles, tumor {s['tumor']['fraction_of_tissue']:.0%}, "
              f"border {s['border']['pattern']} (solidity {s['border']['solidity']}), "
              f"immune {s['immune']['pattern']}, {len(s['risk_regions'])} risk regions, "
              f"flags: {s['high_risk_flags']}")


def ensure_demo_data():
    """Make the demo slides if they aren't there yet (first launch)."""
    if not all((DEMO_DIR / sc["slide_id"] / "summary.json").exists() for sc in SCENARIOS):
        make_all()


if __name__ == "__main__":
    make_all()
    print("saved to", DEMO_DIR)
