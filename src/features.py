"""
Section 3 of the plan: everything computed from the tile map, with no new model.

Each function takes the TileMap (`tm`, src/tilemap.py) and the Front (`fr`, src/front.py) and returns
a plain dict that goes into summary.json. Some also return [R, C] grids that the risk map and the
pictures use. Distances are in µm; all cutoffs live in src/cutoffs.py.

Directions are "on the slide image" (12 o'clock = up on the thumbnail). A slide has no anatomical up,
and it is one 2D section: direction = where the front faces now, never where it will grow.
"""
import math
from collections import Counter

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from src import cutoffs
from src.front import (BORDER_FRONT, BORDER_MUCOSA, ZONE_CORE, ZONE_FRONT, ZONE_PERI, dilate, disk,
                       neighbors_of)
from src.prompts import CLASSES
from src.tilemap import CLASS_INDEX, NO_TISSUE


# --- helpers --------------------------------------------------------------------------------

def rnd(value, digits=3):
    return None if value is None else round(float(value), digits)


def bearing(dr, dc):
    """Direction on the slide image in degrees clockwise from up (0 = 12 o'clock, 90 = 3 o'clock)."""
    return float(np.degrees(np.arctan2(dc, -dr)) % 360)


def clock(bearing_deg):
    hour = int(round(bearing_deg / 30.0)) % 12
    return 12 if hour == 0 else hour


def angle_gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def class_mix(tm, mask):
    """Share of each class among the tissue tiles in `mask` (smoothed labels)."""
    labels = tm.label[mask & tm.tissue]
    counts = np.bincount(labels[labels >= 0], minlength=len(CLASSES))
    total = counts.sum()
    return {name: (float(counts[i]) / total if total else 0.0) for i, name in enumerate(CLASSES)}


def ring_of(mask, tm):
    """The tissue tiles touching `mask` from outside."""
    return dilate(mask) & ~mask & tm.tissue


def center_tile(coords):
    """The tile of a region closest to its centroid (the centroid itself can fall outside the region)."""
    coords = np.asarray(coords)
    centroid = coords.mean(axis=0)
    return coords[np.argmin(((coords - centroid) ** 2).sum(axis=1))]


def mm(um):
    return f"{um / 1000:.1f} mm"


# --- whole slide ----------------------------------------------------------------------------

def tissue_fractions(tm):
    return {name: rnd(share) for name, share in class_mix(tm, tm.tissue).items()}


def uncertainty(tm):
    """
    Where the model is unsure: top probability below cutoffs.LOW_CONFIDENCE. A second model's label (alt_label) is
    not used: CONCH/MUSK disagreement marked 20-28% of the tissue on 125 TCGA slides, mostly MUSK over-calling necrosis.
    """
    low = tm.tissue & (tm.confidence < cutoffs.LOW_CONFIDENCE)
    unsure = low

    regions = []
    lab = measure.label(unsure, connectivity=2)
    for p in sorted(measure.regionprops(lab), key=lambda p: -p.area)[:3]:
        r, c = center_tile(p.coords)
        regions.append({**tm.tile_ref(r, c), "n_tiles": int(p.area), "why": "low confidence"})

    frac = unsure.sum() / max(tm.tissue.sum(), 1)
    text = f"Unsure on {frac:.0%} of tissue tiles."
    if regions:
        text += f" Largest unsure area: {regions[0]['n_tiles']} tiles ({regions[0]['why']})."
    return {"fraction_unsure": rnd(frac), "fraction_low_confidence": rnd(frac),
            "fraction_models_disagree": None,                   # kept for older readers; no longer used
            "regions": regions, "text": text}, unsure


def lymph_nodes(tm, fr):
    """Large lymphocyte nodules (> ~1 mm) are probably lymph nodes; tumor inside one = possible metastasis."""
    lab = measure.label(tm.class_mask("lymphocytes"), connectivity=2)
    nodes, node_mask = [], np.zeros(tm.shape, bool)
    for p in measure.regionprops(lab):
        diameter = math.sqrt(4 * p.area / math.pi) * tm.tile_um
        if diameter < cutoffs.NODE_MIN_DIAMETER_UM:
            continue
        filled = ndimage.binary_fill_holes(lab == p.label)
        node_mask |= filled
        tumor_inside = int((fr.tumor & filled).sum()) if fr is not None else 0
        r, c = center_tile(p.coords)
        text = f"Probable lymph node, {mm(diameter)} across (or a large lymphoid aggregate)."
        if tumor_inside:
            text += " Possible node metastasis: tumor tiles inside a lymphocyte-rich nodule."
        nodes.append({**tm.tile_ref(r, c), "diameter_um": round(diameter), "area_mm2": rnd(p.area * tm.tile_mm2),
                      "tumor_tiles_inside": tumor_inside, "possible_metastasis": tumor_inside > 0, "text": text})
    return nodes, node_mask


# --- tumor ----------------------------------------------------------------------------------

def tumor_summary(tm, fr):
    in_section = fr.in_section
    centroid = ndimage.center_of_mass(fr.main_filled)
    return {"found": True,
            "fraction_of_tissue": rnd((fr.tumor & in_section).sum() / max(in_section.sum(), 1)),
            "area_mm2": rnd(fr.main_filled.sum() * tm.tile_mm2, 2),
            "centroid_rc": [round(centroid[0], 1), round(centroid[1], 1)],
            "n_tumor_pieces": int(fr.pieces.max()),
            "n_tissue_sections": fr.n_sections, "analyzed_section": fr.section_id}


def tumor_pieces(tm, fr, node_mask):
    """Every tumor piece that isn't the main tumor: satellites, possible deposits, node metastases."""
    dist_tiles, nearest_main = ndimage.distance_transform_edt(~fr.main_filled, return_indices=True)
    dist_to_main = dist_tiles * tm.tile_um
    front = fr.border_type == BORDER_FRONT
    near_front = None
    if front.any():
        _, near_front = ndimage.distance_transform_edt(~front, return_indices=True)

    pieces = []
    for p in measure.regionprops(fr.pieces):
        if p.label == fr.main_id:
            continue
        m = fr.pieces == p.label
        mix = class_mix(tm, ring_of(m, tm))
        r, c = center_tile(p.coords)
        gap = max(0.0, float(dist_to_main[m].min()) - tm.tile_um)   # roughly edge to edge
        through_tissue = tissue_between(tm, (r, c), (nearest_main[0][r, c], nearest_main[1][r, c]))
        in_node = bool((m & node_mask).any())
        direction = None
        if near_front is not None:
            fr_r, fr_c = near_front[0][r, c], near_front[1][r, c]
            direction = bearing(r - fr_r, c - fr_c)
        pieces.append({
            "id": int(p.label), **tm.tile_ref(r, c),
            "area_mm2": rnd(p.area * tm.tile_mm2),
            "diameter_um": round(math.sqrt(4 * p.area / math.pi) * tm.tile_um),
            "solidity": rnd(p.solidity, 2),
            "distance_to_main_um": round(gap),
            "sits_in": max(mix, key=mix.get) if any(mix.values()) else "unknown",
            "fat_fraction_around": rnd(mix["fat"], 2),
            "tissue_between": rnd(through_tissue, 2),
            "satellite": gap <= cutoffs.SATELLITE_MAX_UM and through_tissue >= cutoffs.SATELLITE_MIN_TISSUE,
            "possible_deposit": mix["fat"] >= cutoffs.DEPOSIT_RING_FAT and not in_node,
            "in_lymph_node": in_node,
            "bearing_deg": rnd(direction, 1),
        })
    return pieces


def tissue_between(tm, a, b, samples=32):
    """Share of the straight line from tile a to tile b that runs through tissue (not glass, a tear or a gap)."""
    t = np.linspace(0, 1, samples)
    rr = np.rint(a[0] + (b[0] - a[0]) * t).astype(int)
    cc = np.rint(a[1] + (b[1] - a[1]) * t).astype(int)
    return float(tm.tissue[rr, cc].mean())


# --- growth pattern -------------------------------------------------------------------------

def _front_contour(tm, fr, front_near):
    """The main tumor's outline, its deviation from a smoothed outline, and which points are front."""
    contours = measure.find_contours(np.pad(fr.main_filled, 1).astype(float), 0.5)
    if not contours:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool)
    contour = max(contours, key=len) - 1.0               # back to grid (row, col) coordinates
    if len(contour) > 1 and np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    spacing = np.linalg.norm(np.diff(contour, axis=0), axis=1).mean() if len(contour) > 1 else 1.0
    window = max(3, int(round(tm.tiles(cutoffs.ROUGHNESS_SMOOTH_UM) / max(spacing, 1e-6))))
    window = min(window | 1, max(3, (len(contour) // 2) | 1))
    smooth = ndimage.uniform_filter1d(contour, size=window, axis=0, mode="wrap")
    deviation = np.linalg.norm(contour - smooth, axis=1)
    R, C = tm.shape
    rr = np.clip(np.round(contour[:, 0]).astype(int), 0, R - 1)
    cc = np.clip(np.round(contour[:, 1]).astype(int), 0, C - 1)
    return contour, deviation, front_near[rr, cc]


def box_count_dimension(mask, sizes=(1, 2, 4, 8, 16)):
    """
    Fractal dimension of a line by box counting: ~1.0 for a smooth line, higher for a jagged one.
    Box sizes that hit only a few boxes are skipped: on a short front they bias the slope down.
    """
    R, C = mask.shape
    used, counts = [], []
    for s in sizes:
        Rp, Cp = -(-R // s) * s, -(-C // s) * s
        padded = np.zeros((Rp, Cp), bool)
        padded[:R, :C] = mask
        n = padded.reshape(Rp // s, s, Cp // s, s).any(axis=(1, 3)).sum()
        if n >= cutoffs.FRACTAL_MIN_BOXES:
            used.append(s)
            counts.append(n)
    if len(used) < 3:
        return None
    return float(np.polyfit(np.log(1.0 / np.array(used, float)), np.log(np.array(counts, float)), 1)[0])


def find_fingers(tm, fr, front_near):
    """Thin protrusions: tumor that an opening with a 3-tile disk removes, on the front only."""
    r = cutoffs.FINGER_DISK_TILES
    opened = ndimage.binary_opening(np.pad(fr.main_filled, r + 1), structure=disk(r))[r + 1:-(r + 1), r + 1:-(r + 1)]
    if not opened.any():                                  # tumor smaller than the disk: can't measure
        return np.zeros(tm.shape, int), []
    lab = measure.label(fr.main_filled & ~opened, connectivity=2)
    fingers = [p for p in measure.regionprops(lab)
               if p.area >= cutoffs.MIN_FINGER_TILES and (front_near & (lab == p.label)).any()]
    keep = [p.label for p in fingers]
    return np.where(np.isin(lab, keep), lab, 0), fingers


def front_arcs(tm, fr, contour, deviation, is_front_pt, fingers, pieces):
    """Split the front into ~1 mm arcs and score how infiltrative each one is."""
    if len(contour) == 0 or not is_front_pt.any():
        return [], {}
    # outward direction everywhere: gradient of a smoothed signed distance to the tumor outline
    field = ndimage.gaussian_filter(ndimage.distance_transform_edt(~fr.main_filled)
                                    - ndimage.distance_transform_edt(fr.main_filled), sigma=2)
    grad_r, grad_c = np.gradient(field)
    centroid = np.array(ndimage.center_of_mass(fr.main_filled))

    # walk the outline, starting at a non-front point so no arc is cut where the outline starts
    n = len(contour)
    start = int(np.flatnonzero(~is_front_pt)[0]) if (~is_front_pt).any() else 0
    step = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)
    groups, current, length = [], [], 0.0
    for i in np.roll(np.arange(n), -start):
        if is_front_pt[i]:
            current.append(i)
            length += step[i]
            if length >= tm.tiles(cutoffs.ARC_LENGTH_UM):
                groups.append(current)
                current, length = [], 0.0
        elif current:
            groups.append(current)
            current, length = [], 0.0
    if current:
        groups.append(current)
    groups = [g for g in groups if len(g) >= 3]

    finger_centers = np.array([f.centroid for f in fingers]) if fingers else np.zeros((0, 2))
    satellite_centers = np.array([[p["row"], p["col"]] for p in pieces if p["satellite"]]) if pieces else np.zeros((0, 2))
    near_tiles = tm.tiles(cutoffs.ARC_NEIGHBOR_UM)
    front = fr.border_type == BORDER_FRONT
    R, C = tm.shape
    arcs, arc_masks = [], {}
    for k, g in enumerate(groups):
        pts = contour[g]
        center = pts.mean(axis=0)
        r, c = int(np.clip(round(center[0]), 0, R - 1)), int(np.clip(round(center[1]), 0, C - 1))
        normal = np.array([grad_r[r, c], grad_c[r, c]])
        if np.linalg.norm(normal) < 1e-9:
            normal = center - centroid
        normal = normal / max(np.linalg.norm(normal), 1e-9)
        roughness = float(deviation[g].mean()) * tm.tile_um
        n_fingers = int((np.linalg.norm(finger_centers - center, axis=1) <= near_tiles).sum()) if len(finger_centers) else 0
        n_sats = int((np.linalg.norm(satellite_centers - center, axis=1) <= near_tiles).sum()) if len(satellite_centers) else 0
        infiltrative = roughness >= cutoffs.ARC_ROUGHNESS_INFILTRATIVE_UM or n_fingers > 0 or n_sats > 0
        direction = bearing(normal[0], normal[1])
        arcs.append({"id": k, "center_rc": [round(center[0], 1), round(center[1], 1)],
                     "normal_rc": [round(normal[0], 3), round(normal[1], 3)],
                     "bearing_deg": round(direction, 1), "clock": clock(direction),
                     "length_um": round(float(step[g].sum()) * tm.tile_um),
                     "roughness_um": round(roughness, 1), "n_fingers_near": n_fingers,
                     "n_satellites_near": n_sats, "infiltrative": bool(infiltrative),
                     "score": round(roughness / cutoffs.ARC_ROUGHNESS_INFILTRATIVE_UM + n_fingers + n_sats, 2)})
        m = np.zeros(tm.shape, bool)
        m[np.clip(np.round(pts[:, 0]).astype(int), 0, R - 1), np.clip(np.round(pts[:, 1]).astype(int), 0, C - 1)] = True
        arc_masks[k] = dilate(m) & front
    return arcs, arc_masks


def growth_pattern(tm, fr, pieces):
    """Pushing vs infiltrative, measured on the front only (never the luminal surface)."""
    front = fr.border_type == BORDER_FRONT
    front_near = dilate(front)
    solidity = float(measure.regionprops(fr.main_filled.astype(np.uint8))[0].solidity)
    contour, deviation, is_front_pt = _front_contour(tm, fr, front_near)
    roughness = float(deviation[is_front_pt].mean()) * tm.tile_um if is_front_pt.any() else None
    fractal = box_count_dimension(front) if front.sum() >= 10 else None
    finger_lab, fingers = find_fingers(tm, fr, front_near)
    satellites = [p for p in pieces if p["satellite"]]
    arcs, arc_masks = front_arcs(tm, fr, contour, deviation, is_front_pt, fingers, pieces)

    # plain rule over the measures (always shown with the numbers behind it); cutoffs are placeholders
    protrusions = len(fingers) + len(satellites)
    infiltrative = solidity < cutoffs.SOLIDITY_INFILTRATIVE or protrusions >= cutoffs.INFILTRATIVE_MIN_PROTRUSIONS
    pattern = "infiltrative" if infiltrative else "pushing"
    text = (f"{pattern.capitalize()}: solidity {solidity:.2f}, {len(fingers)} finger protrusions, "
            f"{len(satellites)} satellites within {mm(cutoffs.SATELLITE_MAX_UM)}.")
    front_um = float(front.sum()) * tm.tile_um
    if front_um < cutoffs.FRONT_MIN_UM:      # e.g. a tumor-only fragment: the "front" is a few tiles, nothing to judge
        pattern, arcs, arc_masks = "not_assessable", [], {}
        text = (f"Can't judge the growth pattern: only {mm(front_um)} of invasive front in this section (needs "
                f"{mm(cutoffs.FRONT_MIN_UM)}); the tumor mostly meets the tissue edge, the lumen or the lining here.")
    summary = {"n_border_tiles": int((fr.border_type > 0).sum()), "n_front_tiles": int(front.sum()),
               "solidity": rnd(solidity), "irregularity": rnd(roughness, 1), "roughness_um": rnd(roughness, 1),
               "fractal_dimension": rnd(fractal), "n_fingers": len(fingers),
               "n_satellites_within_1mm": len(satellites), "pattern": pattern, "pattern_text": text,
               "pattern_cutoff_calibrated": False, "arcs": arcs}
    grids = {"fingers": finger_lab > 0, "arc_masks": arc_masks, "contour": contour}
    return summary, grids


# --- direction, what lies ahead, depth ------------------------------------------------------

def direction(arcs, pieces):
    """Which way the most infiltrative part of the front faces, plus a 16-bin rose diagram."""
    if not arcs:
        return {"side": None, "clock": None, "bearing_deg": None, "vector_rc": None, "rose_16": [0.0] * 16,
                "text": "Can't judge the direction: no invasive front long enough in this section."}
    infiltrative = [a for a in arcs if a["infiltrative"]]
    use = infiltrative or arcs
    rose = np.zeros(16)
    for a in use:
        weight = a["score"] if infiltrative else a["length_um"]
        rose[int(((a["bearing_deg"] + 11.25) % 360) // 22.5)] += max(weight, 1e-6)
    main_bearing = float(rose.argmax() * 22.5)
    hour = clock(main_bearing)
    satellites = [p for p in pieces if p["satellite"] and p["bearing_deg"] is not None]
    n_same_side = sum(angle_gap(p["bearing_deg"], main_bearing) <= 45 for p in satellites)
    if infiltrative:
        text = f"Front most infiltrative toward {hour} o'clock (on the slide image)"
        text += f", where {n_same_side} of {len(satellites)} satellites lie." if satellites else "."
    else:
        text = f"No clearly infiltrative arc; most of the front faces {hour} o'clock (on the slide image)."
    rad = math.radians(main_bearing)
    return {"side": f"{hour} o'clock", "clock": hour, "bearing_deg": main_bearing,
            "vector_rc": [round(-math.cos(rad), 3), round(math.sin(rad), 3)],
            "rose_16": [round(float(v), 3) for v in rose / max(rose.sum(), 1e-9)],
            "n_infiltrative_arcs": len(infiltrative), "text": text}


def heading_into(tm, fr, arcs):
    """Step outward from each infiltrative arc (112 µm steps, up to 2 mm) and record what tissue is hit."""
    use = [a for a in arcs if a["infiltrative"]] or arcs
    R, C = tm.shape
    counts, nearest = Counter(), {}
    ray = np.zeros(tm.shape, bool)
    for a in use:
        (r0, c0), (dr, dc) = a["center_rc"], a["normal_rc"]
        for s in range(1, int(round(tm.tiles(cutoffs.RAY_MAX_UM))) + 1):
            r, c = int(round(r0 + dr * s)), int(round(c0 + dc * s))
            if not (0 <= r < R and 0 <= c < C):
                break
            ray[r, c] = True
            lab = tm.label[r, c]
            if lab == NO_TISSUE or fr.main_filled[r, c]:
                continue
            name = CLASSES[lab]
            counts[name] += 1
            nearest[name] = min(nearest.get(name, math.inf), s * tm.tile_um)
    total = sum(counts.values())
    if not total:
        return {"mix": {}, "nearest_um": {}, "text": "Nothing but glass ahead of the front."}, ray
    mix = {name: rnd(n / total, 2) for name, n in counts.most_common()}
    text = "Facing " + ", ".join(f"{name} {share:.0%}" for name, share in list(mix.items())[:3]) + "."
    if "fat" in nearest:
        text += f" Nearest fat {mm(nearest['fat'])} ahead."
    return {"mix": mix, "nearest_um": {k: round(v) for k, v in nearest.items()}, "text": text,
            "based_on": "infiltrative arcs" if any(a["infiltrative"] for a in arcs) else "all front arcs"}, ray


DEPTH_PARAMS = ("WALL_MUSCLE_MIN_MM2", "WALL_FAT_MIN_MM2", "WALL_MIN_WIDTH_UM", "DEPTH_CONTACT_MIN_TILES",
                "DEPTH_CONTACT_RADIUS_TILES", "DEPTH_RULE", "DEPTH_BEYOND_MIN_UM", "DEPTH_BEYOND_MIN_TILES")
SEROSA_PARAMS = ("SEROSA_MIN_RUN_UM", "SEROSA_NEEDS_WALL", "SEROSA_MAX_FROM_WALL_UM")


def setting(params, name):
    """A tunable cut-off: the calibration passes params={name: value}; otherwise the value in src/cutoffs.py."""
    return params[name] if params and name in params else getattr(cutoffs, name)


def remembered(tm, key, compute):
    """
    Compute once per slide and settings, then reuse. The calibration asks for the depth hint thousands of times per
    slide (once per setting combination); the slow geometry only depends on a few of the settings.
    """
    cache = tm.__dict__.setdefault("_depth_cache", {})
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def wall_layers(tm, params=None):
    """
    The muscle wall and the fat around the bowel, from the raw (unsmoothed) labels: regions big enough to be real
    layers, so stray tiles and the thin muscularis mucosae don't count. Small gaps (one tile) are closed first.
    With WALL_MIN_WIDTH_UM > 0, a muscle region must also be thick on average (area / length of its midline): the
    muscularis propria is ~1-2 mm thick, the muscularis mucosae well under one tile.
    """
    def big(name, min_mm2, min_width_um=0):
        m = (tm.raw_label == CLASS_INDEX[name]) & tm.tissue
        m = ndimage.binary_closing(m, structure=np.ones((3, 3), bool)) & tm.tissue
        lab = measure.label(m, connectivity=2)
        keep = np.zeros(tm.shape, bool)
        for p in measure.regionprops(lab):
            if p.area * tm.tile_mm2 < min_mm2:
                continue
            width_um = p.area / max(int(morphology.skeletonize(p.image).sum()), 1) * tm.tile_um
            if width_um >= min_width_um:
                keep[lab == p.label] = True
        return keep

    muscle_mm2, width_um, fat_mm2 = (setting(params, "WALL_MUSCLE_MIN_MM2"), setting(params, "WALL_MIN_WIDTH_UM"),
                                     setting(params, "WALL_FAT_MIN_MM2"))
    return (remembered(tm, ("wall", muscle_mm2, width_um), lambda: big("muscle", muscle_mm2, width_um)),
            remembered(tm, ("fat", fat_mm2), lambda: big("fat", fat_mm2)))


def wall_band(wall):
    """The muscle wall as one solid band: gaps between muscle bundles closed, pockets enclosed by muscle filled in."""
    r = cutoffs.WALL_BAND_CLOSE_TILES
    padded = np.pad(wall, r + 1)
    closed = ndimage.binary_closing(padded, structure=disk(r))[r + 1:-r - 1, r + 1:-r - 1]
    return ndimage.binary_fill_holes(closed | wall)


def lining_and_outer_edge(tm):
    """
    Landmarks for "which side of the wall is the inside". Lining = normal mucosa, plus the tissue's edge where
    mucosa or tumor dominate (the luminal surface; a big tumor often hides the mucosa). Outer edge = the rest of the
    tissue edge (fat / muscle / stroma: the outside of the bowel, or a cut). Holes inside the tissue don't count.
    """
    edge = tm.tissue & neighbors_of(~ndimage.binary_fill_holes(tm.tissue), off_grid=True)
    kernel = disk(2).astype(float)
    inner = (tm.class_mask("normal_mucosa") | tm.class_mask("tumor")).astype(float)
    outer = np.logical_or.reduce([tm.class_mask(n) for n in ("fat", "muscle", "stroma")]).astype(float)
    luminal = edge & (ndimage.convolve(inner, kernel, mode="constant") > ndimage.convolve(outer, kernel, mode="constant"))
    return luminal | tm.class_mask("normal_mucosa"), edge & ~luminal


def wall_crossed_um(band, start, ends, tile_um, samples=48):
    """For straight lines from one tile (start) to each of `ends` (two index arrays): µm of each line inside the band."""
    r0, c0 = start
    r1, c1 = ends
    t = np.linspace(0, 1, samples)[None, :]
    rr = np.rint(r0 + (r1 - r0)[:, None] * t).astype(int)
    cc = np.rint(c0 + (c1 - c0)[:, None] * t).astype(int)
    return band[rr, cc].mean(axis=1) * np.hypot(r1 - r0, c1 - c0) * tile_um


def beyond_wall(tm, fr, params=None):
    """
    Raw tumor tiles on the FAR side of the muscle wall (away from the lining), at least DEPTH_BEYOND_MIN_UM past it.
    Position, not contact: tumor in the fat is usually wrapped in scar-like stroma, so it rarely touches fat tiles,
    and fat inside the wall (submucosa) looks like fat outside it. A tumor tile counts when
      1. it lies DEPTH_BEYOND_MIN_UM or more outside the solid wall band (wall_band), and not in a lymph node;
      2. the straight line to the nearest lining tile crosses WALL_CROSS_MIN_UM or more of wall (quick check);
      3. of the lining tiles nearby (up to DEPTH_LINING_REACH_UM farther than the nearest), at least
         DEPTH_BEYOND_MAJORITY are behind the wall too (one stray lining tile can't decide the side);
      4. the nearest outside landmark (fat, or the outer tissue edge) is NOT behind the wall.
    Returns the mask of those tiles.
    """
    wall, _ = wall_layers(tm, params)
    min_um = setting(params, "DEPTH_BEYOND_MIN_UM")
    key = ("beyond", setting(params, "WALL_MUSCLE_MIN_MM2"), setting(params, "WALL_MIN_WIDTH_UM"), min_um)

    def compute():
        none = np.zeros(tm.shape, bool)
        lining, outer_edge = remembered(tm, ("lining",), lambda: lining_and_outer_edge(tm))
        if not wall.any() or not lining.any():
            return none
        band = wall_band(wall)
        past_um = ndimage.distance_transform_edt(~band) * tm.tile_um
        nodes = remembered(tm, ("nodes",), lambda: lymph_nodes(tm, fr)[1])
        raw_tumor = (tm.raw_label == CLASS_INDEX["tumor"]) & tm.tissue
        candidates = raw_tumor & (past_um >= min_um) & ~dilate(nodes, 1)                                # 1
        if not candidates.any():
            return none
        outside = (outer_edge | (tm.raw_label == CLASS_INDEX["fat"])) & tm.tissue & ~band
        to_lining, near_lining = ndimage.distance_transform_edt(~lining, return_indices=True)
        near_outside = ndimage.distance_transform_edt(~outside, return_indices=True)[1] if outside.any() else None
        lr, lc = np.nonzero(lining)
        cross_um, reach = cutoffs.WALL_CROSS_MIN_UM, cutoffs.DEPTH_LINING_REACH_UM / tm.tile_um
        out = np.zeros(tm.shape, bool)
        for r, c in zip(*np.nonzero(candidates)):
            nearest = (near_lining[0][[r], [c]], near_lining[1][[r], [c]])
            if wall_crossed_um(band, (r, c), nearest, tm.tile_um)[0] < cross_um:                       # 2
                continue
            nearby = np.flatnonzero(np.hypot(lr - r, lc - c) <= to_lining[r, c] + reach)
            nearby = nearby[::max(1, len(nearby) // 300)]                                  # at most ~300 lines
            behind = wall_crossed_um(band, (r, c), (lr[nearby], lc[nearby]), tm.tile_um) >= cross_um
            if behind.mean() < cutoffs.DEPTH_BEYOND_MAJORITY:                                          # 3
                continue
            if near_outside is not None:                                                                # 4
                end = (near_outside[0][[r], [c]], near_outside[1][[r], [c]])
                if wall_crossed_um(band, (r, c), end, tm.tile_um)[0] >= tm.tile_um:
                    continue
            out[r, c] = True
        return out

    return remembered(tm, key, compute)


def depth_evidence(tm, fr, params=None):
    """
    What the depth hint is decided from: the wall and fat layers, how many raw tumor tiles touch each, and the tumor
    tiles beyond the wall (beyond_wall). Uses the RAW tumor tiles (not the smoothed main tumor): invasion happens as thin
    strands of glands, which smoothing and the "largest piece" step erase (on a pT2 slide: 99 raw contacts -> 0).
    """
    wall, fat_layer = wall_layers(tm, params)
    radius = setting(params, "DEPTH_CONTACT_RADIUS_TILES")
    key = ("contact", setting(params, "WALL_MUSCLE_MIN_MM2"), setting(params, "WALL_FAT_MIN_MM2"),
           setting(params, "WALL_MIN_WIDTH_UM"), radius)

    def contact():
        raw_tumor = (tm.raw_label == CLASS_INDEX["tumor"]) & tm.tissue

        def touching(layer):
            if not layer.any():
                return np.zeros(tm.shape, bool)
            grown = ndimage.binary_dilation(layer, structure=np.ones((3, 3), bool), iterations=radius)
            return raw_tumor & grown & ~layer
        return int(touching(wall).sum()), int(touching(fat_layer).sum())   # counts only: keeps the cache small

    n_wall, n_fat = remembered(tm, key, contact)
    position = setting(params, "DEPTH_RULE") in ("position", "either")
    beyond = beyond_wall(tm, fr, params) if position else np.zeros(tm.shape, bool)
    return {"wall": wall, "fat_layer": fat_layer, "n_wall_contact": n_wall, "n_fat_contact": n_fat,
            "beyond": beyond, "n_beyond": int(beyond.sum())}


def depth_level(ev, params=None):
    """
    The decision, from depth_evidence. DEPTH_RULE picks what "consistent with at least pT3" means:
      "contact"   raw tumor tiles touch a fat layer (the v1/v2 rule)
      "position"  raw tumor tiles lie beyond the muscle wall (beyond_wall)
      "either"    one of the two
    Returns (deepest layer, level 1-3 or None, reason): reason = "beyond_wall" / "fat_contact" for level 3, else None.
    Depth is only judged when the section shows the muscle wall: without it, fat next to tumor may be fat inside the
    wall (submucosa) or outside it, and "doesn't reach the wall" means nothing (depth_and_fat still warns about the fat).
    """
    need, rule = setting(params, "DEPTH_CONTACT_MIN_TILES"), setting(params, "DEPTH_RULE")
    if not ev["wall"].any():
        return "not_assessable", None, None
    if rule in ("position", "either") and ev["n_beyond"] >= setting(params, "DEPTH_BEYOND_MIN_TILES"):
        return "fat", 3, "beyond_wall"
    if rule in ("contact", "either") and ev["n_fat_contact"] >= need:
        return "fat", 3, "fat_contact"
    if ev["n_wall_contact"] >= need:
        return "muscle", 2, None
    return "mucosa_submucosa", 1, None


def depth_and_fat(tm, fr, params=None):
    """
    Depth hint ("consistent with"), what the border touches, and how far tumor reaches into fat.
    Contact only counts against real layers (wall_layers). When the section shows no muscle wall, depth can't be
    judged; if tumor touches fat there, that is reported as a "look here" warning, not as a depth.
    """
    touch = {name: int(fr.contact[name].sum()) for name in ("muscle", "fat", "stroma", "normal_mucosa")}
    n_border = max(int((fr.border_type > 0).sum()), 1)
    n_front = max(int((fr.border_type == BORDER_FRONT).sum()), 1)

    ev = depth_evidence(tm, fr, params)
    wall, fat_layer = ev["wall"], ev["fat_layer"]
    need = setting(params, "DEPTH_CONTACT_MIN_TILES")
    n_fat, n_wall, n_beyond = ev["n_fat_contact"], ev["n_wall_contact"], ev["n_beyond"]
    deepest, level, reason = depth_level(ev, params)
    if reason == "beyond_wall":
        hint = ("Tumor lies beyond the muscle wall (on the side away from the lining): consistent with at least pT3 "
                "(check that this is the muscularis propria, not a fold of it).")
    elif reason == "fat_contact":
        hint = "Tumor touches fat: consistent with at least pT3 (check that this fat lies outside the muscle wall)."
    elif level == 2:
        hint = "Tumor reaches the muscle wall (muscularis propria): consistent with at least pT2."
    elif level == 1:
        hint = ("The muscle wall is in this section but tumor doesn't reach it: consistent with pTis/pT1 here "
                "(deeper invasion can sit in other sections).")
    elif n_fat >= need:
        hint = ("No muscle wall in this section, so depth can't be judged. Tumor does touch fat here: check it. Fat "
                "outside the bowel wall would mean at least pT3, but fat also sits inside the wall (submucosa).")
    else:
        hint = "No muscle wall in this section: depth can't be judged from this slide."

    depth_um, deep_grid = None, np.zeros(tm.shape, bool)
    if reason == "beyond_wall":
        past_um = ndimage.distance_transform_edt(~wall_band(wall)) * tm.tile_um
        depth_um = float(past_um[ev["beyond"]].max())
        deep_grid = ev["beyond"] & (past_um > cutoffs.FAT_DEPTH_FLAG_UM)
    elif reason == "fat_contact" and wall.any():
        raw_tumor = (tm.raw_label == CLASS_INDEX["tumor"]) & tm.tissue
        to_muscle = ndimage.distance_transform_edt(~wall) * tm.tile_um
        beyond = raw_tumor & dilate(fat_layer, 2)                     # tumor tiles out in the fat
        if beyond.any():
            depth_um = float(to_muscle[beyond].max())
            deep_grid = beyond & (to_muscle > cutoffs.FAT_DEPTH_FLAG_UM)
    fat_text = (f"Tumor touches fat in {n_fat} tiles (about {mm(n_fat * tm.tile_um)})." if n_fat >= need
                else "Tumor doesn't touch fat.")
    if reason == "beyond_wall":
        fat_text += f" {n_beyond} tumor tiles lie beyond the muscle wall."
    if depth_um is not None:
        past = "the muscle wall" if reason == "beyond_wall" else "the nearest muscle"
        fat_text += f" Tumor reaches about {mm(depth_um)} past {past}."
    invasion = {"border_touch_fraction": {name: rnd(n / n_border) for name, n in touch.items()},
                "deepest_layer": deepest, "depth_level": level, "t_stage_hint": hint,
                "depth_rule": setting(params, "DEPTH_RULE"), "depth_reason": reason,
                "wall_in_section": bool(wall.any()), "fat_in_section": bool(fat_layer.any()),
                "wall_contact_tiles": n_wall, "fat_contact_tiles": n_fat, "beyond_wall_tiles": n_beyond,
                "fat_contact_without_wall": bool(level is None and n_fat >= need)}
    fat = {"contact_mm": rnd(n_fat * tm.tile_um / 1000, 2) if n_fat >= need else 0.0,
           "share_of_front": rnd(touch["fat"] / n_front),
           "depth_past_muscle_um": rnd(depth_um, 0),
           "over_5mm": bool(depth_um is not None and depth_um > cutoffs.FAT_DEPTH_FLAG_UM), "text": fat_text}
    return invasion, fat, deep_grid


def sideways_spread(tm, fr):
    """Border against normal mucosa (spread within the lining) vs the invasive front."""
    mucosa_mm = (fr.border_type == BORDER_MUCOSA).sum() * tm.tile_um / 1000
    front_mm = (fr.border_type == BORDER_FRONT).sum() * tm.tile_um / 1000
    return {"mucosa_border_mm": rnd(mucosa_mm, 2), "front_mm": rnd(front_mm, 2),
            "text": f"Border against normal mucosa {mucosa_mm:.1f} mm vs invasive front {front_mm:.1f} mm."}


# --- stroma, necrosis, mucin ----------------------------------------------------------------

def tumor_stroma_ratio(tm, fr):
    """Most stroma-rich ~2 mm field along the front band: stroma / (stroma + tumor)."""
    kernel = disk(tm.tiles(cutoffs.TSR_WINDOW_DIAMETER_UM / 2)).astype(float)
    stroma = (tm.class_mask("stroma") & fr.in_section).astype(float)
    tumor = (fr.tumor & fr.in_section).astype(float)
    s_sum = ndimage.convolve(stroma, kernel, mode="constant")
    t_sum = ndimage.convolve(tumor, kernel, mode="constant")
    ok = (fr.zone == ZONE_FRONT) & (t_sum >= 1) & (s_sum + t_sum >= cutoffs.TSR_MIN_TILES)
    if not ok.any():
        return {"worst_field_stroma_fraction": None, "stroma_high": None, "text": "Not enough tumor and stroma at the front."}, None
    ratio = np.where(ok, s_sum / np.maximum(s_sum + t_sum, 1e-9), -1.0)
    r, c = np.unravel_index(ratio.argmax(), ratio.shape)
    worst = float(ratio[r, c])
    high = worst > cutoffs.TSR_STROMA_HIGH
    field = np.zeros(tm.shape, bool)
    field[r, c] = True
    field = ndimage.binary_dilation(field, structure=kernel.astype(bool))
    text = f"{'Stroma-high' if high else 'Stroma-low'}: {worst:.0%} stroma in the worst field (marked)."
    return {"worst_field_stroma_fraction": rnd(worst), "stroma_high": bool(high), **tm.tile_ref(r, c),
            "window_diameter_um": cutoffs.TSR_WINDOW_DIAMETER_UM, "text": text}, field


def necrosis_and_rim(tm, fr):
    """Necrosis inside the tumor (not ulceration at the surface), central or not, and the rim around it."""
    necrosis = tm.class_mask("necrosis") & fr.main_filled
    lab = measure.label(necrosis, connectivity=2)
    at_surface = np.unique(lab[neighbors_of(~tm.tissue, off_grid=True) & (lab > 0)])
    inside = necrosis & ~np.isin(lab, at_surface)
    area = max(int(fr.main_filled.sum()), 1)
    fraction = inside.sum() / area
    central = bool(inside.any() and (fr.zone[inside] == ZONE_CORE).mean() >= 0.5)

    rim = np.zeros(tm.shape, bool)
    if inside.any():
        rim_tiles = max(1, int(round(tm.tiles(cutoffs.NECROSIS_RIM_UM))))
        rim = fr.main & ~inside & (ndimage.distance_transform_edt(~inside) <= rim_tiles)
    p_lymph = tm.probs[..., CLASS_INDEX["lymphocytes"]]
    rest = fr.main & ~rim
    lymph_rim = float(p_lymph[rim].mean()) if rim.any() else None
    lymph_rest = float(p_lymph[rest].mean()) if rest.any() else None

    if not inside.any():
        text = "No necrosis inside the tumor."
    else:
        text = f"{'Central necrosis' if central else 'Necrosis (not central)'}, {fraction:.0%} of tumor area."
        text += f" Likely low-oxygen rim next to it: {rim.sum() / max(fr.main.sum(), 1):.0%} of tumor tiles."
    return {"fraction_in_tumor": rnd(fraction), "central": central,
            "rim_fraction_of_tumor": rnd(rim.sum() / max(fr.main.sum(), 1)),
            "lymphocytes_in_rim": rnd(lymph_rim), "lymphocytes_rest_of_tumor": rnd(lymph_rest),
            "text": text}, rim


def mucin(tm, fr):
    """
    Mucin in the tumor, and mucin pools out in the fat (past the muscle). "In the tumor" = inside the filled tumor
    outline PLUS mucin pools touching it (within cutoffs.MUCIN_ATTACH_TILES): in mucinous tumors the pools sit at the
    edge of the tumor glands, not only in fully enclosed holes (measured against TCGA's mucinous label, see
    reports/overnight_v4).
    """
    mucus = tm.class_mask("mucus")
    pools_lab = measure.label(mucus & fr.in_section, connectivity=2)
    near = dilate(fr.main_filled, cutoffs.MUCIN_ATTACH_TILES)
    attached = np.isin(pools_lab, np.unique(pools_lab[near & (pools_lab > 0)]))
    region = fr.main_filled | attached
    fraction = (mucus & region).sum() / max(int(region.sum()), 1)
    if fraction > cutoffs.MUCINOUS_FRACTION:
        label = (f"Mucinous: {fraction:.0%} of the tumor (with the mucin pools attached to it) is mucin (over the "
                 f"{cutoffs.MUCINOUS_FRACTION:.0%} cutoff for mucinous adenocarcinoma).")
    elif fraction > 0:
        label = f"{fraction:.0%} of the tumor (with the mucin pools attached to it) is mucin (mucinous features)."
    else:
        label = "No mucin in or next to the tumor."

    muscle = tm.class_mask("muscle") & fr.in_section
    to_muscle = ndimage.distance_transform_edt(~muscle) * tm.tile_um if muscle.any() else None
    pools, pool_grid = [], np.zeros(tm.shape, bool)
    lab = measure.label(mucus & ~fr.main_filled & fr.in_section, connectivity=2)
    for p in measure.regionprops(lab):
        m = lab == p.label
        if class_mix(tm, ring_of(m, tm))["fat"] < cutoffs.POOL_RING_FAT:
            continue
        r, c = center_tile(p.coords)
        past = float(to_muscle[m].min()) if to_muscle is not None else None
        where = f"{mm(past)} beyond the muscle" if past is not None else "no muscle in this section"
        pools.append({**tm.tile_ref(r, c), "area_mm2": rnd(p.area * tm.tile_mm2), "distance_past_muscle_um": rnd(past, 0),
                      "text": f"Mucin pool in fat, {where}: check for floating tumor cells."})
        pool_grid |= m
    return {"fraction_in_tumor": rnd(fraction), "label": label, "pools": pools}, pool_grid


# --- immune cells ---------------------------------------------------------------------------

def immune(tm, fr, node_mask):
    """Lymphocytes in the tumor core vs the front band vs outside, by side. Location only, not type."""
    if "p_immune" in tm.extra:   # lymphocyte detector (src/detectors.py): also counts lymphocytes scattered in stroma/tumor
        p_lymph, measure = tm.extra["p_immune"], "mean p_immune (lymphocyte detector) per tile in each zone (0-1)"
    else:
        p_lymph, measure = tm.probs[..., CLASS_INDEX["lymphocytes"]], "mean P(lymphocytes) per tile in each zone (0-1)"
    valid = tm.tissue & ~node_mask                    # lymph nodes aren't part of the immune reaction

    def mean_in(mask):
        m = mask & valid
        return float(p_lymph[m].mean()) if m.any() else None

    core = mean_in(fr.zone == ZONE_CORE)
    margin = mean_in(fr.zone == ZONE_FRONT)
    peri = mean_in(fr.zone == ZONE_PERI)

    centroid = np.array(ndimage.center_of_mass(fr.main_filled))
    rows, cols = np.nonzero((fr.zone == ZONE_FRONT) & valid)
    sides = {"top": [], "right": [], "bottom": [], "left": []}
    for r, c in zip(rows, cols):
        b = bearing(r - centroid[0], c - centroid[1])
        side = "top" if (b >= 315 or b < 45) else "right" if b < 135 else "bottom" if b < 225 else "left"
        sides[side].append(p_lymph[r, c])
    by_side = {side: rnd(np.mean(v)) if v else None for side, v in sides.items()}

    if margin is None:
        pattern = "unknown"
    elif max(core or 0.0, margin) < cutoffs.IMMUNE_LOW:
        pattern = "few lymphocytes (desert-like)"
    elif core is not None and core >= cutoffs.IMMUNE_CORE_TO_MARGIN * margin:
        pattern = "lymphocytes reach inside the tumor (inflamed-like)"
    else:
        pattern = "lymphocytes stay at the edge (excluded-like)"
    text = (f"Lymphocyte score: core {core or 0:.2f}, front {margin or 0:.2f}, outside {peri or 0:.2f}: {pattern}. "
            "H&E shows where lymphocytes sit, not their type (no CD8/CD4) or what they're doing.")
    return {"core_fraction": rnd(core), "margin_fraction": rnd(margin), "peritumoral_fraction": rnd(peri),
            "margin_by_side": by_side, "pattern": pattern, "pattern_cutoffs_calibrated": False,
            "measure": measure, "text": text}


# --- the outer surface ----------------------------------------------------------------------

def outer_edge(tm, fr, params=None):
    """
    Distance from tumor to the outer tissue edge (the side away from the lumen); flag if it touches.
    Stricter than "tumor reaches any tissue edge" (that fired on all 5 of our first TCGA slides, 4 of them pT1-pT3):
    only judged in sections that show the muscle wall or fat, and the tumor must lie on the outer edge along at least
    SEROSA_MIN_RUN_UM.
    """
    caveat = ("The outer edge may be a cut surface rather than serosa, and the rectum below the "
              "peritoneal fold has no serosa: check before calling pT4a.")
    wall, fat_layer = wall_layers(tm, params)
    if setting(params, "SEROSA_NEEDS_WALL") and not (wall.any() or fat_layer.any()):
        return {"found": False, "distance_um": None, "touches": False,
                "text": "This section doesn't show the muscle wall or fat, so the serosa can't be judged.",
                "caveat": caveat}, np.zeros(tm.shape, bool)
    region = (fr.in_section | fr.main_filled) & tm.tissue
    boundary = region & neighbors_of(~tm.tissue, off_grid=True)
    kernel = disk(2).astype(float)
    wall_like = np.logical_or.reduce([tm.class_mask(n) for n in ("fat", "muscle", "stroma")]) & ~fr.tumor
    inner = np.logical_or.reduce([tm.class_mask(n) for n in ("normal_mucosa", "mucus", "necrosis", "lymphocytes")])
    outer = boundary & (ndimage.convolve(wall_like.astype(float), kernel, mode="constant")
                        > ndimage.convolve(inner.astype(float), kernel, mode="constant"))
    layers = wall | fat_layer
    if layers.any():   # the serosa is outside the muscle wall: ignore edges far from the wall / fat (the luminal side)
        outer &= ndimage.distance_transform_edt(~layers) * tm.tile_um <= setting(params, "SEROSA_MAX_FROM_WALL_UM")
    if not outer.any():
        return {"found": False, "distance_um": None, "touches": False,
                "text": "No outer tissue edge found in this section.", "caveat": caveat}, np.zeros(tm.shape, bool)
    distance = float((ndimage.distance_transform_edt(~outer) * tm.tile_um)[fr.main].min())
    on_edge = fr.main & outer
    run_um = float(on_edge.sum() * tm.tile_um)
    touches = bool(run_um >= setting(params, "SEROSA_MIN_RUN_UM"))
    text = (f"Possible serosal involvement: check (pT4a). Tumor lies on the outer tissue edge along about {mm(run_um)}."
            if touches else f"Tumor within {mm(distance)} of the outer tissue edge.")
    return {"found": True, "distance_um": round(distance), "touches": touches, "run_on_edge_um": round(run_um),
            "text": text, "caveat": caveat}, (on_edge if touches else np.zeros(tm.shape, bool))
