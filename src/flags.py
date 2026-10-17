"""
Section 4 of the plan: "look here" flags = a model part (text-phrase scores, or a trained detector)
plus a geometry part (touching tumor, in the front band, a round island in fat...).

Flags point the pathologist to up to TOP_K tiles to check. They are never a count, a grade or a
diagnosis, and every flag says how much to trust it.

Text phrases always have a "top 5", even on a slide with nothing there. So each candidate is marked
"strong" only if its score stands out from the rest of the slide (PHRASE_STRONG_Z standard deviations),
or if it comes from a trained detector plus geometry. Only strong candidates feed the risk map and the
high-risk list; weak ones are still shown as hints.

Inputs besides the TileMap and Front:
  scores    {phrase key: [R, C] grid} from phrase_scores.csv (src/phrases.py); may be None
  tm.extra  "p_nerve", "p_artery": optional trained yes/no detectors (Models teammate). When a detector
            is missing, the flag falls back to text phrases, which are less reliable.
"""
import math

import numpy as np
from scipy import ndimage
from skimage import measure

from src import cutoffs
from src.front import BORDER_MUCOSA, ZONE_FRONT, ZONE_PERI, dilate, disk

DETECTOR = "trained detector"
PHRASE_ONLY = "text phrase only (less reliable)"


def make_flag(key, title, action, trust, source=None, candidates=(), measure=None, note=None, skipped=None):
    candidates = list(candidates)
    return {"key": key, "title": title, "status": "skipped" if skipped else "ok", "why_skipped": skipped,
            "source": source, "trust": trust, "action": action, "note": note, "candidates": candidates,
            "n_strong": sum(c.get("strong", False) for c in candidates), "measure": measure}


def robust_z(value, reference):
    """
    How far `value` stands out from `reference`, in standard deviations. Uses the median and the
    median absolute deviation, so the hotspots themselves don't drag the yardstick up.
    """
    reference = np.asarray(reference, float)
    if reference.size == 0:
        return 0.0
    median = float(np.median(reference))
    spread = 1.4826 * float(np.median(np.abs(reference - median)))
    return (float(value) - median) / spread if spread > 0 else 0.0


def zscore(tm, score, value):
    """How far `value` stands out from this score over the slide's tissue tiles."""
    return robust_z(value, score[tm.tissue])


def top_candidates(tm, score, mask, why, k=cutoffs.TOP_K, strong=None):
    """
    The k tiles inside `mask` with the highest score, as tile references.
    strong=None: judge each one by its z-score against the slide; True: all strong (detector-based).
    """
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return []
    values = score.ravel()[idx]
    out = []
    for i in np.argsort(values)[::-1][:k]:
        r, c = np.unravel_index(idx[i], tm.shape)
        z = zscore(tm, score, values[i])
        is_strong = strong if strong is not None else z >= cutoffs.PHRASE_STRONG_Z
        out.append({**tm.tile_ref(r, c), "score": round(float(values[i]), 4), "z": round(z, 1),
                    "strong": bool(is_strong), "why": why})
    return out


def strong_grid(tm, candidates):
    """Grid of the strong candidates (what the risk map counts)."""
    grid = np.zeros(tm.shape, bool)
    for cand in candidates:
        if cand.get("strong"):
            grid[cand["row"], cand["col"]] = True
    return grid


def has(scores, *keys):
    return scores is not None and all(k in scores for k in keys)


def structure_mask(tm, scores, detector_key, phrase_key):
    """Where nerves/arteries are: the trained detector if we have it, else the top phrase-scoring tiles."""
    if detector_key in tm.extra:
        strength = tm.extra[detector_key]
        return tm.tissue & (strength >= cutoffs.STRUCTURE_PROB), strength, DETECTOR
    if has(scores, phrase_key):
        strength = scores[phrase_key]
        values = strength[tm.tissue]
        if values.size == 0:
            return None, None, None
        threshold = np.quantile(values, 1 - cutoffs.PHRASE_TOP_FRACTION)
        return tm.tissue & (strength >= threshold) & (strength > 0), strength, PHRASE_ONLY
    return None, None, None


# --- the flags ------------------------------------------------------------------------------

def perineural(tm, fr, scores):
    """Nerve tiles touching tumor. With the trained nerve detector, also measures how far out it goes."""
    title, action = "Possible perineural invasion", "Look for tumor cells wrapping around or inside this nerve."
    nerve, strength, source = structure_mask(tm, scores, "p_nerve", "nerve")
    if nerve is None:
        return make_flag("perineural", title, action, "experimental",
                         skipped="needs the nerve detector (p_nerve) or phrase scores"), None
    touching = nerve & dilate(fr.tumor) & ~fr.tumor
    rank = scores["pni"] if has(scores, "pni") else strength
    candidates = top_candidates(tm, rank, touching, "nerve next to tumor",
                                strong=True if source == DETECTOR else None)
    measure = None
    if source == DETECTOR and touching.any():
        beyond = fr.signed_um[touching]
        farthest = max(float(np.nanmax(beyond)) if np.isfinite(beyond).any() else 0.0, 0.0)
        measure = {"nerve_tiles_touching_tumor": int(touching.sum()), "farthest_beyond_front_um": round(farthest),
                   "text": f"Tumor reaches nerves up to {farthest / 1000:.1f} mm beyond the front."}
    flag = make_flag("perineural", title, action, "experimental", source, candidates, measure)
    return flag, strong_grid(tm, candidates)


def venous(tm, fr, pieces, scores):
    """Round tumor islands in fat next to an artery ("orphan artery" sign), plus vein phrase matches."""
    title, action = "Possible venous invasion", "Consider an elastin stain on this block."
    artery, _, source = structure_mask(tm, scores, "p_artery", "artery")
    if artery is None and not has(scores, "vein"):
        return make_flag("venous", title, action, "experimental",
                         skipped="needs the artery detector (p_artery) or phrase scores"), None
    lo, hi = cutoffs.VEIN_ISLAND_DIAMETER_UM
    islands = [p for p in pieces
               if not p["in_lymph_node"] and lo <= p["diameter_um"] <= hi
               and p["solidity"] >= cutoffs.VEIN_ISLAND_MIN_SOLIDITY
               and p["fat_fraction_around"] >= cutoffs.VEIN_ISLAND_RING_FAT]

    candidates, orphans = [], []
    if artery is not None and artery.any() and islands:
        near_artery = ndimage.distance_transform_edt(~artery) * tm.tile_um <= cutoffs.ARTERY_NEAR_UM
        for p in islands:
            m = fr.pieces == p["id"]
            if not (m & near_artery).any():
                continue
            if has(scores, "vein"):
                best = float(scores["vein"][m].max())
                z = zscore(tm, scores["vein"], best)
            else:
                best, z = float(p["solidity"]), None
            candidates.append({**{k: p[k] for k in ("row", "col", "x", "y")}, "score": round(best, 4),
                               "z": None if z is None else round(z, 1),
                               "strong": source == DETECTOR or (z is not None and z >= cutoffs.PHRASE_STRONG_Z),
                               "why": f"round tumor island ({p['diameter_um']} µm) in fat next to an artery"})
        # orphan arteries: arteries in fat with a round tumor island within 0.5 mm
        to_island = ndimage.distance_transform_edt(~np.isin(fr.pieces, [p["id"] for p in islands])) * tm.tile_um
        lab = measure.label(artery & dilate(tm.class_mask("fat")), connectivity=2)
        for a in measure.regionprops(lab):
            if to_island[lab == a.label].min() <= cutoffs.ARTERY_NEAR_UM:
                r, c = a.coords[0]
                orphans.append({**tm.tile_ref(r, c), "why": "artery with a round tumor island beside it"})
    candidates.sort(key=lambda d: (not d["strong"], -d["score"]))
    if has(scores, "vein"):
        out_in_fat = ~fr.main_filled & dilate(tm.class_mask("fat")) & fr.in_section
        for cand in top_candidates(tm, scores["vein"], out_in_fat, "looks like tumor in a vein (text match)"):
            if all(abs(cand["row"] - d["row"]) > 1 or abs(cand["col"] - d["col"]) > 1 for d in candidates):
                candidates.append(cand)
    candidates = candidates[:cutoffs.TOP_K]
    flag = make_flag("venous", title, action, "experimental", f"shape rule + {source or PHRASE_ONLY}",
                     candidates, measure={"orphan_arteries": orphans})
    return flag, strong_grid(tm, candidates)


def lymphatic(tm, fr, scores):
    """Tumor in thin-walled spaces near the front, by text match only. The weakest flag."""
    title = "Possible lymphatic invasion"
    action = "A D2-40 stain tells lymphatic invasion apart from retraction artifact."
    note = "Expect many false alarms: on H&E, retraction artifact looks almost the same."
    if not has(scores, "lymphatic"):
        return make_flag("lymphatic", title, action, "weak", skipped="needs phrase scores", note=note), None
    near_front = ((fr.zone == ZONE_FRONT) | (fr.zone == ZONE_PERI)) & tm.tissue & (scores["lymphatic"] > 0)
    candidates = top_candidates(tm, scores["lymphatic"], near_front, "tumor in a thin-walled space? (text match)")
    flag = make_flag("lymphatic", title, action, "weak", PHRASE_ONLY, candidates, note=note)
    return flag, strong_grid(tm, candidates)


def budding_hotspot(tm, fr, scores):
    """The 0.785 mm² field along the front with the highest budding-phrase score. Never a grade."""
    title = "Budding hotspot: count buds here"
    action = "Count tumor buds in this field (ITBCC). The app does not grade budding."
    if not has(scores, "budding"):
        return make_flag("budding", title, action, "experimental", skipped="needs phrase scores"), None
    band = (fr.zone == ZONE_FRONT) & tm.tissue
    if not band.any():
        return make_flag("budding", title, action, "experimental", skipped="no front band"), None
    radius = math.sqrt(cutoffs.BUDDING_FIELD_MM2 / tm.tile_mm2 / math.pi)
    kernel = disk(radius).astype(float)
    total = ndimage.convolve(np.where(band, scores["budding"], 0.0), kernel, mode="constant")
    count = ndimage.convolve(band.astype(float), kernel, mode="constant")
    valid = band & (count > 0)
    mean = np.where(valid, total / np.maximum(count, 1e-9), -np.inf)
    r, c = np.unravel_index(mean.argmax(), mean.shape)
    z = robust_z(mean[r, c], mean[valid])
    field = np.zeros(tm.shape, bool)
    field[r, c] = True
    field = ndimage.binary_dilation(field, structure=kernel.astype(bool)) & tm.tissue
    candidate = {**tm.tile_ref(r, c), "score": round(float(mean[r, c]), 4), "z": round(z, 1),
                 "strong": z >= cutoffs.PHRASE_STRONG_Z,
                 "radius_um": round(radius * tm.tile_um), "field_mm2": round(kernel.sum() * tm.tile_mm2, 3),
                 "why": "highest average budding-phrase score in a budding-sized field along the front"}
    flag = make_flag("budding", title, action, "experimental", PHRASE_ONLY, [candidate])
    return flag, (field if candidate["strong"] else None)


def stroma_maturity(tm, fr, scores):
    """Which way the front-band stroma leans: mature, keloid-like, or immature (Ueno). Experimental."""
    keys = ("stroma_mature", "stroma_keloid", "stroma_immature")
    title = "Stroma at the front (experimental)"
    action = "Hand-check about 20 of these tiles with the pathologist before showing this to anyone."
    if not has(scores, *keys):
        return make_flag("stroma_maturity", title, action, "experimental", skipped="needs phrase scores")
    m = tm.class_mask("stroma") & (fr.zone == ZONE_FRONT)
    if not m.any():
        return make_flag("stroma_maturity", title, action, "experimental", skipped="no stroma at the front")
    winner = np.stack([scores[k][m] for k in keys], axis=-1).argmax(axis=1)
    shares = {name: round(float((winner == i).mean()), 3)
              for i, name in enumerate(("mature", "keloid_like", "immature"))}
    text = ("Front stroma leans: " + ", ".join(f"{k.replace('_', '-')} {v:.0%}" for k, v in shares.items())
            + ". Immature stroma is the worst sign (Ueno).")
    candidates = top_candidates(tm, scores["stroma_immature"] - scores["stroma_mature"], m,
                                "most immature-looking stroma at the front (text match)")
    return make_flag("stroma_maturity", title, action, "experimental", PHRASE_ONLY, candidates,
                     measure={"shares": shares, "text": text})


def leftover_adenoma(tm, fr, scores):
    title, action = "Possible leftover adenoma", "Check for adenoma next to the cancer."
    if not has(scores, "adenoma"):
        return make_flag("adenoma", title, action, "experimental", skipped="needs phrase scores")
    m = fr.border_type == BORDER_MUCOSA
    candidates = top_candidates(tm, scores["adenoma"], m, "tumor next to normal mucosa that looks like adenoma")
    return make_flag("adenoma", title, action, "experimental", PHRASE_ONLY, candidates)


def all_flags(tm, fr, pieces, scores):
    """Run every flag. Returns (list of flag dicts, {risk feature key: grid of the strong candidates})."""
    pni, pni_grid = perineural(tm, fr, scores)
    vein, vein_grid = venous(tm, fr, pieces, scores)
    lymph, lymph_grid = lymphatic(tm, fr, scores)
    bud, bud_grid = budding_hotspot(tm, fr, scores)
    flags = [pni, vein, lymph, bud, stroma_maturity(tm, fr, scores), leftover_adenoma(tm, fr, scores)]
    grids = {"perineural_candidate": pni_grid, "venous_candidate": vein_grid,
             "lymphatic_candidate": lymph_grid, "budding_hotspot": bud_grid}
    return flags, {k: v for k, v in grids.items() if v is not None}
