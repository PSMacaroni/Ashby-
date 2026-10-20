"""
Search inside the open slide.

Two ways, depending on what the slide has:
  - live text search: the query becomes a CONCH text embedding, compared with every tile's saved
    image embedding (embeddings.npy, same row order as tiles.csv). Needs CONCH installed on this
    laptop and a slide with embeddings. (Not testable yet: no slide has embeddings.npy so far.)
  - known-words search (always works): the query's words are matched to the 8 tissue classes, the
    nerve/artery detectors (p_nerve, p_artery) and the flag phrases the analysis already scored
    (phrase_scores.csv). Less flexible, but it says exactly what it matched.

Either way: each tile's score is averaged with its 3x3 neighbors (regions, not single noisy tiles),
words like "border" or "core" limit it to that zone, and the top spots are returned spaced apart.
"""
import re

import numpy as np

from common import CLASSES, NAMES

CLASS_WORDS = {
    "tumor": ["tumor", "tumour", "cancer", "carcinoma", "adenocarcinoma", "malignant"],
    "normal_mucosa": ["normal", "mucosa", "glands", "benign"],
    "muscle": ["muscle", "muscularis"],
    "fat": ["fat", "adipose", "adipocytes", "pericolic"],
    "lymphocytes": ["lymphocytes", "lymphocyte", "immune", "inflammation", "inflammatory", "tils", "lymphoid"],
    "mucus": ["mucus", "mucin", "mucinous"],
    "necrosis": ["necrosis", "necrotic", "dead", "debris"],
    "stroma": ["stroma", "stromal", "desmoplastic", "desmoplasia", "fibrous", "connective", "fibroblasts"],
}
PHRASE_WORDS = {   # keys of src/phrases.py FLAG_PHRASES (the columns of phrase_scores.csv)
    "budding": ["budding", "buds", "bud"],
    "lymphatic": ["lymphatic", "lymphovascular", "lvi"],
    "vein": ["vein", "venous", "veins", "emvi"],
    "artery": ["artery", "arteries", "arterial"],
    "nerve": ["nerve", "nerves"],
    "pni": ["perineural", "pni"],
    "stroma_mature": ["mature"],
    "stroma_keloid": ["keloid", "hyalinized"],
    "stroma_immature": ["immature", "myxoid"],
    "adenoma": ["adenoma", "dysplasia", "polyp"],
    "pen": ["pen", "ink"],
    "fold": ["fold", "folds"],
    "blur": ["blur", "blurry", "focus"],
}
PHRASE_NAMES = {"budding": "tumor budding", "lymphatic": "lymphatic invasion", "vein": "tumor in a vein",
                "artery": "artery", "nerve": "nerve", "pni": "tumor around a nerve", "stroma_mature": "mature stroma",
                "stroma_keloid": "keloid-like collagen", "stroma_immature": "immature (myxoid) stroma",
                "adenoma": "leftover adenoma", "pen": "pen ink", "fold": "tissue fold", "blur": "blur"}
DETECTORS = {"nerve": "p_nerve", "artery": "p_artery"}     # trained detectors beat text phrases
ZONE_WORDS = {
    "front_band": ["border", "front", "edge", "margin", "periphery", "boundary"],
    "core": ["core", "center", "centre", "inside", "interior"],
    "peritumoral": ["around", "surrounding", "peritumoral", "outside"],
}
ZONE_NAMES = {"front_band": "invasive front band", "core": "tumor core", "peritumoral": "around the tumor"}
# Things smaller than a tile: be honest that the map can't find them
SINGLE_CELL = {"mitosis", "mitoses", "mitotic", "cd8", "cd4", "foxp3", "treg", "tregs", "macrophage", "macrophages",
               "neutrophil", "neutrophils", "eosinophil", "eosinophils", "nucleus", "nuclei", "signet"}
MIN_SCORE = 0.15          # weaker spots aren't shown as results
SPACING = 5               # tiles between results, so they point at different places


def words(query):
    return set(re.findall(r"[a-z]+", query.lower()))


def understand(query):
    """Which classes, phrases and zone the words point to."""
    w = words(query)
    classes = [c for c, ws in CLASS_WORDS.items() if w & set(ws)]
    phrases = [k for k, ws in PHRASE_WORDS.items() if w & set(ws)]
    zone = next((z for z, ws in ZONE_WORDS.items() if w & set(ws)), None)
    if "stroma" in classes and any(p.startswith("stroma_") for p in phrases):
        classes.remove("stroma")          # "immature stroma": use the stroma-maturity phrase only
    if "tumor" in classes and (phrases or (zone and len(classes) > 1)):
        classes.remove("tumor")           # "tumor in a vein", "lymphocytes at the tumor border":
                                          # the phrase or the place already says it's about tumor
    return classes, phrases, zone


def _phrase_table(slide):
    path = slide.folder / "phrase_scores.csv"
    if not path.exists():
        return None
    import pandas as pd
    return pd.read_csv(path)


def known_words_scores(slide, data, classes, phrases):
    """Per-tile 0-1 score from the matched classes/phrases, and what was used (for the UI)."""
    cols, used = [], []
    for c in classes:
        if c == "lymphocytes" and "p_immune" in data.extra:     # the lymphocyte detector sees scattered cells too
            cols.append(np.nan_to_num(np.clip(data.extra["p_immune"], 0, 1)))
            used.append("Lymphocytes (immune detector)")
            continue
        cols.append(data.probs[:, CLASSES.index(c)])
        used.append(f"{NAMES[c]} (tissue score)")
    table = None
    for p in phrases:
        det = DETECTORS.get(p)
        if det and det in data.extra:
            cols.append(np.clip(data.extra[det], 0, 1))
            used.append(f"{PHRASE_NAMES[p]} (trained detector)")
            continue
        if table is None:
            table = _phrase_table(slide)
        if table is None or p not in table:
            continue
        raw = (table.set_index(["row", "col"])[p].reindex(list(zip(data.rows, data.cols))).to_numpy(float))
        raw = np.nan_to_num(raw, nan=np.nanmedian(raw))
        z = (raw - np.median(raw)) / (1.4826 * np.median(np.abs(raw - np.median(raw))) + 1e-9)
        cols.append(np.clip(z / 6, 0, 1))         # "stands out" scale: 6 robust SDs = full score
        used.append(f"{PHRASE_NAMES[p]} (text match)")
    if not cols:
        return None, used
    return np.mean(cols, axis=0), used


def live_available(slide):
    """True when this slide has CONCH embeddings and CONCH can be loaded here."""
    import importlib.util
    return (slide.folder / "embeddings.npy").exists() and importlib.util.find_spec("conch") is not None


_model = {}


def live_scores(slide, data, query):
    """Cosine similarity of every tile to the query (CONCH), rescaled to 0-1. Untested until real
    embeddings exist; any failure falls back to known-words search."""
    if "conch" not in _model:
        from src.models import load_conch
        _model["conch"] = load_conch(device="cpu")[0]
    from src.models import encode_texts
    emb = np.load(slide.folder / "embeddings.npy").astype(np.float32)
    q = encode_texts(_model["conch"], [query]).float().cpu().numpy()[0]
    sim = emb @ q
    z = (sim - np.median(sim)) / (sim.std() + 1e-9)
    return np.clip(z / 4, 0, 1)


def smooth(data, values):
    """Average each tile with its 3x3 neighbors (tissue only)."""
    from scipy import ndimage
    g = data.geom
    grid = np.zeros((g["rows"], g["cols"]))
    has = np.zeros_like(grid)
    grid[data.rows, data.cols] = values * data.ok
    has[data.rows, data.cols] = data.ok
    num = ndimage.uniform_filter(grid, 3, mode="constant")
    den = ndimage.uniform_filter(has, 3, mode="constant")
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0)[data.rows, data.cols]


def search(slide, data, query, top=5):
    """
    Returns {"query", "how", "used", "zone", "values" (0-1 per tile, for the map), "spots"
    [{"r", "c", "score", "label"}], "message"}.
    """
    query = query.strip()
    out = {"query": query, "how": "", "used": [], "zone": None, "values": None, "spots": [], "message": ""}
    if not query:
        return out
    classes, phrases, zone = understand(query)
    if words(query) & SINGLE_CELL:
        out["message"] = ("That's a single-cell feature. The map gives one answer per 112 µm square "
                          "(10-15 cells), so it can't find it; check at high power. ")
    values = None
    if live_available(slide):
        try:
            values = live_scores(slide, data, query)
            out["how"], out["used"] = "live", [f"CONCH text match for “{query}”"]
        except Exception as e:
            out["message"] = f"Live search failed ({type(e).__name__}); used known words instead. "
    if values is None:
        values, out["used"] = known_words_scores(slide, data, classes, phrases)
        out["how"] = "known words"
    if values is None:
        known = sorted({w for ws in list(CLASS_WORDS.values()) + list(PHRASE_WORDS.values()) for w in ws})
        out["message"] += ("No known words found. Try a tissue (tumor, stroma, lymphocytes, necrosis, fat, "
                           "muscle, mucin) or a feature (budding, nerve, vein, lymphatic, adenoma), plus "
                           "'border' or 'core'. Free-text search turns on when slides have embeddings.npy "
                           "and CONCH is installed.")
        out["known_words"] = known
        return out

    values = smooth(data, values)
    if zone:
        values = values * (data.zone == zone)
        out["zone"] = zone
    out["values"] = values / max(values.max(), 1e-9)

    g = data.geom
    grid = np.full((g["rows"], g["cols"]), -1.0)
    grid[data.rows, data.cols] = values
    for n in range(1, top + 1):
        r, c = np.unravel_index(int(np.argmax(grid)), grid.shape)
        if grid[r, c] < MIN_SCORE:
            break
        out["spots"].append({"r": int(r), "c": int(c), "score": float(grid[r, c]), "label": f"S{n}"})
        grid[max(0, r - SPACING):r + SPACING + 1, max(0, c - SPACING):c + SPACING + 1] = -1
    if not out["spots"]:
        out["message"] += "Nothing stands out for this search on this slide."
    return out
