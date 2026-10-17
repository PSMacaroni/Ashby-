"""
The risk map: every feature adds points to the tiles within 1 mm of where it is found, and regions
with enough points get a card the pathologist can click through ("Level 4: possible venous invasion,
infiltrative arc, faces fat").

The points table: the "doc" rows come from "What else the tumor recruits" (additions to the risk map).
The features doc's own points table (its section 7) wasn't in the PDF we have, so the rows marked
PLACEHOLDER are our stand-ins. Replace them with section 7's rows and points.
"""
import numpy as np
from scipy import ndimage
from skimage import measure

from src import cutoffs
from src.front import disk
from src.features import center_tile

RISK_POINTS = [
    # key, points, bin, text on the card, source
    {"key": "venous_candidate", "points": 2, "bin": "flag",
     "text": "possible venous invasion (consider an elastin stain)", "source": "doc"},
    {"key": "lymphatic_candidate", "points": 2, "bin": "flag",
     "text": "possible lymphatic invasion (D2-40 can confirm)", "source": "doc"},
    {"key": "perineural_candidate", "points": 2, "bin": "flag", "text": "possible perineural invasion", "source": "doc"},
    {"key": "serosal_involvement", "points": 2, "bin": "flag", "text": "possible serosal involvement (pT4a?)", "source": "doc"},
    {"key": "tumor_in_fat_over_5mm", "points": 1, "bin": "measure", "text": "tumor more than 5 mm past the muscle",
     "source": "doc"},
    {"key": "mucin_pool_past_muscle", "points": 1, "bin": "measure", "text": "mucin pool past the muscle", "source": "doc"},
    {"key": "necrosis_rim", "points": 1, "bin": "measure", "text": "next to necrosis (likely low oxygen)", "source": "doc"},
    {"key": "infiltrative_arc", "points": 1, "bin": "measure", "text": "infiltrative arc", "source": "PLACEHOLDER"},
    {"key": "satellite", "points": 1, "bin": "measure", "text": "satellite tumor island", "source": "PLACEHOLDER"},
    {"key": "faces_fat", "points": 1, "bin": "measure", "text": "front faces fat", "source": "PLACEHOLDER"},
    {"key": "budding_hotspot", "points": 1, "bin": "flag", "text": "budding hotspot (count buds here)",
     "source": "PLACEHOLDER"},
    {"key": "stroma_high_field", "points": 1, "bin": "measure", "text": "stroma-high field", "source": "PLACEHOLDER"},
]


def risk_map(tm, fr, where):
    """
    where: {feature key: [R, C] bool grid of where that feature is}.
    Returns (points grid [R, C] int, list of region cards, the points table used).
    """
    near = disk(tm.tiles(cutoffs.RISK_NEIGHBORHOOD_UM))
    region = (fr.in_section | fr.main_filled) & tm.tissue
    points = np.zeros(tm.shape, int)
    reach = {}
    for row in RISK_POINTS:
        grid = where.get(row["key"])
        if grid is None or not grid.any():
            continue
        reach[row["key"]] = ndimage.binary_dilation(grid, structure=near) & region
        points += row["points"] * reach[row["key"]]

    cards = []
    lab = measure.label(points >= cutoffs.RISK_CARD_MIN_POINTS, connectivity=2)
    for p in measure.regionprops(lab):
        m = lab == p.label
        level = int(points[m].max())
        top = np.argwhere(m & (points == level))
        r, c = center_tile(top)
        features = [row["text"] for row in RISK_POINTS if row["key"] in reach and (reach[row["key"]] & m).any()]
        cards.append({**tm.tile_ref(r, c), "level": level, "n_tiles": int(p.area), "features": features,
                      "text": f"Level {level}: " + ", ".join(features) + "."})
    cards.sort(key=lambda d: (-d["level"], -d["n_tiles"]))
    return points, cards
