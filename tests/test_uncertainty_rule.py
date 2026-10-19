"""Checks for the "unsure here" rule in src/features.uncertainty: low confidence only; a second model's label
(alt_label) doesn't count.  pytest -q"""
import numpy as np

from src import cutoffs
from src.features import uncertainty
from src.prompts import CLASSES
from src.tilemap import CLASS_INDEX, TileMap


def tile_map(tiles):
    """One row of tiles: tiles = [(CONCH label, CONCH top probability, MUSK label), ...]."""
    n = len(tiles)
    probs = np.zeros((1, n, len(CLASSES)))
    for i, (label, top, _) in enumerate(tiles):
        probs[0, i, :] = (1 - top) / (len(CLASSES) - 1)
        probs[0, i, CLASS_INDEX[label]] = top
    alt = np.array([[CLASS_INDEX[m] for _, _, m in tiles]])
    return TileMap(probs=probs, tissue=np.ones((1, n), bool), tile_um=112.0, x=np.zeros((1, n)), y=np.zeros((1, n)),
                   extra={"alt_label": alt})


def unsure_tiles(tiles):
    return uncertainty(tile_map(tiles))[1][0].tolist()


HIGH = min(cutoffs.LOW_CONFIDENCE + 0.05, 0.99)


def test_disagreement_alone_is_not_unsure():
    assert unsure_tiles([("tumor", HIGH, "necrosis"), ("muscle", HIGH, "stroma"), ("tumor", HIGH, "normal_mucosa")]) \
        == [False, False, False]


def test_low_confidence_is_unsure():
    low = cutoffs.LOW_CONFIDENCE - 0.1
    assert unsure_tiles([("stroma", low, "stroma"), ("tumor", HIGH, "tumor")]) == [True, False]
