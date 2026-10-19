"""Checks for src/features.mucin: mucin pools touching the tumor count as part of it.  pytest -q"""
import numpy as np

from src.features import mucin
from src.front import find_front
from src.prompts import CLASSES
from src.tilemap import CLASS_INDEX, TileMap


def grid_map(layout):
    """layout: [R, C] array of class names ("" = no tissue), each tile 100% that class."""
    R, C = layout.shape
    probs = np.zeros((R, C, len(CLASSES)))
    tissue = layout != ""
    for (r, c), name in np.ndenumerate(layout):
        if name:
            probs[r, c, CLASS_INDEX[name]] = 1.0
    return TileMap(probs=probs, tissue=tissue, tile_um=112.0, x=np.zeros((R, C)), y=np.zeros((R, C)))


def test_mucin_pool_at_the_tumor_edge_counts():
    layout = np.full((30, 40), "stroma", dtype=object)
    layout[8:22, 6:20] = "tumor"
    layout[8:22, 20:30] = "mucus"            # a pool on the tumor's side, not enclosed by it
    tm = grid_map(layout)
    out, _ = mucin(tm, find_front(tm))
    assert 0.3 < out["fraction_in_tumor"] < 0.5      # about 10 / (14 + 10) of the width
    assert "mucin pools attached" in out["label"]
    assert out["check_mucinous"] and "check for mucinous features" in out["label"]


def test_mucin_far_from_the_tumor_does_not_count():
    layout = np.full((30, 50), "stroma", dtype=object)
    layout[8:22, 6:20] = "tumor"
    layout[8:22, 35:45] = "mucus"            # 15 tiles away
    tm = grid_map(layout)
    out, _ = mucin(tm, find_front(tm))
    assert out["fraction_in_tumor"] == 0
    assert not out["check_mucinous"]
