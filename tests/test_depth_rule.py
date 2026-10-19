"""
Tests for the depth hint's two rules (src/features.depth_level): "contact" (tumor touches fat) and "position"
(tumor beyond the muscle wall). Each test builds a small strip of bowel wall by hand, top to bottom:
glass (lumen), mucosa, submucosa (stroma), muscle wall, fat, glass. Runs in about a second:

    pytest -q tests/test_depth_rule.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import depth_and_fat  # noqa: E402
from src.front import find_front  # noqa: E402
from src.prompts import CLASSES  # noqa: E402
from src.tilemap import TileMap  # noqa: E402

K = {name: i for i, name in enumerate(CLASSES)}
GLASS = -1


def strip(cols=60):
    """Layers of a normal colon wall; one tile = 112 µm, so the muscle wall here is ~1.1 mm thick."""
    label = np.full((50, cols), GLASS)
    label[2:6] = K["normal_mucosa"]
    label[6:12] = K["stroma"]          # submucosa
    label[12:22] = K["muscle"]         # muscularis propria
    label[22:46] = K["fat"]            # fat outside the wall
    label[:, :1] = label[:, -1:] = GLASS
    return label


def tile_map(label):
    """A TileMap where every tile's top class is `label` (90% sure)."""
    rows, cols = np.nonzero(label != GLASS)
    probs = np.full((len(rows), len(CLASSES)), 0.1 / (len(CLASSES) - 1))
    probs[np.arange(len(rows)), label[rows, cols]] = 0.9
    df = pd.DataFrame({"row": rows, "col": cols, "tile_size_px": 224, "qc_ok": True})
    for i, name in enumerate(CLASSES):
        df[f"p_{name}"] = probs[:, i]
    return TileMap.from_dataframe(df, mpp=0.5, shape=label.shape)


def depth(label, **params):
    tm = tile_map(label)
    return depth_and_fat(tm, find_front(tm), params)[0]


def with_tumor_beyond_the_wall():
    """Tumor in the mucosa, plus nests out in the fat, each wrapped in stroma (desmoplasia): no tile touches fat."""
    label = strip()
    label[2:9, 20:40] = K["tumor"]
    for c in (14, 26, 38):
        label[27:34, c:c + 7] = K["stroma"]
        label[28:33, c + 1:c + 6] = K["tumor"]
    return label


def test_position_rule_finds_tumor_wrapped_in_stroma():
    label = with_tumor_beyond_the_wall()
    contact = depth(label, DEPTH_RULE="contact")
    position = depth(label, DEPTH_RULE="position", DEPTH_BEYOND_MIN_TILES=30)
    assert contact["fat_contact_tiles"] == 0 and contact["depth_level"] == 1     # the old rule misses it
    assert position["depth_level"] == 3 and position["depth_reason"] == "beyond_wall"
    assert position["beyond_wall_tiles"] == 75                                    # all 3 nests, 25 tiles each
    assert "beyond the muscle wall" in position["t_stage_hint"] and "consistent with" in position["t_stage_hint"]
    assert depth(label, DEPTH_RULE="either")["depth_level"] == 3


def test_beyond_needs_enough_tiles_and_distance():
    label = with_tumor_beyond_the_wall()
    assert depth(label, DEPTH_RULE="position", DEPTH_BEYOND_MIN_TILES=100)["depth_level"] == 1
    # the nests start 6 tiles (~670 µm) past the wall: asking for 1 mm past it leaves only the deeper rows
    far = depth(label, DEPTH_RULE="position", DEPTH_BEYOND_MIN_UM=1000, DEPTH_BEYOND_MIN_TILES=1)
    assert 0 < far["beyond_wall_tiles"] < 75


def test_fat_inside_the_wall_is_not_beyond_it():
    """Tumor next to fat in the submucosa (inside the wall): the contact rule over-calls pT3, the position rule doesn't."""
    label = strip()
    label[6:12, 30:55] = K["fat"]                   # submucosal fat, 150 tiles = 1.9 mm²
    label[2:10, 15:30] = K["tumor"]                 # tumor in mucosa + submucosa, touching that fat, not the wall
    assert depth(label, DEPTH_RULE="contact")["depth_level"] == 3
    position = depth(label, DEPTH_RULE="position", DEPTH_BEYOND_MIN_TILES=1)
    assert position["beyond_wall_tiles"] == 0 and position["depth_level"] == 1


def test_tumor_on_the_lining_side_is_not_beyond():
    """A big tumor filling mucosa and submucosa down to the wall: it reaches the wall (pT2 hint), nothing beyond."""
    label = strip()
    label[2:13, 10:50] = K["tumor"]
    position = depth(label, DEPTH_RULE="position", DEPTH_BEYOND_MIN_TILES=1)
    assert position["beyond_wall_tiles"] == 0 and position["depth_level"] == 2


def test_no_wall_means_no_depth_but_fat_contact_is_still_flagged():
    """Fat but no muscle wall: depth can't be judged (the fat may be inside the wall), but tumor on fat gets a warning."""
    label = strip()
    label[12:22] = K["stroma"]                      # no muscle wall in this section
    label[2:26, 20:40] = K["tumor"]                 # tumor down into the fat
    for rule in ("contact", "position", "either"):
        invasion = depth(label, DEPTH_RULE=rule)
        assert invasion["depth_level"] is None and invasion["deepest_layer"] == "not_assessable"
        assert invasion["fat_contact_without_wall"] and "can't be judged" in invasion["t_stage_hint"]
    label[14:26, 20:40] = K["stroma"]               # tumor stays in the mucosa: no warning
    invasion = depth(label)
    assert invasion["depth_level"] is None and not invasion["fat_contact_without_wall"]


def test_thin_muscle_is_not_the_wall():
    """A one-tile line of muscle (like the muscularis mucosae) only counts as wall when the width check is off."""
    label = np.full((30, 200), GLASS)
    label[4:10, 1:-1] = K["normal_mucosa"]
    label[10, 1:-1] = K["muscle"]                   # 198 tiles = 2.5 mm², but one tile thick
    label[11:20, 1:-1] = K["stroma"]
    label[4:10, 80:120] = K["tumor"]                # tumor in the mucosa, sitting on the thin muscle
    assert depth(label, WALL_MIN_WIDTH_UM=0)["depth_level"] == 2
    thick_only = depth(label, WALL_MIN_WIDTH_UM=300)
    assert not thick_only["wall_in_section"] and thick_only["depth_level"] is None
