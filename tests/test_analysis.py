"""
Tests for the slide analysis, on fake slides with known contents (src/fake_slide.py).
No GPU, no model, runs in a few seconds:

    pytest -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis import analyze  # noqa: E402
from src.fake_slide import make_fake_slide  # noqa: E402
from src.front import BORDER_FRONT, BORDER_MUCOSA, BORDER_SURFACE, ZONE_CORE, find_front  # noqa: E402
from src.tilemap import TileMap  # noqa: E402


def run(variant, with_scores=True, with_detectors=True):
    tiles, scores_df, meta, truth = make_fake_slide(variant)
    if not with_detectors:
        tiles = tiles.drop(columns=["p_nerve", "p_artery"])
    tm = TileMap.from_dataframe(tiles, mpp=0.5, shape=tuple(meta["grid_shape"]))
    scores = None
    if with_scores:
        rows, cols = scores_df["row"].to_numpy(), scores_df["col"].to_numpy()
        scores = {}
        for key in scores_df.columns.drop(["row", "col"]):
            grid = np.zeros(tm.shape)
            grid[rows, cols] = scores_df[key].to_numpy()
            scores[key] = grid
    summary, table = analyze(tm, scores, meta)
    return tm, summary, table, truth


@pytest.fixture(scope="module")
def infiltrative():
    return run("infiltrative")


@pytest.fixture(scope="module")
def pushing():
    return run("pushing")


def flag(summary, key):
    return next(f for f in summary["flags"] if f["key"] == key)


# --- the tile map and step 0 ------------------------------------------------------------------

def test_tile_map_units(infiltrative):
    tm, summary, _, _ = infiltrative
    assert tm.tile_um == pytest.approx(112.0)
    assert summary["n_tissue_tiles"] == int(tm.tissue.sum())


def test_front_is_not_the_luminal_surface(infiltrative):
    tm, _, _, truth = infiltrative
    fr = find_front(tm)
    top_border = fr.border_type[6]                                 # tumor touching the lumen (glass)
    assert (top_border[top_border > 0] == BORDER_SURFACE).all()
    assert (fr.border_type == BORDER_MUCOSA).any()                # tumor touching normal mucosa
    assert (fr.contact["fat"] & (fr.border_type == BORDER_FRONT)).any()
    assert (fr.zone == ZONE_CORE).any()


def test_tumor_found(infiltrative):
    _, s, _, _ = infiltrative
    assert s["tumor"]["found"]
    assert 0.1 < s["tumor"]["fraction_of_tissue"] < 0.4


# --- growth pattern, direction, depth ---------------------------------------------------------

def test_growth_pattern(infiltrative, pushing):
    inf, push = infiltrative[1]["border"], pushing[1]["border"]
    assert inf["pattern"] == "infiltrative"
    assert push["pattern"] == "pushing"
    assert inf["solidity"] < push["solidity"]
    assert inf["n_fingers"] >= 2 and push["n_fingers"] == 0
    # 3 islands were planted 224, 242 and 569 µm from the tumor; pieces closer than cutoffs.TUMOR_JOIN_UM (500 µm)
    # are part of the main tumor, so 1 or 2 remain satellites (smoothing moves the edges by a tile)
    assert 1 <= inf["n_satellites_within_1mm"] <= 2 and push["n_satellites_within_1mm"] == 0


def test_direction_toward_lower_left(infiltrative):
    direction = infiltrative[1]["invasion"]["direction"]
    assert direction["clock"] in (6, 7, 8)       # fingers and satellites were planted lower left / below
    assert abs(sum(direction["rose_16"]) - 1) < 0.01         # bins are rounded to 3 decimals


def test_depth_hint(infiltrative):
    invasion = infiltrative[1]["invasion"]
    assert invasion["deepest_layer"] == "fat"
    assert "pT3" in invasion["t_stage_hint"] and "consistent with" in invasion["t_stage_hint"]
    assert infiltrative[1]["fat"]["contact_mm"] > 0


# --- necrosis, mucin, immune, nodes, deposits, edges ------------------------------------------

def test_central_necrosis(infiltrative):
    necrosis = infiltrative[1]["necrosis"]
    assert necrosis["central"] and necrosis["fraction_in_tumor"] > 0
    assert necrosis["rim_fraction_of_tumor"] > 0


def test_mucin_pool_in_fat(infiltrative):
    pools = infiltrative[1]["mucin"]["pools"]
    assert len(pools) == 1 and pools[0]["distance_past_muscle_um"] > 0


def test_immune_pattern(infiltrative):
    assert "edge" in infiltrative[1]["immune"]["pattern"]      # lymphocytes planted at the edge only


def test_lymph_node_metastasis(infiltrative):
    nodes = infiltrative[1]["lymph_nodes"]
    assert len(nodes) == 1 and nodes[0]["possible_metastasis"]


def test_deposits(infiltrative, pushing):
    assert any(p["possible_deposit"] for p in infiltrative[1]["satellites"])
    assert not any(p["possible_deposit"] for p in pushing[1]["satellites"])
    assert any(p["in_lymph_node"] for p in pushing[1]["satellites"])   # the tumor inside the node


def test_outer_edge(infiltrative):
    edge = infiltrative[1]["outer_edge"]
    assert edge["found"] and not edge["touches"] and edge["distance_um"] > 1000


def test_uncertainty(infiltrative):
    unsure = infiltrative[1]["uncertainty"]
    # low confidence only: the fake slide's CONCH/MUSK disagreement area no longer counts
    assert unsure["fraction_unsure"] == unsure["fraction_low_confidence"]
    assert all(r["why"] == "low confidence" for r in unsure["regions"])


# --- flags and risk ---------------------------------------------------------------------------

def test_flags_find_planted_spots(infiltrative):
    s = infiltrative[1]
    pni = flag(s, "perineural")
    assert pni["n_strong"] >= 1 and any(c["row"] == 32 and c["col"] in (56, 57) for c in pni["candidates"])
    vein = flag(s, "venous")
    assert any(c["strong"] and 44 <= c["row"] <= 48 and 62 <= c["col"] <= 66 for c in vein["candidates"])
    assert vein["measure"]["orphan_arteries"]
    assert flag(s, "budding")["n_strong"] == 1


def test_no_strong_flags_without_signal(pushing):
    s = pushing[1]
    assert flag(s, "venous")["n_strong"] == 0      # nothing planted: the top 5 are noise, all weak
    assert flag(s, "budding")["n_strong"] == 0
    assert not any("venous" in line for line in s["high_risk_flags"])


def test_risk_cards(infiltrative):
    risk = infiltrative[1]["risk"]
    assert risk["max_level"] >= 3 and risk["cards"]
    assert risk["cards"][0]["text"].startswith(f"Level {risk['cards'][0]['level']}:")


def test_missing_inputs_are_skipped_not_crashing():
    _, s, _, _ = run("infiltrative", with_scores=False, with_detectors=False)
    assert all(f["status"] == "skipped" for f in s["flags"])
    assert any("Not computed" in c for c in s["caveats"])


def test_normal_slide_has_no_tumor():
    _, s, table, _ = run("normal")
    assert not s["tumor"]["found"]
    assert s["high_risk_flags"] == []
    assert len(table) == s["n_tissue_tiles"]


def test_outputs_are_plain_json(infiltrative):
    _, s, table, _ = infiltrative
    json.dumps(s)                                   # no numpy types left
    assert {"row", "col", "zone", "border", "risk_points", "unsure"} <= set(table.columns)


def test_short_front_is_not_judged():
    """A tumor-only fragment: the 'front' is a few tiles, so pattern and direction say they can't be judged."""
    from src.analysis import analyze
    from src.prompts import CLASSES
    from src.tilemap import CLASS_INDEX, TileMap
    R, C = 30, 30
    probs = np.zeros((R, C, len(CLASSES)))
    probs[..., CLASS_INDEX["tumor"]] = 1.0
    tissue = np.zeros((R, C), bool)
    tissue[5:25, 5:25] = True                   # all tumor, surrounded by glass
    probs[22:25, 12:15] = 0
    probs[22:25, 12:15, CLASS_INDEX["stroma"]] = 1.0   # a small stroma spot on the edge
    tm = TileMap(probs=probs, tissue=tissue, tile_um=112.0, x=np.zeros((R, C)), y=np.zeros((R, C)))
    s, _ = analyze(tm)
    assert s["border"]["pattern"] == "not_assessable"
    assert s["invasion"]["direction"]["clock"] is None
