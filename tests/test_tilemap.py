"""Checks for the necrosis -> muscle overrule in src/tilemap.py (CONCH says necrosis, MUSK says muscle).  pytest -q"""
import numpy as np
import pandas as pd

from src import cutoffs
from src.prompts import CLASSES
from src.tilemap import CLASS_INDEX, TileMap


def block(alt):
    """A 3 x 8 strip of tiles that CONCH calls necrosis (0.7) over muscle (0.2); left half / right half alt labels."""
    rows = []
    for r in range(3):
        for c in range(8):
            p = {f"p_{k}": 0.0 for k in CLASSES}
            p.update(p_necrosis=0.7, p_muscle=0.2, p_stroma=0.1)
            rows.append({"row": r, "col": c, "qc_ok": True, "tile_size_px": 224, **p,
                         "alt_label": alt[0] if c < 4 else alt[1]})
    return pd.DataFrame(rows)


def test_necrosis_with_musk_muscle_becomes_muscle_for_the_analysis_only(monkeypatch):
    monkeypatch.setattr(cutoffs, "ALT_OVERRULES", (("necrosis", "muscle"),))   # off by default, see below
    df = block(("muscle", "necrosis"))
    tm = TileMap.from_dataframe(df)
    nec, mus = CLASS_INDEX["necrosis"], CLASS_INDEX["muscle"]
    assert tm.overruled[:, :4].all() and not tm.overruled[:, 4:].any()
    assert (tm.raw_label == nec).all()                       # what the model said is kept
    assert (tm.label[:, :2] == mus).all()                    # away from the boundary: muscle
    assert (tm.label[:, 6:] == nec).all()                    # MUSK agreed: still necrosis
    assert np.allclose(tm.probs[..., nec], 0.7)              # raw probabilities untouched
    assert (df.p_necrosis == 0.7).all()                      # tiles.csv data untouched


def test_no_alt_label_means_no_overrule():
    tm = TileMap.from_dataframe(block(("muscle", "muscle")).drop(columns="alt_label"))
    assert not tm.overruled.any()
    assert (tm.label == CLASS_INDEX["necrosis"]).all()


def test_overrule_is_off_by_default():
    # team decision: MUSK is used only for the blood check (cutoffs.ALT_OVERRULES = ())
    tm = TileMap.from_dataframe(block(("muscle", "muscle")))
    assert not tm.overruled.any()
    assert (tm.label == CLASS_INDEX["necrosis"]).all()
