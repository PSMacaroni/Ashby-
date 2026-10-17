"""
Finds processed slides and reads their result files, plus the small "slide box" file where the
app remembers renames, copies and hidden slides. Result files are only ever read, never changed.

A slide = one folder holding summary.json + tiles.csv (+ thumbnail.png, optional tiles/ previews),
either in app/demo_data/ (fake) or data/results/ (real, copied down from Delta).
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from common import BOX_STATE_FILE, CLASSES, DEMO_DIR, RESULTS_DIR

PROB_COLS = [f"p_{c}" for c in CLASSES]


@dataclass
class Slide:
    key: str          # unique id in the slide box: the folder name, or "<folder>~copy2" for a copy
    folder: Path
    slide_id: str
    name: str         # shown on the card (can be renamed)
    case_id: str
    entered: str      # date the results were made (YYYY-MM-DD)
    is_demo: bool
    is_copy: bool
    hidden: bool
    flags: list
    top_risk_level: int
    has_tumor: bool


def stamp(folder):
    """Changes whenever a result file changes, so cached data gets reloaded."""
    return max((p.stat().st_mtime for p in Path(folder).glob("*") if p.is_file()), default=0)


# ---------------------------------------------------------------- reading result files

@st.cache_data(show_spinner=False)
def _read_summary(folder_str, _stamp):
    path = Path(folder_str) / "summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_summary(folder):
    return _read_summary(str(folder), stamp(folder))


@st.cache_data(show_spinner=False)
def _read_tiles(folder_str, _stamp):
    df = pd.read_csv(Path(folder_str) / "tiles.csv")
    for col in PROB_COLS:
        if col not in df:
            df[col] = np.nan
    df["qc_ok"] = df["qc_ok"].astype(bool) if "qc_ok" in df else True
    df["qc_reason"] = df["qc_reason"].fillna("").astype(str) if "qc_reason" in df else ""
    probs = df[PROB_COLS].to_numpy(dtype=float)
    if "label" not in df:
        df["label"] = [CLASSES[i] for i in np.nan_to_num(probs, nan=-1).argmax(1)]
    if "confidence" not in df:
        df["confidence"] = np.nanmax(probs, axis=1)
    for col, default in (("risk_level", 0), ("risk_points", 0)):
        df[col] = df[col].fillna(default).astype(int) if col in df else default
    df["risk_flags"] = df["risk_flags"].fillna("").astype(str) if "risk_flags" in df else ""
    return df.reset_index(drop=True)


def read_tiles(folder):
    return _read_tiles(str(folder), stamp(folder))


def has_model_columns(tiles, model):
    """True if tiles.csv has this model's own probabilities (p_conch_tumor, ...)."""
    return all(f"p_{model}_{c}" in tiles for c in CLASSES)


@st.cache_data(show_spinner=False)
def _read_thumbnail(folder_str, _stamp):
    path = Path(folder_str) / "thumbnail.png"
    return Image.open(path).convert("RGB") if path.exists() else None


def read_thumbnail(folder):
    return _read_thumbnail(str(folder), stamp(folder))


@st.cache_data(show_spinner=False)
def _card_picture(folder_str, _stamp, size=(360, 220)):
    """The slide thumbnail fitted into a fixed-size box, so all cards line up."""
    box = Image.new("RGB", size, (236, 236, 240))
    thumb = _read_thumbnail(folder_str, _stamp)
    if thumb is None:
        return box
    t = thumb.copy()
    t.thumbnail((size[0] - 10, size[1] - 10))
    box.paste(t, ((size[0] - t.width) // 2, (size[1] - t.height) // 2))
    return box


def card_picture(folder):
    return _card_picture(str(folder), stamp(folder))


def tile_picture(slide, tile):
    """
    (image, caption) for one tile: the real preview from Delta if there is one, a fake one for
    demo slides, otherwise (None, reason).
    """
    path = slide.folder / "tiles" / f"{int(tile['row'])}_{int(tile['col'])}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB"), "Tile preview from the slide"
    if slide.is_demo:
        from make_demo_data import fake_preview
        seed = int(tile["row"]) * 1000 + int(tile["col"])
        return fake_preview(tile["label"], seed), "Fake picture (DEMO slide)"
    return None, "No preview saved for this tile (ask Delta for tiles/<row>_<col>.jpg)."


# ---------------------------------------------------------------- the slide box

def _folders():
    """Every result folder, real results first."""
    out = []
    for base in (RESULTS_DIR, DEMO_DIR):
        if base.is_dir():
            out += sorted(p for p in base.iterdir()
                          if (p / "summary.json").exists() and (p / "tiles.csv").exists())
    return out


def load_box():
    try:
        state = json.loads(BOX_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    for k in ("names", "hidden", "copies"):
        state.setdefault(k, {} if k != "hidden" else [])
    return state


def save_box(state):
    tmp = BOX_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(BOX_STATE_FILE)


def case_id_from(slide_id):
    """TCGA barcodes: the first 12 characters are the (public, de-identified) case ID."""
    return slide_id[:12] if slide_id.startswith("TCGA-") else "—"


def _make_slide(key, folder, state, is_copy, copy_info=None):
    s = read_summary(folder)
    slide_id = s.get("slide_id") or folder.name
    default_name = (copy_info or {}).get("name") or s.get("display_name") or slide_id
    entered = s.get("processed_at") or datetime.fromtimestamp(
        (folder / "summary.json").stat().st_mtime).strftime("%Y-%m-%d")
    regions = s.get("risk_regions") or []
    tumor = (s.get("tumor") or {}).get("fraction_of_tissue") or 0
    return Slide(
        key=key, folder=folder, slide_id=slide_id,
        name=state["names"].get(key, default_name),
        case_id=s.get("case_id") or case_id_from(slide_id),
        entered=str(entered)[:10],
        is_demo=bool(s.get("is_demo")), is_copy=is_copy,
        hidden=key in state["hidden"],
        flags=list(s.get("high_risk_flags") or []),
        top_risk_level=max((int(r.get("level", 0)) for r in regions), default=0),
        has_tumor=tumor > 0,
    )


def list_slides(include_hidden=False):
    state = load_box()
    folders = {f.name: f for f in _folders()}
    slides = [_make_slide(name, f, state, False) for name, f in folders.items()]
    for key, info in state["copies"].items():
        if info.get("source") in folders:
            slides.append(_make_slide(key, folders[info["source"]], state, True, info))
    return [s for s in slides if include_hidden or not s.hidden]


def get_slide(key):
    return next((s for s in list_slides(include_hidden=True) if s.key == key), None)


def rename(key, name):
    name = name.strip()
    if not name:
        return
    state = load_box()
    state["names"][key] = name
    save_box(state)


def duplicate(key):
    """Add a copy of a slide to the box (same result files, its own name)."""
    original = get_slide(key)
    if original is None:
        return
    state = load_box()
    source = original.folder.name
    n = 1
    while f"{source}~copy{n}" in state["copies"]:
        n += 1
    state["copies"][f"{source}~copy{n}"] = {"source": source, "name": f"{original.name} (copy)",
                                            "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
    save_box(state)


def remove(key):
    """Copies are deleted; original slides are only hidden (their files stay on disk)."""
    state = load_box()
    if key in state["copies"]:
        del state["copies"][key]
        state["names"].pop(key, None)
    elif key not in state["hidden"]:
        state["hidden"].append(key)
    save_box(state)


def restore(key):
    state = load_box()
    state["hidden"] = [k for k in state["hidden"] if k != key]
    save_box(state)
