"""
Finds processed slides and reads their result files (never changes them), plus the small state
file where the app remembers renames, hidden slides, copies, notes and reviewed items.

A slide = one folder with summary.json + tiles.csv in showcase/results/ (the 5 showcase slides,
in git) or data/results/ (copied from Delta, git-ignored). The data contract is in CLAUDE.md.
Optional files the app uses: tile_analysis.csv (zones, risk points, unsure), thumbnail.png,
analysis_overview.png, tiles/<row>_<col>.jpg previews, phrase_scores.csv, embeddings.npy.
If a slide has tiles.csv but no analysis yet, the app runs the team's analysis (src/analysis.py)
in memory, so new slides from Delta show up complete.

Speed: the slide list only reads summary.json. Everything a slide screen needs is loaded once
into memory (SlideData) and reused on every click; warm_up() loads all slides in the background.
"""
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from common import CLASSES, LOW_CONFIDENCE, NAMES, RESULTS_DIRS, STATE_FILE, get

ANALYSIS_KEYS = ("tumor", "border", "invasion", "risk", "uncertainty")
# optional per-tile scores from the side models (tiles.csv or structures.csv): nerve / vessel /
# artery detectors and the immune ("lymphocytes present?") detector
EXTRA_COLUMNS = ("p_nerve", "p_vessel", "p_artery", "p_immune", "p_blood")


_stamps = {}


def stamp(folder, max_age_s=2.0):
    """Changes whenever a result file changes, so cached data gets reloaded. Checking the files is
    slow on some disks, so the answer is reused for a couple of seconds."""
    now = time.monotonic()
    hit = _stamps.get(str(folder))
    if hit and now - hit[0] < max_age_s:
        return hit[1]
    value = max((p.stat().st_mtime for p in Path(folder).glob("*") if p.is_file()), default=0)
    _stamps[str(folder)] = (now, value)
    return value


def read_summary(folder):
    try:
        return json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------- the state file

_state_lock = threading.Lock()


def load_state():
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    for key, default in (("names", {}), ("hidden", []), ("copies", {}), ("notes", {}), ("reviewed", {})):
        state.setdefault(key, default)
    return state


def save_state(state):
    with _state_lock:
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(STATE_FILE)


def update_state(fn):
    """Read-modify-write the state file in one go. fn(state) changes it in place."""
    state = load_state()
    fn(state)
    save_state(state)


# ---------------------------------------------------------------- the slide list

@dataclass
class Slide:
    key: str          # unique id in the list: the folder name, or "<folder>~copy2" for a copy
    folder: Path
    slide_id: str
    name: str
    case_id: str
    entered: str
    is_copy: bool
    hidden: bool
    flags: list
    tumor_found: bool
    tumor_fraction: float
    risk_max: int
    stage: str        # pT/pN/stage from the TCGA report if the pipeline recorded it
    site: str
    pattern: str
    depth: str
    immune: str
    unsure: float
    n_tiles: int


def _folders():
    out = {}
    for base in RESULTS_DIRS:
        if base.is_dir():
            for p in sorted(base.iterdir()):
                if (p / "summary.json").exists() and (p / "tiles.csv").exists() and p.name not in out:
                    out[p.name] = p
    return out


def case_id_from(slide_id):
    """TCGA barcodes: the first 12 characters are the (public, de-identified) case ID."""
    return slide_id[:12] if slide_id.startswith("TCGA-") else "—"


def short_depth(s):
    """'≥ pT3 (fat)' / '≥ pT2 (muscle)' / 'can't judge' from the invasion section."""
    level = get(s, "invasion.depth_level")
    layer = get(s, "invasion.deepest_layer")
    if not get(s, "tumor.found", True):
        return "no tumor"
    if level is None and not layer:
        return "not judged"
    if layer == "fat":
        return "≥ pT3 (touches fat)"
    if layer == "muscle":
        return "≥ pT2 (reaches muscle)"
    if layer:
        return f"{layer} only"
    return "no muscle / fat contact"


def _make_slide(key, folder, state, is_copy, copy_info=None):
    from common import site_name, stage_text
    s = read_summary(folder)
    slide_id = s.get("slide_id") or folder.name
    entered = s.get("processed_at") or datetime.fromtimestamp(
        (folder / "summary.json").stat().st_mtime).strftime("%Y-%m-%d")
    pattern = get(s, "border.pattern") or ("no tumor" if not get(s, "tumor.found", True) else "not judged")
    from report import short_immune
    immune = short_immune(get(s, "immune.pattern"))
    return Slide(
        key=key, folder=folder, slide_id=slide_id,
        name=state["names"].get(key, (copy_info or {}).get("name") or s.get("display_name") or slide_id),
        case_id=s.get("case_id") or case_id_from(slide_id), entered=str(entered)[:10],
        is_copy=is_copy, hidden=key in state["hidden"],
        flags=list(s.get("high_risk_flags") or []),
        tumor_found=bool(get(s, "tumor.found", (get(s, "tumor.fraction_of_tissue", 0) or 0) > 0)),
        tumor_fraction=float(get(s, "tumor.fraction_of_tissue", 0) or 0),
        risk_max=int(get(s, "risk.max_level", 0) or 0), stage=stage_text(s), site=site_name(s),
        pattern=pattern, depth=short_depth(s), immune=immune,
        unsure=float(get(s, "uncertainty.fraction_unsure", 0) or 0),
        n_tiles=int(s.get("n_tissue_tiles") or 0),
    )


def list_slides(include_hidden=False):
    state = load_state()
    folders = _folders()
    slides = [_make_slide(name, f, state, False) for name, f in folders.items()]
    for key, info in state["copies"].items():
        if info.get("source") in folders:
            slides.append(_make_slide(key, folders[info["source"]], state, True, info))
    return [s for s in slides if include_hidden or not s.hidden]


def get_slide(key):
    return next((s for s in list_slides(include_hidden=True) if s.key == key), None)


def rename(key, name):
    name = name.strip()
    if name:
        update_state(lambda st: st["names"].__setitem__(key, name))


def duplicate(key):
    """Add a copy of a slide to the list (same result files, its own name and notes)."""
    original = get_slide(key)
    if original is None:
        return
    state = load_state()
    source, n = original.folder.name, 1
    while f"{source}~copy{n}" in state["copies"]:
        n += 1
    state["copies"][f"{source}~copy{n}"] = {"source": source, "name": f"{original.name} (copy)",
                                            "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
    save_state(state)


def remove(key):
    """Copies are deleted; original slides are only hidden (their files stay on disk)."""
    state = load_state()
    if key in state["copies"]:
        del state["copies"][key]
        state["names"].pop(key, None)
    elif key not in state["hidden"]:
        state["hidden"].append(key)
    save_state(state)


def restore(key):
    update_state(lambda st: st.__setitem__("hidden", [k for k in st["hidden"] if k != key]))


def notes(key):
    return load_state()["notes"].get(key, "")


def set_notes(key, text):
    update_state(lambda st: st["notes"].__setitem__(key, text))


def reviewed(key):
    """Set of worklist item ids marked as reviewed for this slide."""
    return set(load_state()["reviewed"].get(key, []))


def set_reviewed(key, item_id, done):
    def change(st):
        items = set(st["reviewed"].get(key, []))
        (items.add if done else items.discard)(item_id)
        st["reviewed"][key] = sorted(items)
    update_state(change)


# ---------------------------------------------------------------- pictures

_thumbs = {}


def thumbnail(folder):
    path = Path(folder) / "thumbnail.png"
    key = (str(path), path.stat().st_mtime if path.exists() else 0)
    if key not in _thumbs:
        _thumbs[key] = Image.open(path).convert("RGB") if path.exists() else None
    return _thumbs[key]


def small_thumbnail(folder, size=(220, 150)):
    """A small copy of the thumbnail on a white frame (for the slide list)."""
    key = ("small", str(folder), stamp(folder), size)
    if key not in _thumbs:
        box = Image.new("RGB", size, (255, 255, 255))
        thumb = thumbnail(folder)
        if thumb is not None:
            t = thumb.copy()
            t.thumbnail((size[0] - 8, size[1] - 8))
            box.paste(t, ((size[0] - t.width) // 2, (size[1] - t.height) // 2))
        _thumbs[key] = box
    return _thumbs[key]


# ---------------------------------------------------------------- one slide, fully loaded

@dataclass
class SlideData:
    summary: dict
    tiles: object            # pandas DataFrame, tiles.csv
    rows: np.ndarray
    cols: np.ndarray
    ok: np.ndarray           # passed quality checks
    probs: np.ndarray        # [n, 8]
    label: np.ndarray        # [n] class index the model gave (raw)
    conf: np.ndarray
    alt: np.ndarray          # [n] second model's class index, -1 if none
    unsure: np.ndarray
    certainty: np.ndarray    # "Sure" / "Unsure" / "Not analyzed"
    risk_points: np.ndarray
    zone: np.ndarray         # "core" / "front_band" / "peritumoral" / "outside" / ""
    on_front: np.ndarray
    dist_um: np.ndarray
    in_main_tumor: np.ndarray
    share: np.ndarray        # share of the 8 neighbors with the same label
    index_grid: np.ndarray   # [R, C] -> tile index, -1 for no tile
    geom: dict
    has_alt: bool
    extra: dict = field(default_factory=dict)   # p_nerve, p_artery, p_immune, p_blood when present
    analyzed_here: bool = False                 # analysis was run by the app (no tile_analysis.csv)


_loaded = {}
# One lock for loading slides AND the heavy libraries: two threads importing scipy at the same time
# breaks it ("partially initialized module"), e.g. the background warm-up and a user opening a slide.
_lock = threading.RLock()


def load(slide):
    """Everything the slide screen needs, loaded once per version of the files."""
    key = (str(slide.folder), stamp(slide.folder))
    with _lock:
        if key not in _loaded:
            _loaded[key] = _load(Path(slide.folder))
        return _loaded[key]


def _load(folder):
    import pandas as pd
    from scipy import ndimage

    summary = read_summary(folder)
    tiles = pd.read_csv(folder / "tiles.csv")
    tiles["qc_ok"] = tiles["qc_ok"].astype(bool) if "qc_ok" in tiles else True
    tiles["qc_reason"] = tiles["qc_reason"].fillna("").astype(str) if "qc_reason" in tiles else ""
    n = len(tiles)
    rows, cols = tiles["row"].to_numpy(int), tiles["col"].to_numpy(int)
    probs = tiles[[f"p_{c}" for c in CLASSES]].to_numpy(float)
    label = probs.argmax(1)
    conf = tiles["confidence"].to_numpy(float) if "confidence" in tiles else probs.max(1)
    ok = tiles["qc_ok"].to_numpy(bool)
    index = {c: i for i, c in enumerate(CLASSES)}
    has_alt = "alt_label" in tiles
    alt = np.array([index.get(v, -1) for v in tiles["alt_label"]]) if has_alt else np.full(n, -1)

    analysis, analyzed_here = None, False
    if (folder / "tile_analysis.csv").exists() and all(k in summary for k in ANALYSIS_KEYS):
        analysis = pd.read_csv(folder / "tile_analysis.csv")
    else:
        summary, analysis = _run_analysis(folder, tiles, summary)
        analyzed_here = True
    a = analysis.set_index(["row", "col"]).reindex(list(zip(rows, cols)))

    def column(name, default, kind):
        if name not in a:
            return np.full(n, default) if kind is not bool else np.zeros(n, bool)
        col = a[name]
        if kind is bool:
            return col.fillna(False).astype(bool).to_numpy()
        if kind is int:
            return col.fillna(0).astype(int).to_numpy()
        if kind is str:
            return col.fillna("").astype(str).to_numpy()
        return col.to_numpy(float)

    unsure = column("unsure", False, bool)
    risk_points = column("risk_points", 0, int)
    zone = column("zone", "", str)
    on_front = column("on_front", False, bool)
    dist_um = column("dist_to_front_um", np.nan, float)
    in_main = column("in_main_tumor", False, bool)
    certainty = np.where(~ok, "Not analyzed", np.where(unsure | (conf < LOW_CONFIDENCE), "Unsure", "Sure"))

    geom = geometry(tiles, summary, thumbnail(folder))
    R, C = geom["rows"], geom["cols"]
    index_grid = np.full((R, C), -1)
    index_grid[rows, cols] = np.arange(n)
    label_grid = np.full((R, C), -1)
    label_grid[rows[ok], cols[ok]] = label[ok]
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    n_nb = ndimage.convolve((label_grid >= 0).astype(float), kernel, mode="constant")
    same = np.zeros((R, C))
    for k in range(len(CLASSES)):
        m = label_grid == k
        same[m] = ndimage.convolve(m.astype(float), kernel, mode="constant")[m]
    share = np.where(n_nb > 0, same / np.maximum(n_nb, 1), 0)[rows, cols]

    extra = {c: tiles[c].to_numpy(float) for c in EXTRA_COLUMNS if c in tiles}
    if (folder / "structures.csv").exists():      # the side classifier (nerve / vessel), CLAUDE.md
        st_df = pd.read_csv(folder / "structures.csv").set_index(["row", "col"]).reindex(list(zip(rows, cols)))
        for c in EXTRA_COLUMNS:
            if c in st_df and c not in extra:
                extra[c] = st_df[c].to_numpy(float)
    return SlideData(summary=summary, tiles=tiles, rows=rows, cols=cols, ok=ok, probs=probs, label=label,
                     conf=conf, alt=alt, unsure=unsure, certainty=certainty, risk_points=risk_points, zone=zone,
                     on_front=on_front, dist_um=dist_um, in_main_tumor=in_main, share=share, index_grid=index_grid,
                     geom=geom, has_alt=has_alt, extra=extra, analyzed_here=analyzed_here)


def _run_analysis(folder, tiles, summary):
    """No analysis files yet: run the team's analysis in memory (nothing is written to the folder)."""
    import pandas as pd
    from src.analysis import analyze
    from src.tilemap import TileMap

    shape = tuple(summary["grid_shape"]) if summary.get("grid_shape") else None
    tm = TileMap.from_dataframe(tiles, mpp=summary.get("mpp", 0.5), shape=shape)
    scores = None
    if (folder / "phrase_scores.csv").exists():
        scores = phrase_grids(pd.read_csv(folder / "phrase_scores.csv"), tm.shape)
    return analyze(tm, scores, summary)


def phrase_grids(scores, shape):
    """phrase_scores table -> {phrase key: [R, C] grid}, like scripts/analyze_slide.py does."""
    rows, cols = scores["row"].to_numpy(int), scores["col"].to_numpy(int)
    out = {}
    for key in scores.columns.drop(["row", "col"]):
        grid = np.zeros(shape)
        grid[rows, cols] = scores[key].to_numpy(float)
        out[key] = grid
    return out


def geometry(tiles, summary, thumb):
    """Grid size, and where tile (row, col) sits on the thumbnail (in thumbnail pixels)."""
    shape = summary.get("grid_shape") or [0, 0]
    rows = int(max(tiles["row"].max() + 1, shape[0]))
    cols = int(max(tiles["col"].max() + 1, shape[1]))

    def spacing(pos, idx):        # full-resolution pixels between neighboring tiles
        span = idx.max() - idx.min()
        return (pos.max() - pos.min()) / span if span > 0 else None

    step = spacing(tiles["x"], tiles["col"]) or spacing(tiles["y"], tiles["row"]) or 448.0
    x0 = float(np.median(tiles["x"] - tiles["col"] * step))
    y0 = float(np.median(tiles["y"] - tiles["row"] * step))
    ds = summary.get("thumbnail_downsample") or ((x0 + cols * step) / thumb.width if thumb is not None else 64)
    tile_um = float(summary.get("tile_um") or float(summary.get("tile_size_px") or 224) * float(summary.get("mpp") or 0.5))
    width = thumb.width if thumb is not None else int((x0 + cols * step) / ds) + 1
    height = thumb.height if thumb is not None else int((y0 + rows * step) / ds) + 1
    return {"rows": rows, "cols": cols, "step": step / ds, "x0": x0 / ds, "y0": y0 / ds,
            "tile_um": tile_um, "width": width, "height": height}


def has_previews(slide):
    return (slide.folder / "tiles").is_dir()


def tile_picture(slide, data, i):
    """
    (image, caption) for tile i: the saved 128 px preview from Delta if there is one, otherwise a
    crop of the slide thumbnail at that spot (real, but blurrier).
    """
    r, c = int(data.rows[i]), int(data.cols[i])
    path = slide.folder / "tiles" / f"{r}_{c}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB"), "This square at 0.5 µm/px (saved preview)"
    thumb = thumbnail(slide.folder)
    if thumb is None:
        return None, "No preview saved for this square."
    g = data.geom
    x, y = g["x0"] + c * g["step"], g["y0"] + r * g["step"]
    box = (max(0, round(x)), max(0, round(y)), min(thumb.width, round(x + g["step"])), min(thumb.height, round(y + g["step"])))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, "No preview saved for this square."
    crop = thumb.crop(box).resize((160, 160), Image.BICUBIC)
    return crop, "From the slide overview (no 0.5 µm/px preview saved for this slide)"


def warm_up(delay_s=0):
    """Load the heavy libraries and every slide in the background, so opening a slide is quick."""
    time.sleep(delay_s)          # let the first screen finish drawing before we compete for the CPU
    try:
        with _lock:
            import pandas  # noqa: F401
            import plotly.graph_objects  # noqa: F401
            import scipy.ndimage  # noqa: F401
            for s in list_slides():
                load(s)
        import tile_map
        for s in list_slides():
            tile_map.scan_jpeg(s)
            for m in tile_map.MODES:
                tile_map.overlay_png(s, m)
    except Exception as e:      # warming up is only a speed-up; never break the app over it
        print(f"[warm-up] skipped: {type(e).__name__}: {e}")


def class_name(i):
    return NAMES[CLASSES[i]] if i >= 0 else "—"
