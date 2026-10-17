"""
Turns a slide's tiles.csv into the interactive map: where each tile sits on the thumbnail, the
colored overlay picture, hover text, clickable points, markers, and the "AI is unsure" spots.

Coordinates: everything on the map is in THUMBNAIL pixels. A tile's spot on the thumbnail comes
from its x, y (full-resolution pixels) divided by thumbnail_downsample from summary.json.
"""
import base64
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from scipy import ndimage

import slide_data
from common import (CLASSES, COLORS, HIGH_CONF, LOW_CONF, NAMES, NOT_ANALYZED_COLOR, RISK_LEVELS,
                    SHORT_NAMES)

SURE_COLORS = {"Sure": "#2B83BA", "Somewhat sure": "#F2C14E", "Unsure": "#D55E00"}
SQUARE = np.ones((3, 3), bool)


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- loading + per-tile facts

def geometry(tiles, summary, thumb):
    """Grid size and where tile (row, col) sits on the thumbnail."""
    rows = int(max(tiles["row"].max() + 1, (summary.get("grid_shape") or [0, 0])[0]))
    cols = int(max(tiles["col"].max() + 1, (summary.get("grid_shape") or [0, 0])[1]))

    def spacing(pos, idx):        # full-resolution pixels between neighboring tiles
        span = idx.max() - idx.min()
        return (pos.max() - pos.min()) / span if span > 0 else None

    step = spacing(tiles["x"], tiles["col"]) or spacing(tiles["y"], tiles["row"]) \
        or float(summary.get("tile_size_px") or 224)
    x0 = float(np.median(tiles["x"] - tiles["col"] * step))
    y0 = float(np.median(tiles["y"] - tiles["row"] * step))
    ds = summary.get("thumbnail_downsample")
    if not ds:   # guess it from the thumbnail size
        ds = (x0 + cols * step) / thumb.width if thumb is not None else 64
    tile_um = float(summary.get("tile_size_px") or 224) * float(summary.get("mpp") or 0.5)
    width = thumb.width if thumb is not None else int((x0 + cols * step) / ds) + 1
    height = thumb.height if thumb is not None else int((y0 + rows * step) / ds) + 1
    return {"rows": rows, "cols": cols, "step": step / ds, "x0": x0 / ds, "y0": y0 / ds,
            "tile_um": tile_um, "width": width, "height": height}


def neighbor_share(label_grid, ok_grid):
    """For each tile: share of its 8 neighbors (analyzed tissue only) with the same label."""
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    n_tissue = ndimage.convolve(ok_grid.astype(float), kernel, mode="constant")
    same = np.zeros(label_grid.shape)
    for k in range(len(CLASSES)):
        m = (label_grid == k) & ok_grid
        same[m] = ndimage.convolve(m.astype(float), kernel, mode="constant")[m]
    return np.where(n_tissue > 0, same / np.maximum(n_tissue, 1), 0)


def sure_level(conf, agree, share):
    """'Sure' / 'Somewhat sure' / 'Unsure' from the top score, model agreement and neighbors."""
    if conf < LOW_CONF or not agree:
        return "Unsure"
    if conf >= HIGH_CONF and share >= 0.5:
        return "Sure"
    return "Somewhat sure"


def hover_text(t, sure, model_labels, model_probs):
    if not t["qc_ok"]:
        return f"<b>Not analyzed</b><br>{t['qc_reason'] or 'failed quality check'}"
    lines = [f"<b>{NAMES[t['label']]}</b> · {t['confidence']:.0%} · {sure.lower()}"]
    if model_labels:
        (cl, ml), (cp, mp) = model_labels, model_probs
        if cl == ml:
            lines.append(f"CONCH {cp:.0%} · MUSK {mp:.0%} (they agree)")
        else:
            lines.append(f"⚠ CONCH says {SHORT_NAMES[cl]} · MUSK says {SHORT_NAMES[ml]}")
    if t["risk_level"]:
        name = RISK_LEVELS[int(t["risk_level"])][0]
        lines.append(f"Risk level {int(t['risk_level'])} ({name})"
                     + (f": {t['risk_flags']}" if t["risk_flags"] else ""))
    lines.append("<i>click for details</i>")
    return "<br>".join(lines)


@st.cache_data(show_spinner="Loading slide...")
def _prepare(folder_str, _stamp):
    folder = Path(folder_str)
    tiles = slide_data.read_tiles(folder)
    summary = slide_data.read_summary(folder)
    thumb = slide_data.read_thumbnail(folder)
    geom = geometry(tiles, summary, thumb)
    n = len(tiles)
    rows, cols = tiles["row"].to_numpy(), tiles["col"].to_numpy()
    probs = tiles[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
    label_idx = np.array([CLASSES.index(lbl) if lbl in CLASSES else -1 for lbl in tiles["label"]])
    ok = tiles["qc_ok"].to_numpy(bool)

    has_models = slide_data.has_model_columns(tiles, "conch") and slide_data.has_model_columns(tiles, "musk")
    model = {}
    if has_models:
        for name in ("conch", "musk"):
            p = tiles[[f"p_{name}_{c}" for c in CLASSES]].to_numpy(dtype=float)
            model[name] = {"probs": p, "label": p.argmax(1)}
        agree = model["conch"]["label"] == model["musk"]["label"]
    else:
        agree = np.ones(n, bool)

    label_grid = np.full((geom["rows"], geom["cols"]), -1)
    ok_grid = np.zeros((geom["rows"], geom["cols"]), bool)
    index_grid = np.full((geom["rows"], geom["cols"]), -1)
    label_grid[rows, cols] = label_idx
    ok_grid[rows, cols] = ok
    index_grid[rows, cols] = np.arange(n)
    share = neighbor_share(label_grid, ok_grid)[rows, cols]
    conf = tiles["confidence"].to_numpy(dtype=float)
    sure = [sure_level(conf[i], agree[i], share[i]) if ok[i] else "Not analyzed" for i in range(n)]

    hover = []
    for i, t in enumerate(tiles.to_dict("records")):
        if has_models:
            cl, ml = model["conch"]["label"][i], model["musk"]["label"][i]
            labels = (CLASSES[cl], CLASSES[ml])
            pr = (model["conch"]["probs"][i, label_idx[i]], model["musk"]["probs"][i, label_idx[i]])
        else:
            labels = pr = None
        hover.append(hover_text(t, sure[i], labels, pr))

    data = {"tiles": tiles, "summary": summary, "geom": geom, "probs": probs, "label_idx": label_idx,
            "ok": ok, "agree": agree, "share": share, "sure": np.array(sure), "hover": hover,
            "model": model, "has_models": has_models, "index_grid": index_grid}
    data["spots"] = unsure_spots(data)
    return data


def prepare(slide):
    return _prepare(str(slide.folder), slide_data.stamp(slide.folder))


# ---------------------------------------------------------------- "AI is unsure" spots

def unsure_spots(data, max_spots=8, min_tiles=3):
    """Clumps of 'Unsure' tiles, biggest and least confident first."""
    g, tiles = data["geom"], data["tiles"]
    rows, cols = tiles["row"].to_numpy(), tiles["col"].to_numpy()
    unsure = np.zeros((g["rows"], g["cols"]), bool)
    unsure[rows, cols] = data["sure"] == "Unsure"
    lab, n = ndimage.label(unsure, structure=SQUARE)
    second = np.argsort(data["probs"], axis=1)[:, -2]
    conf = tiles["confidence"].to_numpy(dtype=float)
    spots = []
    for k in range(1, n + 1):
        idx = data["index_grid"][lab == k]
        if len(idx) < min_tiles:
            continue
        if data["has_models"]:
            disagree = ~data["agree"][idx]
            pairs = [tuple(sorted((CLASSES[data["model"]["conch"]["label"][i]],
                                   CLASSES[data["model"]["musk"]["label"][i]]))) for i in idx[disagree]]
        else:
            disagree = np.zeros(len(idx), bool)
            pairs = []
        if not pairs:
            pairs = [tuple(sorted((CLASSES[data["label_idx"][i]], CLASSES[second[i]]))) for i in idx]
        pair = max(set(pairs), key=pairs.count)
        cells = np.stack([rows[idx], cols[idx]], 1)
        center = idx[np.argmin(np.hypot(*(cells - cells.mean(0)).T))]
        spots.append({
            "n_tiles": int(len(idx)), "mean_conf": float(conf[idx].mean()),
            "disagree_share": float(disagree.mean()), "pair": pair,
            "center_idx": int(center), "bbox_rc": [int(cells[:, 0].min()), int(cells[:, 1].min()),
                                                    int(cells[:, 0].max()), int(cells[:, 1].max())],
            "score": float(len(idx) * (1 - conf[idx].mean() + disagree.mean())),
        })
    spots.sort(key=lambda s: -s["score"])
    return spots[:max_spots]


# ---------------------------------------------------------------- pictures

def tile_rgba(data, mode):
    """[rows, cols, 4] colors for the overlay: by tissue type, by how sure, or by risk level."""
    g, tiles = data["geom"], data["tiles"]
    out = np.zeros((g["rows"], g["cols"], 4), np.uint8)
    colors = np.zeros((len(tiles), 4), np.uint8)
    if mode == "tissue":
        table = np.array([hex_rgb(COLORS[c]) + (255,) for c in CLASSES], np.uint8)
        colors[:] = table[np.clip(data["label_idx"], 0, None)]
    elif mode == "confidence":
        alpha = {"Sure": 120, "Somewhat sure": 210, "Unsure": 255}
        for level, color in SURE_COLORS.items():
            colors[data["sure"] == level] = hex_rgb(color) + (alpha[level],)
    elif mode == "risk":
        for level, (_, color) in RISK_LEVELS.items():
            colors[tiles["risk_level"].to_numpy() == level] = hex_rgb(color) + (255,)
    colors[~data["ok"]] = hex_rgb(NOT_ANALYZED_COLOR) + (255,)
    out[tiles["row"].to_numpy(), tiles["col"].to_numpy()] = colors
    return out


def blank_thumbnail(geom):
    return Image.new("RGB", (geom["width"], geom["height"]), (236, 236, 240))


@st.cache_data(show_spinner=False)
def _map_jpeg(folder_str, _stamp, mode, opacity, show_overlay):
    data = _prepare(folder_str, _stamp)
    g = data["geom"]
    thumb = slide_data.read_thumbnail(Path(folder_str))
    base = (thumb if thumb is not None else blank_thumbnail(g)).convert("RGBA")
    if show_overlay:
        rgba = tile_rgba(data, mode)
        rgba[..., 3] = (rgba[..., 3] * opacity).astype(np.uint8)
        size = (max(1, round(g["cols"] * g["step"])), max(1, round(g["rows"] * g["step"])))
        over = Image.fromarray(rgba, "RGBA").resize(size, Image.NEAREST)
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        layer.paste(over, (round(g["x0"]), round(g["y0"])))
        base = Image.alpha_composite(base, layer)
    buf = io.BytesIO()
    base.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue()


def map_jpeg(slide, mode, opacity, show_overlay):
    """The slide thumbnail with the colored map blended on top, as JPEG bytes."""
    return _map_jpeg(str(slide.folder), slide_data.stamp(slide.folder), mode, round(opacity, 2), show_overlay)


def neighborhood(slide, data, idx, radius=6, scale=4):
    """A zoomed crop of the thumbnail around one tile, with that tile outlined."""
    g = data["geom"]
    t = data["tiles"].iloc[idx]
    thumb = slide_data.read_thumbnail(slide.folder)
    thumb = thumb if thumb is not None else blank_thumbnail(g)
    x, y = g["x0"] + t["col"] * g["step"], g["y0"] + t["row"] * g["step"]
    pad = radius * g["step"]
    box = [max(0, x - pad), max(0, y - pad), min(thumb.width, x + g["step"] + pad),
           min(thumb.height, y + g["step"] + pad)]
    crop = thumb.crop(tuple(round(v) for v in box))
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.BICUBIC)
    d = ImageDraw.Draw(crop)
    left, top = (x - box[0]) * scale, (y - box[1]) * scale
    d.rectangle([left, top, left + g["step"] * scale, top + g["step"] * scale], outline=(0, 0, 0), width=3)
    d.rectangle([left - 2, top - 2, left + g["step"] * scale + 2, top + g["step"] * scale + 2],
                outline=(255, 255, 255), width=2)
    return crop


# ---------------------------------------------------------------- the Plotly figure

def tile_center(data, idx):
    g, t = data["geom"], data["tiles"].iloc[idx]
    return g["x0"] + (t["col"] + 0.5) * g["step"], g["y0"] + (t["row"] + 0.5) * g["step"]


def rc_center(data, r, c):
    g = data["geom"]
    return g["x0"] + (c + 0.5) * g["step"], g["y0"] + (r + 0.5) * g["step"]


def figure(data, jpeg, selected=None, markers=(), zoom=None, arrow=None, uirevision="map", height=640):
    """
    The map. Hover any tile for its tooltip; clicking selects it (Streamlit returns the tile's row
    number in tiles.csv as customdata).
    markers: [{"idx": tile row, "label": "1", "color": "#hex", "hover": "..."}] drawn as pointers.
    zoom: [r0, c0, r1, c1] tile box to show; None = whole slide.
    arrow: ((r, c) start, (dr, dc) direction, text) for the invasion direction.
    """
    g, tiles = data["geom"], data["tiles"]
    uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    fig = go.Figure(go.Image(source=uri, x0=0, y0=0, dx=1, dy=1, hoverinfo="skip"))

    cx = g["x0"] + (tiles["col"].to_numpy() + 0.5) * g["step"]
    cy = g["y0"] + (tiles["row"].to_numpy() + 0.5) * g["step"]
    fig.add_trace(go.Scattergl(
        x=cx, y=cy, mode="markers", customdata=np.arange(len(tiles)),
        marker=dict(size=max(6.0, g["step"] * 1.2), color="#000", opacity=0),
        selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
        hovertext=data["hover"], hoverinfo="text", showlegend=False, name=""))

    if markers:
        pts = [tile_center(data, m["idx"]) for m in markers]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
            customdata=[m["idx"] for m in markers], hovertext=[m.get("hover", "") for m in markers],
            hoverinfo="text", showlegend=False, name="",
            marker=dict(symbol="circle-open", size=24, color=[m["color"] for m in markers],
                        line=dict(width=3, color=[m["color"] for m in markers])),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
        for (x, y), m in zip(pts, markers):
            fig.add_annotation(x=x, y=y, text=f"<b>{m['label']}</b>", showarrow=True, arrowhead=2,
                               arrowwidth=2, arrowcolor=m["color"], ax=26, ay=-30,
                               bgcolor="white", bordercolor=m["color"], borderwidth=2,
                               font=dict(color="#111", size=13))

    if arrow:
        (r, c), (dr, dc), text = arrow
        x, y = rc_center(data, r, c)
        length = 18 * g["step"]
        fig.add_annotation(x=x + dc * length, y=y + dr * length, ax=x, ay=y, axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=3, arrowcolor="#111",
                           text=text, font=dict(color="#111", size=12), bgcolor="rgba(255,255,255,0.85)")

    if selected is not None:
        t = tiles.iloc[selected]
        x, y = g["x0"] + t["col"] * g["step"], g["y0"] + t["row"] * g["step"]
        for color, pad, width in (("#ffffff", 1.5, 4), ("#000000", 0, 2)):
            fig.add_shape(type="rect", x0=x - pad, y0=y - pad, x1=x + g["step"] + pad,
                          y1=y + g["step"] + pad, line=dict(color=color, width=width))

    if zoom:
        r0, c0, r1, c1 = zoom
        xr = [g["x0"] + c0 * g["step"], g["x0"] + (c1 + 1) * g["step"]]
        yr = [g["y0"] + (r1 + 1) * g["step"], g["y0"] + r0 * g["step"]]
    else:
        xr, yr = [0, g["width"]], [g["height"], 0]
    fig.update_xaxes(visible=False, range=xr, constrain="domain")
    fig.update_yaxes(visible=False, range=yr, scaleanchor="x", constrain="domain")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, dragmode="pan",
                      hovermode="closest", clickmode="event+select", uirevision=uirevision,
                      hoverlabel=dict(bgcolor="white", font_size=13, font_color="#222"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def zoom_box(bbox_rc, data, margin=8, min_size=24):
    """Pad a [r0, c0, r1, c1] box so the zoomed view shows some context around it."""
    r0, c0, r1, c1 = bbox_rc
    rc, cc = (r0 + r1) / 2, (c0 + c1) / 2
    half = max(min_size / 2, (r1 - r0) / 2 + margin, (c1 - c0) / 2 + margin)
    g = data["geom"]
    return [max(0, rc - half), max(0, cc - half * 1.4), min(g["rows"] - 1, rc + half),
            min(g["cols"] - 1, cc + half * 1.4)]
