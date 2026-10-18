"""
The slide map: the colored overlay picture, the hover box, clickable points, markers and the
legend. Map coordinates are in TILES: tile (row, col) covers x = col..col+1, y = row..row+1, and
the thumbnail image is stretched to match.
"""
import base64
import io
import threading

import numpy as np
from PIL import Image, ImageDraw

import slide_data
from common import (CERTAINTY, CLASS_COLORS, CLASSES, NAMES, NOT_ANALYZED, RISK_RAMP, THEME, ZONES,
                    risk_color)

MODES = {"tissue": "Tissue", "certainty": "Certainty", "risk": "Risk", "zones": "Zones"}
FAINT = 0.12          # how much of a class's color stays when another class is highlighted
RULE = "─" * 30       # fixed-width line: keeps every hover box the same size
NB = " "         # non-breaking space (normal spaces would be squeezed together)

ZONE_SHORT = {"core": "Core", "front_band": "Front band", "peritumoral": "Peritumoral", "outside": "Outside", "": "—"}


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- overlay picture

def tile_rgba(data, mode, focus=None):
    """[rows, cols, 4] overlay colors for one coloring mode. focus = a class to highlight."""
    g = data.geom
    out = np.zeros((g["rows"], g["cols"], 4), np.uint8)
    colors = np.zeros((len(data.rows), 4), np.uint8)
    if mode == "tissue":
        table = np.array([hex_rgb(CLASS_COLORS[c]) + (255,) for c in CLASSES], np.uint8)
        colors[:] = table[data.label]
        if focus in CLASSES:
            colors[data.label != CLASSES.index(focus), 3] = int(255 * FAINT)
    elif mode == "certainty":
        for level, color in CERTAINTY.items():
            if color:
                colors[data.certainty == level] = hex_rgb(color) + (255,)
    elif mode == "risk":
        for p in range(1, int(data.risk_points.max(initial=0)) + 1):
            colors[data.risk_points == p] = hex_rgb(risk_color(p)) + (255,)
    elif mode == "zones":
        for z, (_, color) in ZONES.items():
            colors[data.zone == z] = hex_rgb(color) + (255,)
        colors[data.on_front] = (40, 14, 4, 255)
    colors[~data.ok] = hex_rgb(NOT_ANALYZED) + (255,)
    out[data.rows, data.cols] = colors
    return out


_pictures = {}
_pic_lock = threading.Lock()


def _compose(slide, data, mode, opacity, show_overlay, focus):
    g = data.geom
    thumb = slide_data.thumbnail(slide.folder)
    base = (thumb if thumb is not None else Image.new("RGB", (g["width"], g["height"]), (236, 236, 240))).convert("RGBA")
    if show_overlay:
        rgba = tile_rgba(data, mode, focus)
        rgba[..., 3] = (rgba[..., 3] * opacity).astype(np.uint8)
        size = (max(1, round(g["cols"] * g["step"])), max(1, round(g["rows"] * g["step"])))
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        layer.paste(Image.fromarray(rgba, "RGBA").resize(size, Image.NEAREST), (round(g["x0"]), round(g["y0"])))
        base = Image.alpha_composite(base, layer)
    return base.convert("RGB")


def map_image(slide, mode, opacity, show_overlay=True, focus=None):
    """The thumbnail with the colored map blended on top (PIL image), e.g. for exports."""
    return _compose(slide, slide_data.load(slide), mode, opacity, show_overlay, focus)


def map_jpeg(slide, mode, opacity, show_overlay, focus):
    """Same picture as JPEG bytes for the interactive map; remembered per setting."""
    key = (str(slide.folder), slide_data.stamp(slide.folder), mode, round(opacity, 2), show_overlay, focus)
    with _pic_lock:
        if key not in _pictures:
            buf = io.BytesIO()
            map_image(slide, mode, opacity, show_overlay, focus).save(buf, "JPEG", quality=88)
            _pictures[key] = buf.getvalue()
        return _pictures[key]


def legend_items(data, mode):
    """[(label, hex color)] explaining the colors of one mode."""
    ok = data.ok
    n_ok = max(int(ok.sum()), 1)
    if mode == "tissue":
        items = [(f"{NAMES[c]} {np.mean(data.label[ok] == i) * 100:.0f}%", CLASS_COLORS[c])
                 for i, c in enumerate(CLASSES)]
    elif mode == "certainty":
        items = [(f"Check (score under 80%) · {(data.certainty == 'Check').sum() / n_ok:.0%}", CERTAINTY["Check"]),
                 (f"Unsure · {(data.certainty == 'Unsure').sum() / n_ok:.0%}", CERTAINTY["Unsure"])]
    elif mode == "risk":
        top = int(data.risk_points.max(initial=0))
        items = [(f"{p} point{'s' if p > 1 else ''}", risk_color(p)) for p in range(1, max(top, 1) + 1)]
        if top >= len(RISK_RAMP):
            items[-1] = (f"{len(RISK_RAMP)}+ points", RISK_RAMP[-1])
    else:
        items = [(name, color) for name, color in ZONES.values()] + [("Invasive front edge", "#280E04")]
    if (~ok).any():
        items.append((f"Not analyzed · {int((~ok).sum())}", NOT_ANALYZED))
    return items


# ---------------------------------------------------------------- hover box

_hover = {}


def hover_data(data):
    """Per tile: [class, score, certainty, 2nd model, zone, risk] as a numpy text array (fast to send)."""
    key = id(data)
    if key not in _hover:
        n = len(data.rows)
        why_unsure = np.where(data.has_alt & (data.alt >= 0) & (data.alt != data.label), "models split", "low score")
        cert = np.where(data.certainty == "Unsure", np.char.add("Unsure · ", why_unsure.astype(str)),
                        np.where(data.certainty == "Check", "Check", data.certainty)).astype(object)
        names = np.array([NAMES[c] for c in CLASSES], dtype=object)
        cls = names[data.label]
        score = np.char.add((np.round(data.conf * 100).astype(int)).astype(str), "%").astype(object)
        if data.has_alt:
            alt_names = np.append(names, "—")[data.alt]       # -1 -> "—"
            second = np.where(data.alt == data.label, "Agrees", np.char.add("Says ", alt_names.astype(str))).astype(object)
        else:
            second = np.full(n, "—", dtype=object)
        zone = np.array([ZONE_SHORT.get(z, z) for z in data.zone], dtype=object)
        zone[data.on_front] = "Front edge"
        risk = np.where(data.risk_points > 0, np.char.add(data.risk_points.astype(str), " points"), "—").astype(object)
        reason = data.tiles["qc_reason"].to_numpy(dtype=object)
        bad = ~data.ok
        cls[bad], score[bad] = "Not analyzed", ""
        cert[bad] = np.char.add("Skipped · ", reason[bad].astype(str)) if bad.any() else cert[bad]
        second[bad] = zone[bad] = risk[bad] = "—"
        _hover[key] = np.stack([cls, score, cert, second, zone, risk], axis=1).astype(str)
    return _hover[key]


def label(text):
    return (text + NB * 11)[:11]


HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b>" + NB * 2 + "%{customdata[1]}<br>"
    f"<span style='color:#3A404A'>{RULE}</span><br>"
    + label("Certainty") + "%{customdata[2]}<br>"
    + label("2nd model") + "%{customdata[3]}<br>"
    + label("Zone") + "%{customdata[4]}<br>"
    + label("Risk") + "%{customdata[5]}<br>"
    f"<span style='color:#3A404A'>{RULE}</span><br>"
    f"<span style='color:{THEME['muted']}'>Click for details</span><extra></extra>"
)


# ---------------------------------------------------------------- the figure

def figure(data, jpeg, selected=None, markers=(), zoom=None, arrow=None, uirevision="map", height=600):
    """
    The interactive map (plotly). Hover: the fixed-size box above. Click: Streamlit returns the
    clicked point; on the tile layer (curve 1) its point_index is the tile's index. markers: [{"r", "c", "label", "color", "hover"}] pointers.
    zoom: [r0, c0, r1, c1] tiles to show (None = whole slide). arrow: ((r, c), (dr, dc), text).
    """
    import plotly.graph_objects as go

    g = data.geom
    S, X0, Y0 = g["step"], g["x0"], g["y0"]
    uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    fig = go.Figure(go.Image(source=uri, x0=(0.5 - X0) / S, y0=(0.5 - Y0) / S, dx=1 / S, dy=1 / S,
                             hoverinfo="skip"))
    fig.add_trace(go.Scattergl(
        x=data.cols + 0.5, y=data.rows + 0.5, mode="markers", customdata=hover_data(data),
        hovertemplate=HOVER_TEMPLATE, showlegend=False, name="",
        marker=dict(size=max(6.0, 600 / max(g["cols"], g["rows"])), color="#000", opacity=0),
        selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0))))

    if markers:
        idx = [int(data.index_grid[m["r"], m["c"]]) if 0 <= m["r"] < g["rows"] and 0 <= m["c"] < g["cols"]
               else -1 for m in markers]
        fig.add_trace(go.Scatter(
            x=[m["c"] + 0.5 for m in markers], y=[m["r"] + 0.5 for m in markers], mode="markers",
            customdata=[[i] for i in idx], hovertemplate="%{text}<extra></extra>",
            text=[m.get("hover", "") for m in markers], showlegend=False, name="",
            marker=dict(symbol="circle-open", size=22, color=[m["color"] for m in markers],
                        line=dict(width=2.5, color=[m["color"] for m in markers])),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
        for m in markers:
            fig.add_annotation(x=m["c"] + 0.5, y=m["r"] + 0.5, text=f"<b>{m['label']}</b>", showarrow=True,
                               arrowhead=0, arrowwidth=1.5, arrowcolor=m["color"], ax=22, ay=-26,
                               bgcolor=THEME["surface"], bordercolor=m["color"], borderwidth=1.5, borderpad=3,
                               font=dict(color=THEME["text"], size=12))

    if arrow:
        (r, c), (dr, dc), text = arrow
        length = max(g["rows"], g["cols"]) * 0.16
        fig.add_annotation(x=c + 0.5 + dc * length, y=r + 0.5 + dr * length, ax=c + 0.5, ay=r + 0.5,
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5,
                           arrowcolor=THEME["bg"], text=text, font=dict(color=THEME["text"], size=12),
                           bgcolor="rgba(11,13,16,0.85)", borderpad=4)

    if selected is not None:
        r, c = int(data.rows[selected]), int(data.cols[selected])
        for color, pad, width in (("#FFFFFF", 0.15, 3), (THEME["button"], 0, 2)):
            fig.add_shape(type="rect", x0=c - pad, y0=r - pad, x1=c + 1 + pad, y1=r + 1 + pad,
                          line=dict(color=color, width=width))

    if zoom:
        r0, c0, r1, c1 = zoom
        xr, yr = [c0, c1 + 1], [r1 + 1, r0]
    else:
        xr, yr = [-X0 / S, (g["width"] - X0) / S], [(g["height"] - Y0) / S, -Y0 / S]
    fig.update_xaxes(visible=False, range=xr, constrain="domain")
    fig.update_yaxes(visible=False, range=yr, scaleanchor="x", constrain="domain")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=height, dragmode="pan", hovermode="closest",
        clickmode="event+select", uirevision=uirevision, paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        hoverlabel=dict(bgcolor=THEME["surface"], bordercolor="#3A404A", align="left",
                        font=dict(family=THEME["mono"], size=12, color=THEME["text"])))
    return fig


def zoom_box(r, c, data, half=12):
    """A view of about (2 x half) tiles around tile (r, c), kept inside the slide."""
    g = data.geom
    return [max(0, r - half), max(0, c - half * 1.4), min(g["rows"] - 1, r + half), min(g["cols"] - 1, c + half * 1.4)]


def neighborhood(slide, data, i, radius=6, scale=4):
    """A zoomed crop of the thumbnail around tile i, with that tile outlined."""
    g = data.geom
    thumb = slide_data.thumbnail(slide.folder) or Image.new("RGB", (g["width"], g["height"]), (236, 236, 240))
    x, y = g["x0"] + data.cols[i] * g["step"], g["y0"] + data.rows[i] * g["step"]
    pad = radius * g["step"]
    box = [max(0, x - pad), max(0, y - pad), min(thumb.width, x + g["step"] + pad), min(thumb.height, y + g["step"] + pad)]
    crop = thumb.crop(tuple(round(v) for v in box))
    crop = crop.resize((max(1, crop.width * scale), max(1, crop.height * scale)), Image.BICUBIC)
    d = ImageDraw.Draw(crop)
    left, top, size = (x - box[0]) * scale, (y - box[1]) * scale, g["step"] * scale
    d.rectangle([left - 2, top - 2, left + size + 2, top + size + 2], outline=(255, 255, 255), width=3)
    d.rectangle([left, top, left + size, top + size], outline=hex_rgb(THEME["button"]), width=2)
    return crop
