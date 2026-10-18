"""
The slide map: the scan with a see-through colored layer on top, the hover box, clickable points,
pointers and the legend. The layers and colors are the same as the team's analysis figure
(src/viz.py): tissue map, zones + invasive front + growth arrows, risk points (Reds), flags.

Map coordinates are in TILES: tile (row, col) covers x = col..col+1, y = row..row+1, and the
slide thumbnail is stretched to match.
"""
import base64
import io
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import slide_data
from common import (CERTAINTY, CLASS_COLORS, CLASSES, FLAG_MARKERS, FRONT_COLOR, GROWTH_COLOR, NAMES, NOT_ANALYZED,
                    REDS, RISK_RAMP, THEME, ZONES, risk_color)

MODES = {"tissue": "Tissue", "zones": "Zones", "risk": "Risk", "flags": "Flags", "certainty": "Certainty"}
FAINT = 0.12          # how much of a class's color stays when another class is highlighted
RULE = "─" * 30       # fixed-width line: keeps every hover box the same size
NB = " "         # non-breaking space (normal spaces would be squeezed together)

ZONE_SHORT = {"core": "Core", "front_band": "Front band", "peritumoral": "Peritumoral", "outside": "Other tissue", "": "—"}


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- overlay picture

def tile_rgba(data, mode, focus=None, values=None):
    """[rows, cols, 4] overlay colors for one mode. focus = a class to highlight; values = per-tile
    0-1 match scores for the "match" mode (search)."""
    g = data.geom
    out = np.zeros((g["rows"], g["cols"], 4), np.uint8)
    colors = np.zeros((len(data.rows), 4), np.uint8)
    table = np.array([hex_rgb(CLASS_COLORS[c]) + (255,) for c in CLASSES], np.uint8)
    if mode == "tissue":
        colors[:] = table[data.label]
        if focus in CLASSES:
            colors[data.label != CLASSES.index(focus), 3] = int(255 * FAINT)
    elif mode == "flags":                               # pale tissue map under the flag markers
        colors[:] = table[data.label]
        colors[:, 3] = 110
    elif mode == "certainty":
        for level, color in CERTAINTY.items():
            if color:
                colors[data.certainty == level] = hex_rgb(color) + (255,)
    elif mode == "risk":
        top = int(data.risk_points.max(initial=0))
        for p in range(1, top + 1):
            colors[data.risk_points == p] = hex_rgb(risk_color(p, top)) + (255,)
    elif mode == "match" and values is not None:
        ramp = np.array([hex_rgb(c) for c in RISK_RAMP], float)
        v = np.clip(np.asarray(values, float), 0, 1)
        shown = v > 0
        step = np.minimum((v * len(RISK_RAMP)).astype(int), len(RISK_RAMP) - 1)
        colors[shown, :3] = ramp[step[shown]].astype(np.uint8)
        colors[shown, 3] = (90 + 165 * v[shown]).astype(np.uint8)
    elif mode == "zones":
        for z, (_, color) in ZONES.items():
            colors[data.zone == z] = hex_rgb(color) + (255,)
        colors[data.zone == ""] = hex_rgb(ZONES["outside"][1]) + (255,)
        colors[data.on_front] = hex_rgb(FRONT_COLOR) + (255,)
    colors[~data.ok] = hex_rgb(NOT_ANALYZED) + (255,)
    out[data.rows, data.cols] = colors
    return out


_pictures = {}
_pic_lock = threading.Lock()


def _compose(slide, data, mode, opacity, show_overlay, focus, values=None, dark_glass=False):
    """The slide scan with the see-through map layer blended on top."""
    g = data.geom
    thumb = slide_data.thumbnail(slide.folder)
    base = (thumb if thumb is not None else Image.new("RGB", (g["width"], g["height"]), (236, 236, 240))).convert("RGBA")
    size = (max(1, round(g["cols"] * g["step"])), max(1, round(g["rows"] * g["step"])))
    at = (round(g["x0"]), round(g["y0"]))
    if dark_glass:
        # dim the empty glass around the tissue (no tiles there), so the scan sits on the dark page
        # like in a slide viewer. Tissue pixels are untouched (fat looks white too)
        grid = ((data.index_grid >= 0) * 255).astype(np.uint8)
        tissue = Image.new("L", base.size, 0)
        tissue.paste(Image.fromarray(grid).resize(size, Image.NEAREST), at)
        tissue = tissue.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
        dark = Image.new("RGBA", base.size, hex_rgb(THEME["surface"]) + (255,))
        base = Image.composite(base, Image.blend(base, dark, 0.88), tissue)
    if show_overlay:
        rgba = tile_rgba(data, mode, focus, values)
        rgba[..., 3] = (rgba[..., 3] * opacity).astype(np.uint8)
        if mode == "zones":                              # the invasive front stays a solid line
            rgba[data.rows[data.on_front], data.cols[data.on_front], 3] = 235
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        layer.paste(Image.fromarray(rgba, "RGBA").resize(size, Image.NEAREST), at)
        base = Image.alpha_composite(base, layer)
    return base.convert("RGB")


def map_image(slide, mode, opacity, show_overlay=True, focus=None, values=None):
    """The scan with the map on top (PIL image, white glass), for exports."""
    return _compose(slide, slide_data.load(slide), mode, opacity, show_overlay, focus, values)


def map_jpeg(slide, mode, opacity, show_overlay, focus, values=None, values_key=None):
    """The picture for the interactive map (dark glass) as JPEG bytes; remembered per setting."""
    key = (str(slide.folder), slide_data.stamp(slide.folder), mode, round(opacity, 2), show_overlay, focus, values_key)
    with _pic_lock:
        if key not in _pictures:
            buf = io.BytesIO()
            _compose(slide, slide_data.load(slide), mode, opacity, show_overlay, focus, values,
                     dark_glass=True).save(buf, "JPEG", quality=88)
            if len(_pictures) > 60:          # keep memory in check: forget the oldest pictures
                _pictures.pop(next(iter(_pictures)))
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
        top = max(4, int(data.risk_points.max(initial=0)))
        items = [(f"{p} point{'s' if p > 1 else ''}", risk_color(p, top)) for p in range(1, top + 1)]
    elif mode == "flags":
        items = [(name.replace("_", " ").capitalize(), color) for name, (_, color) in FLAG_MARKERS.items()]
        items.append(("Unsure area", "#7D8590"))
    else:
        items = [(name, color) for name, color in ZONES.values()] + [("Invasive front", FRONT_COLOR),
                                                                     ("Growth arrows, tumor islands", GROWTH_COLOR)]
    if (~ok).any():
        items.append((f"Not analyzed · {int((~ok).sum())}", NOT_ANALYZED))
    return items


# ---------------------------------------------------------------- hover box (grouped = fast)

_groups = {}


def label(text):
    return (text + NB * 11)[:11]


def hover_groups(data):
    """
    Squares grouped by everything their hover box SAYS (class, certainty, second model, zone).
    Each group becomes one map layer whose words are written once; per square only two small numbers
    go to the browser (score %, risk points). On a 34,000-tile slide that is ~5x less to send than
    one text box per square. Returns [(texts, tile indices)], in drawing order.
    """
    key = id(data)
    if key not in _groups:
        why_unsure = np.where(data.has_alt & (data.alt >= 0) & (data.alt != data.label), "models split", "low score")
        cert = np.where(data.certainty == "Unsure", np.char.add("Unsure · ", why_unsure.astype(str)),
                        data.certainty.astype(str)).astype(object)
        names = np.array([NAMES[c] for c in CLASSES], dtype=object)
        cls = names[data.label]
        if data.has_alt:
            alt_names = np.append(names, "—")[data.alt]
            second = np.where(data.alt == data.label, "Agrees", np.char.add("Says ", alt_names.astype(str))).astype(object)
        else:
            second = np.full(len(data.rows), "—", dtype=object)
        zone = np.array([ZONE_SHORT.get(z, z) for z in data.zone], dtype=object)
        zone[data.on_front] = "Invasive front"
        bad = ~data.ok
        reason = data.tiles["qc_reason"].to_numpy(dtype=object)
        cls[bad] = "Not analyzed"
        cert[bad] = np.array([f"Skipped · {r}" for r in reason[bad]], dtype=object)
        second[bad] = zone[bad] = "—"
        keys = np.array([f"{a}\x1f{b}\x1f{c}\x1f{d}" for a, b, c, d in zip(cls, cert, second, zone)])
        uniq, inverse = np.unique(keys, return_inverse=True)
        _groups[key] = [(tuple(k.split("\x1f")), np.flatnonzero(inverse == i)) for i, k in enumerate(uniq)]
    return _groups[key]


def group_template(texts):
    cls, cert, second, zone = texts
    score = "" if cls == "Not analyzed" else NB * 2 + "%{customdata[0]}%"
    return (f"<b>{cls}</b>{score}<br><span style='color:#545C68'>{RULE}</span><br>"
            + label("Certainty") + f"{cert}<br>" + label("2nd model") + f"{second}<br>"
            + label("Zone") + f"{zone}<br>" + label("Risk pts") + "%{customdata[1]}<br>"
            f"<span style='color:#545C68'>{RULE}</span><br>"
            f"<span style='color:{THEME['muted']}'>Click for details</span><extra></extra>")


def clicked_tile(data, curve, point_index, customdata):
    """Which tile a click hit: curves 1..N are the hover groups; later curves carry the index."""
    groups = hover_groups(data)
    if 1 <= curve <= len(groups) and point_index is not None:
        return int(groups[curve - 1][1][int(point_index)])
    while isinstance(customdata, (list, tuple)):
        customdata = customdata[0] if customdata else None
    return int(customdata) if customdata is not None and int(customdata) >= 0 else None


# ---------------------------------------------------------------- the figure

def growth_layer(summary):
    """Where the tumor is pushing out: arrows on the infiltrative arcs of the front (from the team's
    analysis, border.arcs) and rings on the tumor islands (satellites)."""
    arcs = [a for a in (summary.get("border") or {}).get("arcs", []) if a.get("infiltrative")]
    sats = [p for p in summary.get("satellites") or [] if p.get("satellite")]
    return arcs, sats


def flag_layer(summary):
    """Flag spots and unsure areas from the analysis, drawn like the team's figure."""
    spots = [(f["key"], f["title"], c) for f in summary.get("flags") or [] if f.get("status") == "ok"
             for c in f.get("candidates", [])]
    unsure = (summary.get("uncertainty") or {}).get("regions", [])
    return spots, unsure


def figure(data, jpeg, selected=None, markers=(), zoom=None, arrow=None, uirevision="map", height=600,
           growth=None, flags=None):
    """
    The interactive map (plotly). Hover: the fixed-size box. Click: see clicked_tile().
    markers: [{"r", "c", "label", "color", "hover"}] pointers. zoom: [r0, c0, r1, c1] tiles to
    show (None = whole slide). arrow: ((r, c), (dr, dc)) the main direction of invasion.
    growth: growth_layer(); flags: flag_layer().
    """
    import plotly.graph_objects as go

    g = data.geom
    S, X0, Y0 = g["step"], g["x0"], g["y0"]
    uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    fig = go.Figure(go.Image(source=uri, x0=(0.5 - X0) / S, y0=(0.5 - Y0) / S, dx=1 / S, dy=1 / S,
                             hoverinfo="skip"))
    size = max(6.0, 600 / max(g["cols"], g["rows"]))
    numbers = np.stack([np.round(data.conf * 100), data.risk_points], axis=1).astype(np.uint8)
    for texts, idx in hover_groups(data):           # curves 1..N: the hover layers
        fig.add_trace(go.Scattergl(
            x=(data.cols[idx] + 0.5).astype(np.float32), y=(data.rows[idx] + 0.5).astype(np.float32),
            customdata=numbers[idx], hovertemplate=group_template(texts), mode="markers", showlegend=False,
            name="", marker=dict(size=size, color="#000", opacity=0),
            selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0))))

    if growth:
        arcs, sats = growth
        reach = max(g["rows"], g["cols"]) * 0.07
        for a in arcs:
            (r, c), (dr, dc) = a["center_rc"], a["normal_rc"]
            fig.add_annotation(x=c + 0.5 + dc * reach, y=r + 0.5 + dr * reach, ax=c + 0.5, ay=r + 0.5,
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2.4,
                               arrowcolor=GROWTH_COLOR, text="")
        if sats:
            fig.add_trace(go.Scatter(
                x=[p["col"] + 0.5 for p in sats], y=[p["row"] + 0.5 for p in sats], mode="markers",
                marker=dict(symbol="circle-open", size=18, color=GROWTH_COLOR, line=dict(width=2)),
                hovertemplate="%{text}<extra></extra>", showlegend=False, name="",
                text=[f"Tumor island · {p.get('diameter_um', '?')} µm across · {p.get('distance_to_main_um', '?')} µm "
                      f"from the main tumor · in {p.get('sits_in', '?')}"
                      + (" · possible deposit" if p.get("possible_deposit") else "") for p in sats],
                customdata=[[-1] for _ in sats]))

    if flags:
        spots, unsure = flags
        for key in dict.fromkeys(k for k, _, _ in spots):
            symbol, color = FLAG_MARKERS.get(key, ("circle", "#000000"))
            mine = [(t, c) for k, t, c in spots if k == key]
            fig.add_trace(go.Scatter(
                x=[c["col"] + 0.5 for _, c in mine], y=[c["row"] + 0.5 for _, c in mine], mode="markers",
                marker=dict(symbol=symbol, size=11, color=color, line=dict(width=1, color="#FFFFFF")),
                hovertemplate="%{text}<extra></extra>", showlegend=False, name="",
                text=[f"{t}{' (strong)' if c.get('strong') else ''}: {c.get('why', '')}" for t, c in mine],
                customdata=[[int(data.index_grid[c["row"], c["col"]])] for _, c in mine]))
        if unsure:
            fig.add_trace(go.Scatter(
                x=[u["col"] + 0.5 for u in unsure], y=[u["row"] + 0.5 for u in unsure], mode="markers",
                marker=dict(symbol="circle-open", size=24, color="#7D8590", line=dict(width=2.5)),
                hovertemplate="%{text}<extra></extra>", showlegend=False, name="",
                text=[f"Unsure area · {u['n_tiles']} squares · {u.get('why', '')}" for u in unsure],
                customdata=[[-1] for _ in unsure]))

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

    if arrow:                                       # main direction of invasion: just the arrow
        (r, c), (dr, dc) = arrow
        length = max(g["rows"], g["cols"]) * 0.16
        fig.add_annotation(x=c + 0.5 + dc * length, y=r + 0.5 + dr * length, ax=c + 0.5, ay=r + 0.5,
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=3,
                           arrowcolor=GROWTH_COLOR, text="")

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
    fig.update_yaxes(visible=False, range=yr, scaleanchor="x", constrain="domain", constraintoward="top")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=height, dragmode="pan", hovermode="closest",
        clickmode="event+select", uirevision=uirevision, paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
        hoverlabel=dict(bgcolor=THEME["surface"], bordercolor="#3A404A", align="left",
                        font=dict(family=THEME["mono"], size=12, color=THEME["text"])))
    return fig


def zoom_box(r, c, data, half=12):
    """A view of about (2 x half) tiles around tile (r, c), kept inside the slide."""
    g = data.geom
    return [max(0, r - half), max(0, c - half * 1.4), min(g["rows"] - 1, r + half), min(g["cols"] - 1, c + half * 1.4)]


def neighborhood(slide, data, i, radius=6, scale=4):
    """A zoomed crop of the scan around tile i, with that tile outlined."""
    g = data.geom
    thumb = slide_data.thumbnail(slide.folder)
    thumb = thumb if thumb is not None else Image.new("RGB", (g["width"], g["height"]), (236, 236, 240))
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
