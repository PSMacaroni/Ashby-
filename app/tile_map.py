"""
The slide map: the scan with see-through colored layers on top, the hover box, clickable squares,
pointers and the legend. The layers and colors are the same as the team's analysis figure
(src/viz.py): tissue map, zones + invasive front + growth arrows, risk points (Reds), certainty.

Why the layer buttons and the opacity slider live INSIDE the chart: Streamlit rebuilds the chart
on every rerun, which resets any zoom or pan the user made. Plotly's own buttons and slider act in
the browser with no rerun, so switching layers or changing the opacity is instant and keeps the view.
The chart gets the scan plus one image per layer; the buttons toggle which image is visible.

Map coordinates are in TILES: tile (row, col) covers x = col..col+1, y = row..row+1, and the
slide thumbnail is stretched to match.
"""
import base64
import io
import threading

import numpy as np
from PIL import Image, ImageFilter

import slide_data
from common import (CLASS_COLORS, CLASSES, FLAG_MARKERS, FRONT_COLOR, GROWTH_COLOR, MATCH_RAMP, NAMES, NOT_ANALYZED,
                    THEME, UNSURE_COLOR, ZONE_SHORT, ZONES, risk_color)

MODES = {"tissue": "Tissue", "zones": "Zones & front", "risk": "Risk", "certainty": "Certainty"}
FAINT = 0.12          # how much of a class's color stays when another class is highlighted
RULE = "─" * 26       # fixed-width line: keeps every hover box the same size
NB = " "         # non-breaking space (normal spaces would be squeezed together)
GLASS_TINT = (233, 236, 239)
OPACITY_STEPS = [round(v * 0.05, 2) for v in range(21)]
LEGEND_HEIGHT = 62           # bottom margin that holds the key (two rows of entries)
CONTROLS_HEIGHT = 50 + LEGEND_HEIGHT   # pixels the in-chart buttons, slider and key add to the figure


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- overlay pictures

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
    elif mode == "certainty":
        colors[data.certainty == "Unsure"] = hex_rgb(UNSURE_COLOR) + (255,)
    elif mode == "risk":
        top = int(data.risk_points.max(initial=0))
        for p in range(1, top + 1):
            colors[data.risk_points == p] = hex_rgb(risk_color(p, top)) + (255,)
    elif mode == "match" and values is not None:
        ramp = np.array([hex_rgb(c) for c in MATCH_RAMP], float)
        v = np.clip(np.asarray(values, float), 0, 1)
        shown = v > 0
        step = np.minimum((v * len(MATCH_RAMP)).astype(int), len(MATCH_RAMP) - 1)
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


def _remember(key, make):
    with _pic_lock:
        if key not in _pictures:
            if len(_pictures) > 80:          # keep memory in check: forget the oldest pictures
                _pictures.pop(next(iter(_pictures)))
            _pictures[key] = make()
        return _pictures[key]


def _scan(slide, data, tint_glass):
    """The slide scan as RGBA, optionally with the empty glass tinted grey."""
    g = data.geom
    thumb = slide_data.thumbnail(slide.folder)
    base = (thumb if thumb is not None else Image.new("RGB", (g["width"], g["height"]), (236, 236, 240))).convert("RGBA")
    if tint_glass:
        # tint the empty glass around the tissue (no tiles there) a soft grey, so the tissue
        # outline stands out on the white page. Tissue pixels are untouched (fat looks white too)
        size = (max(1, round(g["cols"] * g["step"])), max(1, round(g["rows"] * g["step"])))
        grid = ((data.index_grid >= 0) * 255).astype(np.uint8)
        tissue = Image.new("L", base.size, 0)
        tissue.paste(Image.fromarray(grid).resize(size, Image.NEAREST), (round(g["x0"]), round(g["y0"])))
        tissue = tissue.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
        grey = Image.new("RGBA", base.size, GLASS_TINT + (255,))
        base = Image.composite(base, Image.blend(base, grey, 0.75), tissue)
    return base


def _layer(data, mode, focus=None, values=None, size=None):
    """One overlay as a transparent RGBA image the size of the scan."""
    g = data.geom
    rgba = tile_rgba(data, mode, focus, values)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    tiles = (max(1, round(g["cols"] * g["step"])), max(1, round(g["rows"] * g["step"])))
    layer.paste(Image.fromarray(rgba, "RGBA").resize(tiles, Image.NEAREST), (round(g["x0"]), round(g["y0"])))
    return layer


def map_image(slide, mode, opacity, show_overlay=True, focus=None, values=None):
    """The scan with the map blended on top (PIL image, white glass), for exports."""
    data = slide_data.load(slide)
    base = _scan(slide, data, tint_glass=False)
    if show_overlay:
        layer = _layer(data, mode, focus, values, base.size)
        alpha = layer.getchannel("A").point(lambda a: int(a * opacity))
        layer.putalpha(alpha)
        base = Image.alpha_composite(base, layer)
    return base.convert("RGB")


def scan_jpeg(slide):
    """The scan alone (grey glass) as JPEG bytes, for the interactive map."""
    key = ("scan", str(slide.folder), slide_data.stamp(slide.folder))

    def make():
        buf = io.BytesIO()
        _scan(slide, slide_data.load(slide), tint_glass=True).convert("RGB").save(buf, "JPEG", quality=88)
        return buf.getvalue()
    return _remember(key, make)


def overlay_png(slide, mode, focus=None, values=None, values_key=None):
    """One overlay (full alpha; the chart applies the opacity) as PNG bytes."""
    key = ("overlay", str(slide.folder), slide_data.stamp(slide.folder), mode, focus, values_key)

    def make():
        data = slide_data.load(slide)
        buf = io.BytesIO()
        _layer(data, mode, focus, values, (data.geom["width"], data.geom["height"])).save(buf, "PNG", optimize=True)
        return buf.getvalue()
    return _remember(key, make)


def legend_items(data, mode):
    """[(label, hex color)] explaining the colors of one mode."""
    ok = data.ok
    n_ok = max(int(ok.sum()), 1)
    if mode == "tissue":
        items = [(f"{NAMES[c]} {np.mean(data.label[ok] == i) * 100:.0f}%", CLASS_COLORS[c])
                 for i, c in enumerate(CLASSES)]
    elif mode == "certainty":
        items = [(f"Unsure here · {(data.certainty == 'Unsure').sum() / n_ok:.0%} of tissue", UNSURE_COLOR)]
    elif mode == "risk":
        top = max(4, int(data.risk_points.max(initial=0)))
        items = [(f"{p} point{'s' if p > 1 else ''}", risk_color(p, top)) for p in range(1, top + 1)]
        items += [(name.replace("_", " ").capitalize() + " spot", color) for name, (_, color) in FLAG_MARKERS.items()]
    elif mode == "match":
        items = [("Weaker match", MATCH_RAMP[1]), ("Stronger match", MATCH_RAMP[-1])]
    else:
        items = [(name, color) for name, color in ZONES.values()] + [("Invasive front", FRONT_COLOR),
                                                                     ("Growth arrows, satellites", GROWTH_COLOR)]
    if (~ok).any():
        items.append((f"Not analyzed · {int((~ok).sum())} squares", NOT_ANALYZED))
    return items


# ---------------------------------------------------------------- hover box (grouped = fast)

_groups = {}


def label(text):
    return (text + NB * 11)[:11]


def hover_groups(data):
    """
    Squares grouped by everything their hover box SAYS (class, certainty, zone). Each group becomes
    one map layer whose words are written once; per square only two small numbers go to the browser
    (score %, risk points). On a 34,000-tile slide that is ~5x less to send than one text box per
    square. Returns [(texts, tile indices)], in drawing order.
    """
    key = id(data)
    if key not in _groups:
        names = np.array([NAMES[c] for c in CLASSES], dtype=object)
        cls = names[data.label]
        cert = data.certainty.astype(object)
        zone = np.array([ZONE_SHORT.get(z, z) for z in data.zone], dtype=object)
        zone[data.on_front] = "Invasive front"
        bad = ~data.ok
        reason = data.tiles["qc_reason"].to_numpy(dtype=object)
        cls[bad] = "Not analyzed"
        cert[bad] = np.array([f"Skipped · {r}" for r in reason[bad]], dtype=object)
        zone[bad] = "—"
        keys = np.array([f"{a}\x1f{b}\x1f{c}" for a, b, c in zip(cls, cert, zone)])
        uniq, inverse = np.unique(keys, return_inverse=True)
        _groups[key] = [(tuple(k.split("\x1f")), np.flatnonzero(inverse == i)) for i, k in enumerate(uniq)]
    return _groups[key]


def group_template(texts):
    cls, cert, zone = texts
    score = "" if cls == "Not analyzed" else NB * 2 + "%{customdata[0]}%"
    return (f"<b>{cls}</b>{score}<br><span style='color:#B8BEC6'>{RULE}</span><br>"
            + label("Certainty") + f"{cert}<br>" + label("Zone") + f"{zone}<br>"
            + label("Risk pts") + "%{customdata[1]}<br>"
            f"<span style='color:#B8BEC6'>{RULE}</span><br>"
            f"<span style='color:{THEME['muted']}'>Click for details</span><extra></extra>")


def clicked_tile(data, curve, point_index, customdata, first_hover_curve):
    """Which tile a click hit: the hover groups carry the index by position; other curves carry
    the tile index in customdata."""
    groups = hover_groups(data)
    i = curve - first_hover_curve
    if 0 <= i < len(groups) and point_index is not None:
        return int(groups[i][1][int(point_index)])
    while isinstance(customdata, (list, tuple)):
        customdata = customdata[0] if customdata else None
    return int(customdata) if customdata is not None and int(customdata) >= 0 else None


# ---------------------------------------------------------------- the figure

def growth_layer(summary):
    """Where the tumor is pushing out: arrows on the infiltrative arcs of the front (from the team's
    analysis, border.arcs) and rings on the tumor islands that count as satellites."""
    arcs = [a for a in (summary.get("border") or {}).get("arcs", []) if a.get("infiltrative")]
    sats = [p for p in summary.get("satellites") or [] if p.get("satellite")]
    return arcs, sats


def flag_layer(summary):
    """Flag spots from the analysis, drawn like the team's figure."""
    return [(f["key"], f["title"], c) for f in summary.get("flags") or [] if f.get("status") == "ok"
            for c in f.get("candidates", [])]


def _uri(kind, data_bytes):
    return f"data:image/{kind};base64," + base64.b64encode(data_bytes).decode()


def figure(data, scan, overlays, legends, opacity=0.45, active=None, selected=None, markers=(), zoom=None,
           uirevision="map", height=600, growth=None, flags=None, arrow=None):
    """
    The interactive map (plotly).
      scan: JPEG bytes of the slide; overlays: {mode: PNG bytes} (one image per layer; the in-chart
      buttons pick one when there are several); legends: {mode: [(label, color)]}.
      opacity: the slider's starting value; active: the layer shown first.
      markers: [{"r", "c", "label", "color", "hover"}] pointers. zoom: [r0, c0, r1, c1] tiles to
      show (None = whole slide). growth: growth_layer(); flags: flag_layer(); arrow: the main
      direction of invasion ((r, c), (dr, dc)). growth, flags and arrow show with the zones / risk layers.
    Returns (figure, index of the first hover curve) for clicked_tile().
    """
    import plotly.graph_objects as go

    g = data.geom
    S, X0, Y0 = g["step"], g["x0"], g["y0"]
    modes = list(overlays)
    active = active if active in modes else modes[0]
    image_kw = dict(x0=(0.5 - X0) / S, y0=(0.5 - Y0) / S, dx=1 / S, dy=1 / S, hoverinfo="skip")
    traces = [go.Image(source=_uri("jpeg", scan), **image_kw)]
    for m in modes:
        traces.append(go.Image(source=_uri("png", overlays[m]), opacity=opacity, visible=m == active, **image_kw))
    overlay_idx = list(range(1, 1 + len(modes)))
    first_hover = len(traces)

    size = max(6.0, 600 / max(g["cols"], g["rows"]))
    numbers = np.stack([np.round(data.conf * 100), data.risk_points], axis=1).astype(np.uint8)
    for texts, idx in hover_groups(data):
        traces.append(go.Scattergl(
            x=(data.cols[idx] + 0.5).astype(np.float32), y=(data.rows[idx] + 0.5).astype(np.float32),
            customdata=numbers[idx], hovertemplate=group_template(texts), mode="markers", showlegend=False,
            name="",
            # invisible markers to hover; the clicked one is shown by Plotly itself (no rebuild of the
            # chart) as an open blue square
            marker=dict(size=size, symbol="square-open", opacity=0, line=dict(width=2.5, color=THEME["button"])),
            selected=dict(marker=dict(opacity=1, size=16)),
            unselected=dict(marker=dict(opacity=0))))
    per_mode = {m: [] for m in modes}          # trace indices that belong to one layer only

    arcs, sats = growth or ([], [])
    if sats and "zones" in modes:
        per_mode["zones"].append(len(traces))
        traces.append(go.Scatter(
            x=[p["col"] + 0.5 for p in sats], y=[p["row"] + 0.5 for p in sats], mode="markers",
            marker=dict(symbol="circle-open", size=18, color=GROWTH_COLOR, line=dict(width=2)),
            hovertemplate="%{text}<extra></extra>", showlegend=False, name="", visible=active == "zones",
            text=[f"Satellite · {p.get('diameter_um', '?')} µm across · {p.get('distance_to_main_um', '?')} µm "
                  f"from the main tumor · in {p.get('sits_in', '?')}"
                  + (" · possible deposit" if p.get("possible_deposit") else "") for p in sats],
            customdata=[[int(data.index_grid[p["row"], p["col"]])] for p in sats]))
    if flags and "risk" in modes:
        for key in dict.fromkeys(k for k, _, _ in flags):
            symbol, color = FLAG_MARKERS.get(key, ("circle", "#000000"))
            mine = [(t, c) for k, t, c in flags if k == key]
            per_mode["risk"].append(len(traces))
            traces.append(go.Scatter(
                x=[c["col"] + 0.5 for _, c in mine], y=[c["row"] + 0.5 for _, c in mine], mode="markers",
                marker=dict(symbol=symbol, size=11, color=color, line=dict(width=1, color="#FFFFFF")),
                hovertemplate="%{text}<extra></extra>", showlegend=False, name="", visible=active == "risk",
                text=[f"{t}{' (strong)' if c.get('strong') else ''}: {c.get('why', '')}" for t, c in mine],
                customdata=[[int(data.index_grid[c["row"], c["col"]])] for _, c in mine]))

    if markers:
        idx = [int(data.index_grid[m["r"], m["c"]]) if 0 <= m["r"] < g["rows"] and 0 <= m["c"] < g["cols"]
               else -1 for m in markers]
        traces.append(go.Scatter(
            x=[m["c"] + 0.5 for m in markers], y=[m["r"] + 0.5 for m in markers], mode="markers",
            customdata=[[i] for i in idx], hovertemplate="%{text}<extra></extra>",
            text=[m.get("hover", "") for m in markers], showlegend=False, name="",
            marker=dict(symbol="circle-open", size=22, color=[m["color"] for m in markers],
                        line=dict(width=2.5, color=[m["color"] for m in markers])),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))

    for m in modes:                             # legend entries: invisible points that only carry a name
        for text, color in legends.get(m, []):
            per_mode[m].append(len(traces))
            traces.append(go.Scatter(
                x=[None], y=[None], mode="markers", name=text, visible=m == active, hoverinfo="skip",
                marker=dict(symbol="square", size=11, color=color, line=dict(width=1, color="rgba(0,0,0,.18)"))))

    fig = go.Figure(traces)

    # annotations: pointer labels (always) + growth arrows and the direction arrow (zones only)
    base_ann = []
    for m in markers:
        base_ann.append(dict(x=m["c"] + 0.5, y=m["r"] + 0.5, text=f"<b>{m['label']}</b>", showarrow=True, arrowhead=0,
                             arrowwidth=1.5, arrowcolor=m["color"], ax=22, ay=-26, bgcolor="#FFFFFF",
                             bordercolor=m["color"], borderwidth=1.5, borderpad=3, font=dict(color=m["color"], size=12)))
    zone_ann = []
    reach = max(g["rows"], g["cols"]) * 0.07
    for a in arcs:
        (r, c), (dr, dc) = a["center_rc"], a["normal_rc"]
        zone_ann.append(dict(x=c + 0.5 + dc * reach, y=r + 0.5 + dr * reach, ax=c + 0.5, ay=r + 0.5, axref="x",
                             ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2.4,
                             arrowcolor=GROWTH_COLOR, text=""))
    if arrow:
        (r, c), (dr, dc) = arrow
        length = max(g["rows"], g["cols"]) * 0.16
        zone_ann.append(dict(x=c + 0.5 + dc * length, y=r + 0.5 + dr * length, ax=c + 0.5, ay=r + 0.5, axref="x",
                             ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=3,
                             arrowcolor=GROWTH_COLOR, text=""))
    fig.update_layout(annotations=base_ann + (zone_ann if active == "zones" else []))

    if selected is not None:
        r, c = int(data.rows[selected]), int(data.cols[selected])
        for color, pad, width in (("#FFFFFF", 0.15, 3), (THEME["button"], 0, 2)):
            fig.add_shape(type="rect", x0=c - pad, y0=r - pad, x1=c + 1 + pad, y1=r + 1 + pad,
                          line=dict(color=color, width=width))

    # in-chart controls: layer buttons and the opacity slider (no rerun, so the view is kept)
    n = len(traces)
    menus = []

    def visibility(mode):
        """Trace visibility for one layer."""
        visible = [True] * n
        for other in modes:
            for i in [1 + modes.index(other)] + per_mode[other]:
                visible[i] = other == mode
        return visible

    style = dict(bgcolor="#FFFFFF", bordercolor=THEME["border"], borderwidth=1, font=dict(size=12, color=THEME["text"]))
    if len(modes) > 1:
        buttons = []
        for m in modes:
            layout_args = {"annotations": base_ann + (zone_ann if m == "zones" else [])}
            buttons.append(dict(label=MODES.get(m, m), method="update", args=[{"visible": visibility(m)}, layout_args]))
        menus.append(dict(type="buttons", direction="right", x=0, y=1.0, xanchor="left", yanchor="bottom",
                          pad=dict(b=8), showactive=True, active=modes.index(active), buttons=buttons, **style))
    steps = [dict(method="restyle", args=[{"opacity": v}, overlay_idx], label="") for v in OPACITY_STEPS]
    slider = dict(active=min(range(len(OPACITY_STEPS)), key=lambda i: abs(OPACITY_STEPS[i] - opacity)), steps=steps,
                  x=0.62, len=0.38, xanchor="left", y=1.0, yanchor="bottom", pad=dict(t=2, b=4), ticklen=0,
                  minorticklen=0, bgcolor=THEME["border"], bordercolor=THEME["border"], borderwidth=0,
                  currentvalue=dict(visible=True, prefix="Overlay ", suffix="", xanchor="left", offset=0,
                                    font=dict(size=11, color=THEME["muted"])), font=dict(size=1))
    for s_, v in zip(slider["steps"], OPACITY_STEPS):
        s_["value"] = f"{v:.0%}"

    full_x, full_y = full_ranges(data)
    if zoom:
        r0, c0, r1, c1 = zoom
        xr, yr = [c0, c1 + 1], [r1 + 1, r0]
    else:
        xr, yr = full_x, full_y
    # The drawing area keeps the slide's shape (constrain="domain"); every view the app sets has that
    # same shape (zoom_box), so the drawn size never changes. minallowed / maxallowed stop panning and
    # zooming out at the slide's edges.
    pad_x, pad_y = (full_x[1] - full_x[0]) * 0.02, (full_y[0] - full_y[1]) * 0.02
    fig.update_xaxes(visible=False, range=xr, constrain="domain", minallowed=full_x[0] - pad_x, maxallowed=full_x[1] + pad_x)
    fig.update_yaxes(visible=False, range=yr, scaleanchor="x", constrain="domain", minallowed=full_y[1] - pad_y,
                     maxallowed=full_y[0] + pad_y)
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=LEGEND_HEIGHT), height=height, dragmode="pan", hovermode="closest",
        hoverdistance=-1,       # always hover (and so select) the nearest square, however far you are zoomed in
        clickmode="event+select", uirevision=uirevision, paper_bgcolor="#FFFFFF", plot_bgcolor="#F4F6F8",
        updatemenus=menus, sliders=[slider],
        legend=dict(orientation="h", x=0, y=-0.02, xanchor="left", yanchor="top", font=dict(size=11),
                    itemclick=False, itemdoubleclick=False, bgcolor="rgba(0,0,0,0)", entrywidthmode="pixels",
                    entrywidth=150),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#B8BEC6", align="left",
                        font=dict(family=THEME["mono"], size=12, color=THEME["text"])))
    return fig, first_hover


def full_ranges(data):
    """The whole-slide view in tile units: (x range, y range); y runs downward like the image."""
    g = data.geom
    S, X0, Y0 = g["step"], g["x0"], g["y0"]
    return [-X0 / S, (g["width"] - X0) / S], [(g["height"] - Y0) / S, -Y0 / S]


def aspect(data):
    """Width / height of the whole slide (the drawing area has the same shape)."""
    (x0, x1), (y0, y1) = full_ranges(data)
    return (x1 - x0) / max(y0 - y1, 1e-9)


def zoom_box(r, c, data, half=12):
    """A view of about (2 x half) rows around tile (r, c), with the slide's own shape so the drawn
    size doesn't change; shifted (not shrunk) to stay inside the slide."""
    (x0, x1), (y0, y1) = full_ranges(data)
    h = 2 * half
    w = h * aspect(data)
    top, left = r + 0.5 - h / 2, c + 0.5 - w / 2
    top = min(max(top, y1), y0 - h)
    left = min(max(left, x0), x1 - w)
    return [top, left, top + h - 1, left + w - 1]


def neighborhood(slide, data, i, radius=6, scale=4):
    """A zoomed crop of the scan around tile i, with that tile outlined."""
    from PIL import ImageDraw
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
