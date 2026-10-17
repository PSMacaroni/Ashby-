"""
Level 2: one slide. Left: the slide with the tissue map on top (hover a spot = tooltip, click =
details). Right: four panels: the slide summary, the clicked spot, "look here" spots where the AI
is unsure, and risk regions by level.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import explain
import slide_data
import tile_map
import ui
from common import (CAVEATS, CLASSES, COLORS, KNOWN_MIXUPS, NAMES, NOT_ANALYZED_COLOR, RISK_LEVELS,
                    SHORT_NAMES, pct)

PANELS = ["Summary", "Selected spot", "Look here", "Risk regions"]
COLOR_BY = {"Tissue type": "tissue", "How sure the AI is": "confidence", "Risk level": "risk"}
LOOK_COLOR = "#D55E00"
MAX_CARDS = 12
CHART_CONFIG = {"displayModeBar": False}


# ---------------------------------------------------------------- per-slide memory + callbacks

def memory(key):
    """What a slide's screen remembers between clicks: clicked tile, zoom box, last click seen."""
    store = st.session_state.setdefault("slide_memory", {})
    return store.setdefault(key, {"selected": None, "zoom": None, "zoom_rev": 0, "last_click": None})


def focus(key, zoom, idx):
    """'Show on map': zoom the map to a box of tiles and select one tile."""
    mem = memory(key)
    mem.update(zoom=zoom, selected=idx)
    mem["zoom_rev"] += 1


def reset_view(key):
    mem = memory(key)
    mem["zoom"] = None
    mem["zoom_rev"] += 1


def read_click(slide):
    """
    A click on the map reruns the app; the clicked tile arrives in the chart's state. Handle it
    before anything is drawn, so the side panel shows the new tile in this same run.
    """
    mem = memory(slide.key)
    event = st.session_state.get(f"map_{slide.key}")
    try:
        points = event["selection"]["points"]
    except (TypeError, KeyError):
        return
    if not points:
        return
    p = points[0]
    sig = (p.get("curve_number"), p.get("point_index"), p.get("x"), p.get("y"))
    if sig == mem["last_click"]:
        return
    mem["last_click"] = sig
    idx = p.get("customdata")
    if isinstance(idx, (list, tuple)):
        idx = idx[0] if idx else None
    if idx is None:
        return
    mem["selected"] = int(idx)
    st.session_state[f"panel_{slide.key}"] = "Selected spot"


# ---------------------------------------------------------------- the screen

def show(slide):
    data = tile_map.prepare(slide)
    st.session_state.setdefault(f"panel_{slide.key}", "Summary")
    read_click(slide)

    if slide.is_demo:
        ui.demo_banner()
    header(slide, data)
    left, right = st.columns([0.6, 0.4], gap="medium")
    with right:
        panel = st.segmented_control("Panel", PANELS, key=f"panel_{slide.key}", required=True,
                                     label_visibility="collapsed", width="stretch")
        markers, arrow = [], None
        with st.container(height=720, border=False):   # scrolls on its own, so the map stays in view
            if panel == "Selected spot":
                spot_panel(slide, data)
            elif panel == "Look here":
                markers = look_here_panel(slide, data)
            elif panel == "Risk regions":
                markers = risk_panel(slide, data)
            else:
                arrow = summary_panel(slide, data)
    with left:
        map_panel(slide, data, markers, arrow)


def header(slide, data):
    s, g = data["summary"], data["geom"]
    st.markdown(f"### {slide.name}")
    model = explain.get(s, "model.name", "not recorded")
    st.caption(f"Case {slide.case_id} · Slide {slide.slide_id} · Entered {slide.entered} · "
               f"{int(data['ok'].sum()):,} analyzed squares of about {g['tile_um']:.0f} µm · Model: {model}")


def map_panel(slide, data, markers, arrow):
    k = slide.key
    mem = memory(k)
    c1, c2, c3, c4 = st.columns([0.2, 0.33, 0.29, 0.18], vertical_alignment="bottom")
    show_overlay = c1.toggle("Tissue map", value=True, key=f"overlay_{k}")
    mode_name = c2.selectbox("Color by", list(COLOR_BY), key=f"colorby_{k}", disabled=not show_overlay)
    opacity = c3.slider("Map opacity", 0.1, 1.0, 0.5, 0.05, key=f"opacity_{k}", disabled=not show_overlay)
    c4.button("Reset view", key=f"reset_{k}", on_click=reset_view, args=(k,), width="stretch",
              help="Zoom back out to the whole slide")
    mode = COLOR_BY[mode_name]

    jpeg = tile_map.map_jpeg(slide, mode, opacity, show_overlay)
    g = data["geom"]
    height = int(min(720, max(380, 800 * g["height"] / g["width"])))   # roughly the slide's shape
    fig = tile_map.figure(data, jpeg, selected=mem["selected"], markers=markers, zoom=mem["zoom"],
                          arrow=arrow, uirevision=f"{k}-{mem['zoom_rev']}", height=height)
    st.plotly_chart(fig, key=f"map_{k}", on_select="rerun", selection_mode="points",
                    config={"scrollZoom": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]})
    if show_overlay:
        legend(data, mode)
    st.caption("Hover a spot to see what the AI thinks it is; click it for details. "
               "Scroll to zoom, drag to move.")


def legend(data, mode):
    tiles = data["tiles"]
    items = []
    if mode == "tissue":
        fr = (tiles.loc[data["ok"], "label"].value_counts(normalize=True)).to_dict()
        items = [(f"{SHORT_NAMES[c]} {pct(fr.get(c, 0))}", COLORS[c]) for c in CLASSES]
    elif mode == "confidence":
        for level, color in tile_map.SURE_COLORS.items():
            items.append((f"{level} ({int((data['sure'] == level).sum()):,})", color))
    else:
        for level, (name, color) in RISK_LEVELS.items():
            items.append((f"Level {level} {name} ({int((tiles['risk_level'] == level).sum()):,})", color))
    if (~data["ok"]).any():
        items.append((f"Not analyzed ({int((~data['ok']).sum())})", NOT_ANALYZED_COLOR))
    ui.legend(items)


# ---------------------------------------------------------------- right-hand panels

def summary_panel(slide, data):
    s = data["summary"]
    flags = s.get("high_risk_flags") or []
    tumor = explain.get(s, "tumor.fraction_of_tissue")

    st.markdown("**Top risk flags**")
    if flags:
        for f in flags[:3]:
            ui.flag(f)
        if len(flags) > 3:
            st.caption(f"+ {len(flags) - 3} more in the full summary")
    elif tumor == 0:
        st.caption("No tumor found, so no tumor flags.")
    else:
        st.caption("None flagged by the map. That isn't proof they're absent: vessel and nerve "
                   "invasion still need a look by eye.")

    st.markdown("**Key findings**")
    top, frac = explain.most_common_tissue(s)
    lines = ([f"Top AI prediction for the slide: mostly {NAMES[top].lower()} ({pct(frac)} of tissue)."]
             if top else []) + explain.key_findings(s)
    st.markdown("\n".join(f"- {line}" for line in lines))

    if st.toggle("Expand summary", key=f"expand_summary_{slide.key}"):
        if len(flags) > 3:
            st.markdown("**All flags**")
            for f in flags[3:]:
                ui.flag(f)
        for title, section in explain.full_report(s).items():
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {line}" for line in section))
        st.markdown("**All 8 tissue classes (share of tissue)**")
        st.plotly_chart(tissue_chart(s), config=CHART_CONFIG, key=f"tissue_chart_{slide.key}")
        st.markdown("**Caveats**")
        st.markdown("\n".join(f"- {c}" for c in (s.get("caveats") or CAVEATS)))
        st.caption(f"Model: {explain.get(s, 'model.name', 'n/a')} · prompts "
                   f"{explain.get(s, 'model.prompts_version', 'n/a')}")
    st.caption("Decision support only, not a diagnosis.")

    vec, cen = explain.get(s, "invasion.direction.vector_rc"), explain.get(s, "tumor.centroid_rc")
    if vec and cen:
        side, clock = explain.get(s, "invasion.direction.side", ""), explain.get(s, "invasion.direction.clock")
        label = f"front faces {side}" + (f" ({clock} o'clock)" if clock else "")
        return (cen, vec, label)
    return None


def tissue_chart(summary):
    fr = explain.get(summary, "tissue_fractions", {}) or {}
    classes = [c for c in CLASSES if c in fr][::-1]
    fig = go.Figure(go.Bar(x=[fr[c] for c in classes], y=[SHORT_NAMES[c] for c in classes], orientation="h",
                           marker_color=[COLORS[c] for c in classes],
                           text=[pct(fr[c]) for c in classes], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=260, margin=dict(l=0, r=30, t=0, b=0), xaxis=dict(visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def spot_panel(slide, data):
    idx = memory(slide.key)["selected"]
    if idx is None:
        st.info("Click any spot on the slide to see what the AI thinks it is, how sure it is, and why.")
        return
    t = data["tiles"].iloc[idx]
    e = explain.tile_explanation(data, idx)
    conf = f" {e['conf']:.0%}" if e["conf"] is not None else ""
    st.html(f'<span class="ctfa-big">{e["headline"]}{conf}</span> &nbsp;'
            f'{ui.sure_badge(e["sure"])}{ui.risk_badge(e["risk_level"])}')
    st.caption(f"Square at row {int(t['row'])}, column {int(t['col'])} · "
               f"about {data['geom']['tile_um']:.0f} µm wide")

    pic, caption = slide_data.tile_picture(slide, t)
    a, b = st.columns(2)
    if pic is not None:
        a.image(pic, caption=caption, width="stretch")
    else:
        a.info(caption)
    b.image(tile_map.neighborhood(slide, data, idx), caption="Where it sits on the slide", width="stretch")

    st.markdown("**Why the AI thinks so**")
    st.write(e["why"][0])
    st.markdown("**How sure it is**")
    st.write(explain.sure_sentence(e))
    if e["advice"]:
        st.warning(e["advice"], icon="🔎")
    if e["risk_level"]:
        name = RISK_LEVELS[e["risk_level"]][0]
        st.markdown(f"**Risk level {e['risk_level']} ({name})**" +
                    (f": {e['risk_flags']}" if e["risk_flags"] else ""))

    if st.toggle("Expand spot details", key=f"expand_spot_{slide.key}"):
        st.markdown("**All 8 probabilities**")
        st.dataframe(prob_table(data, idx), hide_index=True, width="stretch",
                     column_config={"Combined": st.column_config.ProgressColumn(
                         "Combined", min_value=0, max_value=100, format="%.0f%%")})
        st.markdown("**Full reasoning**")
        st.markdown("\n".join(f"- {line}" for line in e["why"]))
        if e["mixup"]:
            st.caption(e["mixup"])
        st.caption("These are guesses for the whole square (many cells across); the AI can't see single cells.")


def prob_table(data, idx):
    rows = {"Class": [NAMES[c] for c in CLASSES], "Combined": data["probs"][idx] * 100}
    if data["has_models"]:
        rows["CONCH"] = [pct(p) for p in data["model"]["conch"]["probs"][idx]]
        rows["MUSK"] = [pct(p) for p in data["model"]["musk"]["probs"][idx]]
    return pd.DataFrame(rows).sort_values("Combined", ascending=False)


def look_here_panel(slide, data):
    spots = data["spots"]
    n_ok = max(1, int(data["ok"].sum()))
    n_unsure = int((data["sure"] == "Unsure").sum())
    st.markdown("**Where the AI is unsure**")
    rule = "the top score is under 60% or CONCH and MUSK disagree" if data["has_models"] \
        else "the top score is under 60%"
    st.caption(f"{n_unsure / n_ok:.0%} of tissue squares are unsure ({rule}). "
               "These are the biggest clumps of them, worth a look by eye.")
    if not spots:
        st.success("No clumps of unsure squares on this slide.")
        return []
    tile_mm2 = (data["geom"]["tile_um"] / 1000) ** 2
    markers = []
    for i, sp in enumerate(spots, 1):
        a, b = sp["pair"]
        body = (f"{sp['n_tiles']} squares ({sp['n_tiles'] * tile_mm2:.2f} mm²) · average score "
                f"{sp['mean_conf']:.0%}<br>Mostly <b>{SHORT_NAMES[a]}</b> vs <b>{SHORT_NAMES[b]}</b>")
        if data["has_models"]:
            body += f" · the models disagree on {pct(sp['disagree_share'])} of it"
        mix = KNOWN_MIXUPS.get(frozenset((a, b)))
        if mix:
            body += f"<br><small>{mix}</small>"
        ui.card(f"#{i}: {SHORT_NAMES[a]} or {SHORT_NAMES[b]}?", body, LOOK_COLOR)
        st.button(f"Show #{i} on the map", key=f"look_{slide.key}_{i}", type="tertiary", icon="📍",
                  on_click=focus, args=(slide.key, tile_map.zoom_box(sp["bbox_rc"], data), sp["center_idx"]))
        markers.append({"idx": sp["center_idx"], "label": str(i), "color": LOOK_COLOR,
                        "hover": f"Unsure spot #{i}: {SHORT_NAMES[a]} or {SHORT_NAMES[b]}?"})
    return markers


def risk_panel(slide, data):
    s = data["summary"]
    regions = s.get("risk_regions") or []
    if not regions:
        if explain.get(s, "tumor.fraction_of_tissue") == 0:
            st.info("No tumor found, so there are no risk regions.")
        else:
            st.info("Risk regions haven't been computed for this slide yet.")
        return []

    levels = st.segmented_control(
        "Show levels", [4, 3, 2, 1], selection_mode="multi", default=[4, 3, 2], key=f"levels_{slide.key}",
        format_func=lambda lvl: f"L{lvl} {RISK_LEVELS[lvl][0]}")
    with st.expander("How risk levels work"):
        st.markdown(
            "Each tissue square gets points for risk features found within about 0.5 mm of it "
            "(for example an infiltrative front +1, a possible venous invasion candidate +2). "
            "Its level is its points, capped at 4. Neighboring squares with the same level form a region.\n\n"
            "These are places to look first, not findings. The points table is a placeholder until the "
            "team's version is final" + (f" (version: {s.get('risk_points_version')})." if s.get("risk_points_version") else "."))

    shown = [r for r in regions if r.get("level") in (levels or [])]
    counts = {lvl: sum(r.get("level") == lvl for r in regions) for lvl in RISK_LEVELS}
    st.caption(" · ".join(f"Level {lvl}: {n} region{'s' if n != 1 else ''}" for lvl, n in counts.items()))
    tile_mm2 = (data["geom"]["tile_um"] / 1000) ** 2
    markers = []
    for r in shown[:MAX_CARDS]:
        name, color = RISK_LEVELS[r["level"]]
        body = (ui.risk_badge(r["level"]) + f" {r.get('n_tiles', 0)} squares "
                f"({r.get('n_tiles', 0) * tile_mm2:.2f} mm²)")
        if r.get("reasons"):
            body += "<br>Nearby: " + "; ".join(r["reasons"])
        if r.get("suggested_actions"):
            body += f"<br><i>Check: {r['suggested_actions'][0]}</i>"
        ui.card(f"Region {r['id']}: {r.get('title', '')}", body, color)
        r0c0r1c1 = r.get("bbox_rc") or (r["center_rc"] * 2)
        center = int(data["index_grid"][r["center_rc"][0], r["center_rc"][1]])
        st.button(f"Show region {r['id']} on the map", key=f"risk_{slide.key}_{r['id']}", type="tertiary",
                  icon="📍", on_click=focus, args=(slide.key, tile_map.zoom_box(r0c0r1c1, data),
                                                   center if center >= 0 else None))
        if center >= 0:
            markers.append({"idx": center, "label": f"R{r['id']}", "color": color,
                            "hover": f"Region {r['id']} · level {r['level']}: {r.get('title', '')}"})
    if len(shown) > MAX_CARDS:
        st.caption(f"+ {len(shown) - MAX_CARDS} smaller regions not listed.")
    st.caption("Tip: set Color by → Risk level to see every square's level.")
    return markers
