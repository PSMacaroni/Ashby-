"""
Level 2: one slide, laid out like a photo post: the map on the left, a details panel on the right
(Summary · Spot · Unsure · Risk · Flags).

Speed: the panel and the map are separate fragments. Clicking inside the panel reruns only the
panel; changing the map's coloring reruns only the map. The whole screen reruns only when one side
must change the other (a click on the map opens that spot; "Show on map" moves the map).
"""
import numpy as np
import streamlit as st

import explain
import export
import lumen
import search
import slide_data
import tile_map
import ui
from common import CLASS_COLORS, CLASSES, NAMES, RISK_RAMP, THEME, get, pct, risk_color

TABS = ["Summary", "Spot", "Search", "Unsure", "Risk", "Flags"]
PANEL_HEIGHT = 600
FLAG_COLOR = "#E5534B"
UNSURE_COLOR = "#E8B341"
MAX_ITEMS = 12


def memory(key):
    """What a slide's screen remembers between clicks."""
    store = st.session_state.setdefault("slide_memory", {})
    return store.setdefault(key, {"selected": None, "zoom": None, "zoom_rev": 0, "last_click": None,
                                  "markers": [], "arrow": None, "marker_sig": None, "panel_run": None})


def focus_on(key, r, c, data, select=True):
    """Zoom the map around tile (r, c) and select it."""
    mem = memory(key)
    mem["zoom"] = tile_map.zoom_box(r, c, data)
    mem["zoom_rev"] += 1
    i = int(data.index_grid[r, c]) if 0 <= r < data.geom["rows"] and 0 <= c < data.geom["cols"] else -1
    if select and i >= 0:
        mem["selected"] = i


def show(slide):
    with st.spinner("Loading slide..."):
        data = slide_data.load(slide)
    st.session_state["_run"] = st.session_state.get("_run", 0) + 1     # counts full-screen runs
    if slide.is_demo:
        st.html('<div class="banner">Demo slide: made-up data used to build the app. Every number here is fake.</div>')
    left, right = st.columns([0.6, 0.4], gap="medium")
    with right:
        panel(slide, data)          # first: it decides which pointers the map shows
    with left:
        map_area(slide, data)


# ---------------------------------------------------------------- the map (left)

@st.fragment
def map_area(slide, data):
    k, mem, g = slide.key, memory(slide.key), data.geom
    with st.container(key="mapcard"):
        top = st.columns([0.66, 0.34], vertical_alignment="center")
        query = top[0].text_input("Search this slide", key=f"q_{k}", label_visibility="collapsed",
                                  icon=":material/search:",
                                  placeholder="Search this slide, e.g. lymphocytes at the border, budding, nerve")
        view = top[1].segmented_control("View", ["Live map", "Figure"], default="Live map", key=f"view_{k}",
                                        label_visibility="collapsed", width="stretch") or "Live map"
        if query.strip() != (mem.get("search") or {}).get("query", ""):
            mem["search"] = search.search(slide, data, query) if query.strip() else None
            mem["match_on"] = True
            mem["want_tab"] = "Search" if query.strip() else "Summary"
            st.rerun()      # whole screen: the panel shows the results, the map the pointers
        found = mem.get("search") or {}
        matching = found.get("values") is not None and mem.get("match_on", True)
        if view == "Figure":
            overview_figure(slide)
            return

        with st.container(horizontal=True, vertical_alignment="center"):
            mode = st.segmented_control("Color by", list(tile_map.MODES), format_func=tile_map.MODES.get,
                                        default="tissue", key=f"mode_{k}", label_visibility="collapsed") or "tissue"
            st.space("stretch")
            overlay = st.toggle("Overlay", value=True, key=f"overlay_{k}")
            opacity = st.slider("Opacity", 0.15, 1.0, 0.55, 0.05, key=f"opacity_{k}", disabled=not overlay,
                                label_visibility="collapsed", width=110)
            if st.button("", icon=":material/fit_screen:", key=f"reset_{k}", help="Show the whole slide"):
                mem["zoom"] = None
                mem["zoom_rev"] += 1

        focus = st.session_state.get(f"focus_{k}") if mode == "tissue" else None
        if matching:        # a search is on: color by how well each square matches it
            jpeg = tile_map.map_jpeg(slide, "match", opacity, True, None, found["values"], found["query"])
        else:
            jpeg = tile_map.map_jpeg(slide, mode, opacity, overlay, focus)
        height = int(min(640, max(360, 780 * g["height"] / max(g["width"], 1))))
        # phones: the map is narrow, so cap its height to the slide's shape (the picture sits at the top)
        ratio = g["height"] / max(g["width"], 1)
        st.html(f"<style>@media (max-width: 640px) {{ .st-key-mapcard [data-testid='stPlotlyChart'] {{"
                f" max-height: calc((100vw - 4rem) * {ratio:.3f} + 8px); overflow: hidden; }} }}</style>")
        growth = tile_map.growth_layer(data.summary) if (mode == "zones" and overlay and not matching) else None
        fig = tile_map.figure(data, jpeg, selected=mem["selected"], markers=mem["markers"], zoom=mem["zoom"],
                              arrow=mem["arrow"], uirevision=f"{k}-{mem['zoom_rev']}", height=height, growth=growth)
        event = st.plotly_chart(fig, key=f"map_{k}", on_select="rerun", selection_mode="points",
                                config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"})
        handle_click(event, k)

        if matching:
            ui.legend([("Weaker match", RISK_RAMP[1]), ("Stronger match", RISK_RAMP[-1])])
            ui.note(f"Colored by your search “{found['query']}”. Turn it off in the Search tab.")
        elif overlay and mode == "tissue":
            class_filter(data, k)
        elif overlay:
            items = tile_map.legend_items(data, mode)
            if growth and (growth[0] or growth[1]):
                items += [("Arrows: infiltrative arcs (where the front pushes out)", THEME["accent"]),
                          ("Rings: tumor islands", THEME["accent"])]
            ui.legend(items)
        ui.note("Hover for details · click a spot to open it · scroll to zoom · drag to move · "
                "double-click to reset")


def overview_figure(slide):
    """The team's four-panel analysis picture (src/viz.py), made by scripts/analyze_slide.py."""
    path = slide.folder / "analysis_overview.png"
    if path.exists():
        st.image(str(path), width="stretch")
        ui.note("The team's analysis figure: tissue map · zones with the invasive front (black) and arrows where "
                "the front is infiltrative · risk points (1 mm neighborhood) · flags to look at. Made by "
                "scripts/analyze_slide.py from the same results as the live map.")
    else:
        ui.note("No analysis figure for this slide yet. Run: python scripts/analyze_slide.py " + str(slide.folder))


def handle_click(event, k):
    """A click on the map: open that spot in the panel (needs a full rerun so the panel updates)."""
    mem = memory(k)
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
    if p.get("curve_number") == 1:              # the tile layer: point order = tile order
        idx = p.get("point_index", p.get("point_number"))
    else:                                       # a pointer: its tile index is in customdata
        idx = p.get("customdata")
        while isinstance(idx, (list, tuple)):
            idx = idx[0] if idx else None
    if idx is not None and int(idx) >= 0:
        mem["selected"] = int(idx)
        mem["want_tab"] = "Spot"
        st.rerun()


def class_filter(data, k):
    """The tissue legend doubles as a filter: pick a class to highlight it on the map."""
    ok = data.ok
    share = {c: np.mean(data.label[ok] == i) for i, c in enumerate(CLASSES)}
    dots = "".join(f'[class*="st-key-focus_"] button:nth-of-type({n})::before {{ background: {CLASS_COLORS[c]}; }}'
                   for n, c in enumerate(CLASSES, 1))
    st.html(f"""<style>
      [class*="st-key-focus_"] button::before {{ content: ""; width: 9px; height: 9px; border-radius: 3px;
          display: inline-block; margin-right: 7px; }}
      {dots}
    </style>""")
    st.pills("Highlight a class", CLASSES, key=f"focus_{k}", label_visibility="collapsed",
             format_func=lambda c: f"{NAMES[c]} {share[c] * 100:.0f}%")


# ---------------------------------------------------------------- the panel (right)

@st.fragment
def panel(slide, data):
    k, mem, s = slide.key, memory(slide.key), data.summary
    if mem.pop("clear_rerun", False):      # the search was cleared: redraw the whole screen
        st.rerun()
    tab_key = f"tab_{k}"
    want = mem.pop("want_tab", None)
    if want:
        st.session_state[tab_key] = want
    st.session_state.setdefault(tab_key, "Summary")

    with st.container(key="panelcard"):
        model = get(s, "model.name", "model not recorded")
        st.html(f'<div style="font-weight:650;font-size:1.1rem">{ui.esc(slide.name)}</div>'
                f'<div class="note">Case {ui.esc(slide.case_id)} · {ui.esc(slide.entered)} · '
                f'{int(data.ok.sum()):,} squares of {data.geom["tile_um"]:.0f} µm · {ui.esc(model)}</div>')
        tabs = TABS if mem.get("search") else [t for t in TABS if t != "Search"]
        if st.session_state.get(tab_key) not in tabs:
            st.session_state[tab_key] = "Summary"
        tab = st.segmented_control("Section", tabs, key=tab_key, required=True, label_visibility="collapsed",
                                   width="stretch")
        markers, arrow = [], None
        with st.container(height=PANEL_HEIGHT, border=False):
            if tab == "Spot":
                spot_tab(slide, data)
            elif tab == "Search":
                markers = search_tab(slide, data)
            elif tab == "Unsure":
                markers = unsure_tab(slide, data)
            elif tab == "Risk":
                markers = risk_tab(slide, data)
            elif tab == "Flags":
                markers = flags_tab(slide, data)
            else:
                arrow = summary_tab(slide, data)

        a, b = st.columns(2)
        a.download_button("PDF report", lambda: export.pdf_report(slide), file_name=f"{slide.slide_id}_report.pdf",
                          mime="application/pdf", key=f"dl_pdf_{k}", type="primary", width="stretch",
                          on_click="ignore", icon=":material/picture_as_pdf:")
        mode = st.session_state.get(f"mode_{k}") or "tissue"
        b.download_button("Map picture", lambda: export.map_png(slide, mode, st.session_state.get(f"opacity_{k}", 0.55)),
                          file_name=f"{slide.slide_id}_{mode}_map.png", mime="image/png", key=f"dl_png_{k}",
                          width="stretch", on_click="ignore", icon=":material/image:")

    # if this tab wants different pointers on the map, redraw the whole screen once
    sig = (tuple((m["r"], m["c"], m["label"]) for m in markers), arrow is not None)
    fragment_only = mem["panel_run"] == st.session_state.get("_run")
    mem["panel_run"] = st.session_state.get("_run")
    if sig != mem["marker_sig"]:
        mem.update(markers=markers, arrow=arrow, marker_sig=sig)
        if fragment_only:
            st.rerun()


def show_button(label, k, r, c, data, key):
    if st.button(label, key=key, type="tertiary", icon=":material/my_location:"):
        focus_on(k, r, c, data)
        st.rerun()


def summary_tab(slide, data):
    s = data.summary
    facts = explain.plain_summary(s, slide.name)
    if lumen.available():
        with st.spinner("Writing the summary with Lumen..."):
            text, source = lumen.summarize(facts)
        if source == "Lumen":
            st.html(f'<div class="lead">{ui.esc(text)}</div>')
            ui.note("Written by Lumen from the findings below. It only rewords them; check before use.")
        else:
            ui.note(f"Lumen summary unavailable ({source}); showing the findings as computed.")

    flags = s.get("high_risk_flags") or []
    ui.section(f"To check ({len(flags)})" if flags else "To check")
    if flags:
        for f in flags:
            ui.item(f, color=FLAG_COLOR)
    elif get(s, "tumor.found", True):
        ui.note("No high-risk features flagged by the map. Vessel and nerve invasion still need a look by eye.")
    else:
        ui.note("No tumor found, so there is nothing to flag.")

    ui.section("Findings")
    more = st.toggle("Show everything", key=f"more_{slide.key}")
    ui.rows(explain.findings(s, more=more))
    if more:
        ui.section("Tissue mix")
        fr = s.get("tissue_fractions") or {}
        ui.legend([(f"{NAMES[c]} {pct(fr.get(c, 0))}", CLASS_COLORS[c]) for c in sorted(CLASSES, key=lambda c: -(fr.get(c) or 0))])
        ui.section("Caveats")
        for c in s.get("caveats") or []:
            ui.item(c, color=THEME["muted"])
        ui.note(f"Model: {get(s, 'model.name', 'n/a')} · prompts {get(s, 'model.prompts_version', 'n/a')}"
                + (" · analysis run by the app (no tile_analysis.csv)" if data.analyzed_here else ""))
    if not lumen.available():
        ui.note("Lumen summaries turn on when LUMEN_API_KEY, LUMEN_BASE_URL and LUMEN_MODEL are in .env.")

    vec, cen = get(s, "invasion.direction.vector_rc"), get(s, "tumor.centroid_rc")
    if vec and cen:
        return (cen, vec, get(s, "invasion.direction.side", "front"))
    return None


def spot_tab(slide, data):
    i = memory(slide.key)["selected"]
    if i is None:
        ui.note("Click any spot on the map to see what the AI thinks it is, how sure it is, and why.")
        return
    e = explain.tile_explanation(data, i)
    tone = {"Sure": "blue", "Check": "amber", "Unsure": "red"}.get(e["sure"], "")
    chips = ui.chip(e["sure"], tone)
    if e["risk_points"]:
        chips += ui.chip(f"Risk {e['risk_points']} points", dot=risk_color(e["risk_points"]))
    conf = f" · {e['conf']:.0%}" if e["conf"] is not None else ""
    st.html(f'<div style="font-size:1.25rem;font-weight:650">{ui.esc(e["headline"])}{conf}</div>'
            f'<div style="margin-top:4px">{chips}</div>')

    pic, caption = slide_data.tile_picture(slide, data, i)
    a, b = st.columns(2)
    if pic is not None:
        a.image(pic, caption=caption, width="stretch")
    else:
        a.html(f'<div class="note">{ui.esc(caption)}</div>')
    b.image(tile_map.neighborhood(slide, data, i), caption="Where it sits", width="stretch")

    pairs = [("Why", e["why"][0])]
    if e["how"]:
        pairs.append(("Certainty", f"{e['sure']}: " + "; ".join(e["how"]) + "."))
    pairs += [(label, line) for label, line in zip(("Second model", "Neighbors", "Looks like"), e["why"][1:])]
    zone = data.zone[i]
    if zone:
        d = data.dist_um[i]
        where = "" if np.isnan(d) else f", {abs(d) / 1000:.1f} mm {'inside' if d < 0 else 'outside'} the front"
        pairs.append(("Zone", tile_map.ZONE_SHORT.get(zone, zone) + where))
    for col, name in (("p_nerve", "Nerve"), ("p_vessel", "Vessel"), ("p_artery", "Artery"), ("p_immune", "Immune cells")):
        if col in data.extra and data.extra[col][i] >= 0.5:
            pairs.append((name, f"Detector score {data.extra[col][i]:.0%}"))
    pairs.append(("Position", f"Row {int(data.rows[i])}, column {int(data.cols[i])}"))
    ui.rows(pairs)
    if e["advice"]:
        st.html(f'<div class="banner" style="margin-top:10px">{ui.esc(e["advice"])}</div>')

    if st.toggle("All 8 scores", key=f"scores_{slide.key}"):
        import pandas as pd
        df = pd.DataFrame({"Class": [NAMES[c] for c in CLASSES], "Score": data.probs[i] * 100})
        st.dataframe(df.sort_values("Score", ascending=False), hide_index=True, width="stretch",
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                                             format="%.0f%%")})
        if e["mixup"]:
            ui.note(e["mixup"])
    ui.note("One answer per square (many cells across); the map can't see single cells.")


def clear_search(k):
    st.session_state[f"q_{k}"] = ""
    mem = memory(k)
    mem["search"], mem["clear_rerun"] = None, True


def search_tab(slide, data):
    k, mem = slide.key, memory(slide.key)
    found = mem.get("search") or {}
    st.html(f'<div class="lead">Results for “{ui.esc(found.get("query", ""))}”</div>')
    how = ("Live text search (CONCH)" if found.get("how") == "live" else
           "Known-words search: matched to tissue classes, detectors and the analysis's flag phrases")
    used = ", ".join(found.get("used") or []) or "nothing"
    zone = search.ZONE_NAMES.get(found.get("zone"), "")
    ui.rows([("How", how), ("Matched", used)] + ([("Limited to", zone)] if zone else []))
    if found.get("message"):
        ui.note(found["message"])
    markers = []
    if found.get("spots"):
        ui.section("Best matching spots")
        for spot in found["spots"]:
            ui.item(f"{spot['label']} · match {spot['score']:.0%}", "", THEME["accent"])
            show_button(f"Show {spot['label']} on map", k, spot["r"], spot["c"], data, f"s_{k}_{spot['label']}")
            markers.append({"r": spot["r"], "c": spot["c"], "label": spot["label"], "color": THEME["accent"],
                            "hover": f"{spot['label']}: match {spot['score']:.0%}"})
    if found.get("values") is not None:
        on = st.toggle("Color the map by match", value=mem.get("match_on", True), key=f"match_{k}")
        if on != mem.get("match_on", True):
            mem["match_on"] = on
            st.rerun()
    st.button("Clear search", key=f"clear_{k}", icon=":material/close:", on_click=clear_search, args=(k,))
    ui.note("Matches are hints about where to look. The map can't see single cells (e.g. CD8 vs CD4, mitoses).")
    return markers


def unsure_tab(slide, data):
    s = data.summary
    st.html(f'<div class="lead">{ui.esc(get(s, "uncertainty.text", "Uncertainty not computed."))}</div>')
    ui.note("Unsure = a low top score or the two models disagree. Color the map by Certainty to see every square.")
    regions = get(s, "uncertainty.regions", []) or []
    markers = []
    ui.section("Biggest unsure areas")
    if not regions:
        ui.note("No unsure areas.")
    for n, reg in enumerate(regions, 1):
        ui.item(f"U{n} · {reg['n_tiles']} squares", reg.get("why", ""), UNSURE_COLOR)
        show_button(f"Show U{n} on map", slide.key, reg["row"], reg["col"], data, f"u_{slide.key}_{n}")
        markers.append({"r": reg["row"], "c": reg["col"], "label": f"U{n}", "color": UNSURE_COLOR,
                        "hover": f"Unsure area U{n}: {reg.get('why', '')}"})
    return markers


def risk_tab(slide, data):
    s = data.summary
    cards = get(s, "risk.cards", []) or []
    if not cards:
        ui.note("No risk regions on this slide." if not get(s, "tumor.found", True) else
                "No risk regions (no area collected enough points).")
        return []
    levels = sorted({c["level"] for c in cards}, reverse=True)
    min_level = st.select_slider("Show levels from", options=sorted(levels), value=min(levels),
                                 key=f"minlevel_{slide.key}")
    ui.note("Every risk feature adds points to the squares within 1 mm; a region's level is its highest total. "
            "Color the map by Risk to see the points.")
    markers = []
    for n, card in enumerate([c for c in cards if c["level"] >= min_level][:MAX_ITEMS], 1):
        color = risk_color(card["level"])
        feats = "; ".join(card.get("features", []))
        ui.item("", feats, color, title_html=f'Level {card["level"]} · {card["n_tiles"]} squares')
        show_button(f"Show R{n} on map", slide.key, card["row"], card["col"], data, f"r_{slide.key}_{n}")
        markers.append({"r": card["row"], "c": card["col"], "label": f"R{n}", "color": color,
                        "hover": f"R{n} · level {card['level']}"})
    with st.expander("Points table"):
        for row in get(s, "risk.points_table", []) or []:
            ui.item(row["key"].replace("_", " "), f"{row['points']} point(s) · {row['bin']} · {row['source']}")
        ui.note("PLACEHOLDER rows are the team's stand-ins until the features doc's points table is final.")
    return markers


def flags_tab(slide, data):
    s = data.summary
    flags = s.get("flags") or []
    if not flags:
        ui.note("No flags computed for this slide." if get(s, "tumor.found", True) else "No tumor, so no flags.")
        return []
    ui.note("Look-here hints from text matching, detectors and shape rules. Expect false alarms.")
    markers = []
    for f in flags:
        if f.get("status") == "skipped":
            ui.item(f["title"], f"Not computed: {f.get('why_skipped')}", THEME["muted"])
            continue
        strong = [c for c in f.get("candidates", []) if c.get("strong")]
        sub = f"{len(strong)} strong of {len(f.get('candidates', []))} spots · {f.get('source', '')} · {f.get('trust', '')}"
        ui.item(f["title"], sub + (f" · {f['action']}" if f.get("action") else ""), FLAG_COLOR if strong else THEME["muted"])
        for c in strong[:3]:
            n = len(markers) + 1
            if n > MAX_ITEMS:
                break
            markers.append({"r": c["row"], "c": c["col"], "label": f"F{n}", "color": FLAG_COLOR,
                            "hover": f"F{n} · {f['title']}: {c.get('why', '')}"})
        if strong:
            show_button(f"Show {f['title'].lower()} on map", slide.key, strong[0]["row"], strong[0]["col"], data,
                        f"f_{slide.key}_{f['key']}")
    return markers
