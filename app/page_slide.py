"""
Screen 2: one slide. Header (what the slide is, a strip of the six key numbers), then the map on
the left and the panel on the right.

Panel tabs:
  Report   high-risk features and key findings; Details has every number in summary.json
  Check    the worklist: every place to look at, grouped, with pointers on the map and a "reviewed"
           tick per item
  Spot     the clicked square: picture, what the model thinks and why, how sure, all 8 scores

Everything runs in one pass: a click on the map or a "Show" link updates both sides.
"""
import numpy as np
import streamlit as st

import export
import lumen
import report
import search
import slide_data
import tile_map
import ui
import worklist
from common import CLASSES, NAMES, THEME, get, risk_color

PANEL_HEIGHT = 640
MAP_WIDTH_PX = 740        # about the map column's width on a 1440-px window; sets the map's height from the slide's shape
FLAG_COLOR = THEME["red"]
MAX_MARKERS = 15


# ---------------------------------------------------------------- state

def memory(key):
    """What a slide's screen remembers between clicks."""
    store = st.session_state.setdefault("slide_memory", {})
    return store.setdefault(key, {"selected": None, "outline": None, "zoom": None, "zoom_rev": 0, "last_click": None,
                                  "search": None, "match_on": True, "markers": [], "last_tab": None,
                                  "spot_via_click": False})


def focus_on(key, loc, data, select=True):
    """Zoom the map around tile loc = (r, c) and select it."""
    mem = memory(key)
    r, c = loc
    mem["zoom"] = tile_map.zoom_box(r, c, data)
    mem["zoom_rev"] += 1
    g = data.geom
    i = int(data.index_grid[r, c]) if 0 <= r < g["rows"] and 0 <= c < g["cols"] else -1
    if select and i >= 0:
        mem["selected"] = mem["outline"] = i


def reset_view(key):
    """Whole slide, no pointers, nothing selected."""
    mem = memory(key)
    mem["zoom"], mem["selected"], mem["outline"], mem["markers"] = None, None, None, []
    mem["zoom_rev"] += 1


def show_link(label, key, loc, data, widget_key):
    """A 'Show on map' link (search results)."""
    if st.button(label, key=widget_key, type="tertiary", icon=":material/my_location:"):
        focus_on(key, loc, data)
        st.rerun()


# ---------------------------------------------------------------- the screen

def show(slide):
    with st.spinner("Loading slide..."):
        data = slide_data.load(slide)
    s = data.summary
    items = worklist.build(s)
    ui.top_bar(f"Slides  ›  {slide.name}", extra=lambda: slide_picker(slide))
    header(slide, data)
    left, right = st.columns([0.6, 0.4], gap="medium")
    with right:
        markers = panel(slide, data, items)
    with left:
        map_area(slide, data, markers)


def slide_picker(slide):
    slides = slide_data.list_slides()
    keys = [s.key for s in slides]
    names = {s.key: s.name for s in slides}
    if slide.key not in keys:
        return

    def jump():
        ui.go({"name": "slide", "key": st.session_state["pick_slide"]})

    st.selectbox("Slide", keys, index=keys.index(slide.key), format_func=names.get, key="pick_slide",
                 label_visibility="collapsed", on_change=jump, width=300)


def header(slide, data):
    s = data.summary
    chips = (ui.chip(slide.site) if slide.site else "") + (ui.chip(f"TCGA report: {slide.stage}", "blue") if slide.stage else "")
    if data.analyzed_here:
        chips += ui.chip("Analysis run by the app", "amber")
    st.html(f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-top:2px">'
            f'<div style="font-weight:650;font-size:1.25rem">{ui.esc(slide.name)}</div>'
            f'<div class="note">Case {ui.esc(slide.case_id)}</div><div>{chips}</div></div>')
    ui.strip(report.strip_cells(s, len(s.get("high_risk_flags") or [])))


# ---------------------------------------------------------------- the map (left)

@st.fragment
def map_area(slide, data, markers):
    """The map is a fragment: the search box, its results and the fit button rerun only the map.
    The layer buttons and the opacity slider are INSIDE the chart (Plotly handles them in the
    browser, no rerun, so your zoom is kept). A click on a square asks for a whole-screen rerun
    (the panel must open the Spot tab)."""
    k, mem, g = slide.key, memory(slide.key), data.geom
    with st.container(key="mapcard"):
        top = st.columns([0.9, 0.1], vertical_alignment="center")
        query = top[0].text_input("Search this slide", key=f"q_{k}", label_visibility="collapsed", icon=":material/search:",
                                  placeholder="Search this slide: lymphocytes at the border, budding, nerve, necrosis in the core")
        if query.strip() != (mem.get("search") or {}).get("query", ""):
            mem["search"] = search.search(slide, data, query) if query.strip() else None
            mem["match_on"] = True
            mem["zoom"], mem["outline"] = None, None       # whole slide, so every match is visible
            mem["zoom_rev"] += 1
            st.rerun()                                      # the map only; the panel stays where it is
        found = mem.get("search") or {}
        matching = found.get("values") is not None and mem.get("match_on", True)
        if found:                                           # the search's pointers replace the panel's while it is on
            markers = [{"r": s_["r"], "c": s_["c"], "label": s_["label"], "color": THEME["accent"],
                        "hover": f"{s_['label']}: match {s_['score']:.0%}"} for s_ in found.get("spots") or []]
        if top[1].button("", icon=":material/fit_screen:", key=f"reset_{k}",
                         help="Whole slide; clears the pointers and the selected square"):
            reset_view(k)
            st.rerun(scope="app")

        if matching:        # a search is on: one layer, colored by how well each square matches it
            overlays = {"match": tile_map.overlay_png(slide, "match", None, found["values"], found["query"])}
        else:
            overlays = {m: tile_map.overlay_png(slide, m) for m in tile_map.MODES}
        legends = {m: tile_map.legend_items(data, m) for m in overlays}
        vec, cen = get(data.summary, "invasion.direction.vector_rc"), get(data.summary, "tumor.centroid_rc")
        # the drawing area gets the slide's shape: MAP_WIDTH_PX wide (the map column on a laptop screen)
        height = int(min(700, max(320, MAP_WIDTH_PX / tile_map.aspect(data)))) + tile_map.CONTROLS_HEIGHT
        fig, first_hover = tile_map.figure(
            data, tile_map.scan_jpeg(slide), overlays, legends, opacity=0.45, selected=mem["outline"],
            markers=markers[:MAX_MARKERS], zoom=mem["zoom"], uirevision=f"{k}-{mem['zoom_rev']}", height=height,
            growth=tile_map.growth_layer(data.summary), flags=tile_map.flag_layer(data.summary),
            arrow=(cen, vec) if (vec and cen) else None)
        event = st.plotly_chart(fig, key=f"map_{k}", on_select="rerun", selection_mode="points",
                                config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"})
        handle_click(event, k, data, first_hover)

        if found:
            search_results(slide, data, found)
        ui.note(f"Hover a square for details · click to inspect it · scroll to zoom · drag to move · "
                f"one square = {g['tile_um']:.0f} µm, {round(1000 / g['tile_um'])} squares ≈ 1 mm")


def handle_click(event, k, data, first_hover):
    """A click on the map: open that square in the Spot tab."""
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
    idx = tile_map.clicked_tile(data, p.get("curve_number", -1), p.get("point_index", p.get("point_number")),
                                p.get("customdata"), first_hover)
    if idx is not None:
        mem["selected"], mem["outline"] = int(idx), None     # Plotly marks the square itself; the chart isn't rebuilt
        mem["tab"], mem["spot_via_click"] = "Spot", True
        st.rerun(scope="app")


# ---------------------------------------------------------------- the panel (right)

def panel(slide, data, items):
    """Draws the panel; returns the markers for the map."""
    k, mem = slide.key, memory(slide.key)
    tab_key = f"tab_{k}"
    want = mem.pop("tab", None)
    if want:
        st.session_state[tab_key] = want
    tabs = ["Report", f"Check ({len(items)})", "Spot", "Details", "Notes"]
    current = st.session_state.get(tab_key) or "Report"       # None after a click on the open tab
    current = next((t for t in tabs if t.split(" ")[0] == current.split(" ")[0]), "Report")
    st.session_state[tab_key] = current

    with st.container(key="panelcard"):
        tab = st.segmented_control("Section", tabs, key=tab_key, label_visibility="collapsed", width="stretch",
                                   required=True) or "Report"       # required: a click on the open tab keeps it
        prev, mem["last_tab"] = mem.get("last_tab"), tab
        if prev is not None and tab != prev and not want:
            mem["spot_via_click"] = False          # the user switched tabs by hand
        if prev == "Spot" and tab != "Spot" and mem["selected"] is not None:
            # leaving the Spot tab: drop the square marker. Plotly drew it in the browser, so the map is
            # redrawn once (the view returns to the last one the app set)
            mem["selected"], mem["outline"] = None, None
            mem["zoom_rev"] += 1
        with st.container(height=PANEL_HEIGHT, border=False):
            # the Check tab sets the pointers. They stay only while a map click has just opened the Spot
            # tab (so that click doesn't redraw the map); switching tabs by hand clears them
            if tab.startswith("Check"):
                mem["markers"] = check_tab(slide, data, items)
            elif tab == "Spot":
                spot_tab(slide, data)
            elif tab == "Details":
                details_tab(slide, data)
            elif tab == "Notes":
                notes_tab(slide)
            else:
                report_tab(slide, data)
        keep = tab.startswith("Check") or (tab == "Spot" and mem.get("spot_via_click"))
        markers = mem["markers"] if keep else []
        a, b = st.columns(2)
        a.download_button("PDF report", lambda: export.pdf_report(slide), file_name=f"{slide.slide_id}_report.pdf",
                          mime="application/pdf", key=f"dl_pdf_{k}", type="primary", width="stretch",
                          on_click="ignore", icon=":material/picture_as_pdf:")
        with b.popover("Map picture", icon=":material/image:", width="stretch"):
            mode = st.selectbox("Layer", list(tile_map.MODES), format_func=tile_map.MODES.get, key=f"png_mode_{k}")
            st.download_button("Download PNG", lambda: export.map_png(slide, mode, 0.55),
                               file_name=f"{slide.slide_id}_{mode}_map.png", mime="image/png", key=f"dl_png_{k}",
                               width="stretch", on_click="ignore")
    return markers


# ---- Report (short: fits without scrolling)

def report_tab(slide, data):
    s = data.summary
    if lumen.available():
        with st.spinner("Writing the summary with Lumen..."):
            text, source = lumen.summarize(report.plain_summary(s, slide.name))
        if source == "Lumen":
            st.html(f'<div class="lead">{ui.esc(text)}</div>')
            ui.note("Written by Lumen from the findings below; it only rewords them.")

    flags = s.get("high_risk_flags") or []
    ui.section(f"High-risk features ({len(flags)})" if flags else "High-risk features")
    if flags:
        for f in flags:
            ui.item(f, color=FLAG_COLOR)
    elif get(s, "tumor.found", True):
        ui.note("None flagged by the map. Vessel and nerve invasion still need a look by eye.")
    else:
        ui.note("No tumor found, so there is nothing to flag.")

    ui.section("Findings")
    ui.rows(report.short_findings(s), tooltips=True)
    ui.note("Hover a line for the analysis's full sentence; every number is in Details.")


# ---- Details (every number the analysis wrote)

def details_tab(slide, data):
    for sec in report.sections(data.summary):
        with st.expander(sec.title):
            if sec.rows:
                ui.rows(sec.rows)
            for it in sec.items:
                ui.item(it["title"], it.get("sub", ""), it["color"])
            if sec.note:
                ui.note(sec.note)


# ---- Notes

def notes_tab(slide):
    k = slide.key
    saved = slide_data.notes(k)
    box_key = f"notes_{k}"
    if box_key not in st.session_state:
        st.session_state[box_key] = saved

    def save():
        slide_data.set_notes(k, st.session_state[box_key])

    st.text_area("Your notes for this slide", key=box_key, height=300, on_change=save,
                 placeholder="Observations, questions for the sign-out, what to look at under the microscope...")
    ui.note("Saved on this computer when you click outside the box (or press Ctrl+Enter), and added to the PDF report."
            + (" Saved." if st.session_state[box_key] == saved and saved else ""))


# ---- Check (the worklist)

def check_tab(slide, data, items):
    k = slide.key
    if not items:
        ui.note("Nothing to check on this slide.")
        return []
    done = slide_data.reviewed(k)
    counts = {g: sum(1 for it in items if it["group"] == g) for g, _ in worklist.GROUPS}
    groups = [g for g, _ in worklist.GROUPS if counts[g]]
    group = st.segmented_control("Group", groups, default=groups[0], key=f"group_{k}", label_visibility="collapsed",
                                 width="stretch", required=True,
                                 format_func=lambda g: f"{worklist.SHORT_NAMES[g]} · {counts[g]}") or groups[0]
    mine = [it for it in items if it["group"] == group]
    ui.note(f"{sum(1 for it in items if it['id'] in done)} of {len(items)} reviewed · the pointers on the map (F1, R1, ...) "
            "are this group's items · hints to look at, not findings")

    for it in mine:
        is_done = it["id"] in done
        c1, c2 = st.columns([0.08, 0.92], vertical_alignment="top")
        tick = c1.checkbox("Reviewed", value=is_done, key=f"done_{k}_{it['id']}", label_visibility="collapsed")
        if tick != is_done:
            slide_data.set_reviewed(k, it["id"], tick)
            st.rerun()
        with c2:
            title = f"{it['label']} · {it['title']}" if it["label"] else it["title"]
            sub = " · ".join(x for x in (it.get("sub", ""), it.get("action", "")) if x)
            ui.item(title, sub, it["color"], done=is_done)
    return worklist.markers(mine)


# ---- Spot

def spot_tab(slide, data):
    k = slide.key
    i = memory(k)["selected"]
    if i is None:
        ui.note("Click any square on the map to see what the model thinks it is, how sure it is, and why.")
        return
    e = report.tile_explanation(data, i)
    tone = {"Sure": "green", "Unsure": "amber", "Not analyzed": ""}.get(e["sure"], "")
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

    pairs = [("Why", " ".join(e["why"][:2]))]
    if e["how"]:
        pairs.append(("Certainty", f"{e['sure']}: " + "; ".join(e["how"]) + "."))
    zone = report.zone_text(data, i)
    if zone:
        pairs.append(("Zone", zone))
    detectors = [f"{name} {data.extra[col][i]:.0%}" for col, name in
                 (("p_immune", "lymphocytes"), ("p_blood", "blood"), ("p_nerve", "nerve"), ("p_vessel", "vessel"), ("p_artery", "artery"))
                 if col in data.extra and not np.isnan(data.extra[col][i])]
    if detectors:
        pairs.append(("Detectors", ", ".join(detectors)))
    pairs.append(("Position", f"Row {int(data.rows[i])}, column {int(data.cols[i])}"))
    ui.rows(pairs)
    if e["advice"]:
        ui.banner(e["advice"])

    ui.section("All 8 scores")
    import pandas as pd
    df = pd.DataFrame({"Class": [NAMES[c] for c in CLASSES], "Score": data.probs[i] * 100})
    st.dataframe(df.sort_values("Score", ascending=False), hide_index=True, width="stretch",
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                                         format="%.0f%%")})
    notes = [e["why"][-1]]
    if e.get("second"):
        notes.append(f"MUSK's second opinion {e['second']} (reference only; the analysis doesn't use it).")
    if e["mixup"]:
        notes.append(e["mixup"])
    ui.note(" ".join(notes))


# ---- Search results (under the map)

def clear_search(k):
    st.session_state[f"q_{k}"] = ""
    mem = memory(k)
    mem["search"] = None
    mem["zoom"], mem["outline"] = None, None
    mem["zoom_rev"] += 1


def search_results(slide, data, found):
    """One strip under the map: what matched, the best spots (click one to zoom there), the match
    coloring switch and a Clear button."""
    k, mem = slide.key, memory(slide.key)
    used = ", ".join(found.get("used") or []) or "nothing"
    zone = search.ZONE_NAMES.get(found.get("zone"), "")
    text = f"“{found.get('query', '')}” · matched {used}" + (f" · limited to the {zone}" if zone else "")
    if found.get("message"):
        text += f" · {found['message'].strip()}"
    ui.note(text)
    with st.container(horizontal=True, vertical_alignment="center"):
        for spot in found.get("spots") or []:
            if st.button(f"{spot['label']} · {spot['score']:.0%}", key=f"s_{k}_{spot['label']}", type="tertiary",
                         help="Zoom the map to this spot"):
                focus_on(k, (spot["r"], spot["c"]), data)
                st.rerun()
        st.space("stretch")
        if found.get("values") is not None:
            on = st.toggle("Color by match", value=mem.get("match_on", True), key=f"match_{k}")
            if on != mem.get("match_on", True):
                mem["match_on"] = on
                st.rerun()
        st.button("Clear", key=f"clear_{k}", icon=":material/close:", on_click=clear_search, args=(k,))
