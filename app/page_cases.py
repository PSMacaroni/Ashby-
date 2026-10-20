"""
Screen 1: the slide list. One row per processed slide with the facts a pathologist picks a slide
by (site and TCGA report, tumor %, growth pattern, depth hint, high-risk features), an Open button
and a "..." menu (rename, duplicate, hide, downloads). Everything else is inside the slide.
"""
import json

import streamlit as st

import export
import slide_data
import ui
from common import THEME as T
from common import pct

SORTS = {
    "Most high-risk features": lambda s: (-len(s.flags), s.name.lower()),
    "Name": lambda s: s.name.lower(),
    "Tumor share": lambda s: (-s.tumor_fraction, s.name.lower()),
    "Newest": lambda s: (s.entered, s.name.lower()),
}
# (header, width) of the slide rows
HEAD = [("", 0.9), ("Slide", 2.3), ("Site · TCGA report", 1.5), ("Tumor", 0.7), ("Growth pattern", 1.1),
        ("Depth hint", 1.6), ("High-risk", 0.7), ("", 0.7), ("", 0.5)]

CSS = f"""
<style>
  .lh {{ display: grid; grid-template-columns: var(--cols); gap: 12px; padding: 6px 4px; color: {T['muted']};
         font-size: 0.74rem; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid {T['border']}; }}
  [class*="st-key-row_"] {{ border-bottom: 1px solid {T['border']}; padding: 6px 4px; }}
  [class*="st-key-row_"]:hover {{ background: {T['surface']}; }}
  [class*="st-key-row_"] img {{ border: 1px solid {T['border']}; border-radius: 4px; }}
  .cell {{ font-size: 0.9rem; line-height: 1.35; }}
  .cell small {{ display: block; color: {T['muted']}; font-size: 0.78rem; }}
  .cell.num {{ font-variant-numeric: tabular-nums; }}
  /* the row menu: three dots only, no dropdown arrow, never squeezed away */
  [class*="st-key-row_"] [data-testid="stPopoverButton"] {{ min-width: 44px; padding: 0 10px; }}
  [class*="st-key-row_"] [data-testid="stPopoverButton"] > div > div[aria-hidden="true"] {{ display: none; }}
</style>
"""


def cell(col, text, sub="", cls=""):
    col.html(f'<div class="cell {cls}">{ui.esc(text)}' + (f'<small>{ui.esc(sub)}</small>' if sub else "") + '</div>')


def show():
    st.html(CSS)
    slides = slide_data.list_slides(include_hidden=True)
    visible = [s for s in slides if not s.hidden]
    hidden = [s for s in slides if s.hidden]

    st.markdown("#### Slides")
    ui.note(f"{len(visible)} processed colon H&E slides. Every number comes from the team's pipeline (CONCH tissue "
            "model + geometry) and is decision support, not a diagnosis.")
    find_col, sort_col = st.columns([0.7, 0.3], vertical_alignment="center")
    find = find_col.text_input("Find", placeholder="Filter by slide, case, site or stage", label_visibility="collapsed",
                               icon=":material/search:")
    order = sort_col.selectbox("Sort by", list(SORTS), key="sort_slides", label_visibility="collapsed")
    if find:
        q = find.lower()
        visible = [s for s in visible if q in s.name.lower() or q in s.case_id.lower() or q in s.slide_id.lower()
                   or q in s.stage.lower() or q in s.site.lower()]
    visible.sort(key=SORTS[order], reverse=order == "Newest")
    if not visible:
        ui.note("No slides match. Results go in showcase/results/<slide_id>/ or data/results/<slide_id>/ "
                "(copied from Delta).")
    else:
        widths = [w for _, w in HEAD]
        st.html('<div class="lh" style="--cols:' + " ".join(f"{w}fr" for w in widths) + '">'
                + "".join(f"<div>{ui.esc(h)}</div>" for h, _ in HEAD) + "</div>")
        for s in visible:
            row(s, widths)

    if hidden:
        with st.expander(f"Hidden slides ({len(hidden)})"):
            for s in hidden:
                a, b = st.columns([0.8, 0.2], vertical_alignment="center")
                a.write(f"{s.name} · {s.case_id}")
                b.button("Restore", key=f"restore_{s.key}", on_click=slide_data.restore, args=(s.key,))


def row(s, widths):
    with st.container(key=f"row_{s.key}"):
        cols = st.columns(widths, vertical_alignment="center", gap="small")
        cols[0].image(slide_data.small_thumbnail(s.folder, size=(150, 100)), width="stretch")
        cell(cols[1], s.name, f"Case {s.case_id} · processed {s.entered}" + (" · copy" if s.is_copy else ""))
        cell(cols[2], s.site or "—", s.stage or "no report recorded")
        cell(cols[3], pct(s.tumor_fraction) if s.tumor_found else "none", "", "num")
        cell(cols[4], s.pattern)
        cell(cols[5], s.depth)
        cell(cols[6], str(len(s.flags)), "features" if s.flags else "", "num")
        cols[7].button("Open", key=f"open_{s.key}", type="primary", width="stretch", on_click=ui.go,
                       args=({"name": "slide", "key": s.key},))
        with cols[8].popover("", icon=":material/more_horiz:", help="Rename, duplicate, hide, download"):
            menu(s)


def menu(s):
    key = s.key
    a, b = st.columns([0.7, 0.3], vertical_alignment="bottom")
    new_name = a.text_input("Name shown in the app", value=s.name, key=f"rename_{key}")
    if b.button("Save", key=f"save_{key}", width="stretch"):
        slide_data.rename(key, new_name)
        st.rerun()
    ui.section("Download")
    c1, c2 = st.columns(2)
    c1.download_button("PDF report", lambda: export.pdf_report(s), file_name=f"{s.slide_id}_report.pdf",
                       mime="application/pdf", key=f"exp_pdf_{key}", width="stretch", on_click="ignore")
    c2.download_button("Map picture", lambda: export.map_png(s, "tissue"), file_name=f"{s.slide_id}_map.png",
                       mime="image/png", key=f"exp_png_{key}", width="stretch", on_click="ignore")
    c1.download_button("Summary (JSON)", lambda: json.dumps(slide_data.read_summary(s.folder), indent=2),
                       file_name=f"{s.slide_id}_summary.json", mime="application/json", key=f"exp_json_{key}",
                       width="stretch", on_click="ignore")
    c2.download_button("Squares (CSV)", lambda: (s.folder / "tiles.csv").read_bytes(),
                       file_name=f"{s.slide_id}_tiles.csv", mime="text/csv", key=f"exp_csv_{key}", width="stretch",
                       on_click="ignore")
    ui.section("More")
    d1, d2 = st.columns(2)
    if d1.button("Duplicate", key=f"dup_{key}", width="stretch", help="A copy with its own name, notes and review ticks"):
        slide_data.duplicate(key)
        st.rerun()
    if d2.button("Delete copy" if s.is_copy else "Hide", key=f"del_{key}", width="stretch",
                 help="Hiding only removes the slide from the list; its files stay on disk"):
        slide_data.remove(key)
        st.rerun()
