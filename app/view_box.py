"""
Level 1: the Slide Box, laid out like a photo grid: a header with counts, then one square card per
slide (three per row). Click a card to open the slide; the ⋯ button opens its options
(rename and copy details, duplicate, export, remove). Streamlit can't catch right-clicks, so the
⋯ button replaces the right-click menu.
"""
import json

import streamlit as st

import export
import slide_data
import ui
from common import THEME as T

COLUMNS = 3
SORTS = {
    "Newest": lambda s: (s.entered, s.name),
    "Name": lambda s: s.name.lower(),
    "Most flags": lambda s: (-len(s.flags), s.name.lower()),
}

CSS = f"""
<style>
  .pf {{ display: flex; align-items: center; gap: 22px; padding: 14px 0 6px; }}
  .pf img {{ width: 84px; height: 84px; border-radius: 50%; padding: 12px; background: {T['surface']};
             border: 1px solid {T['border']}; }}
  .pf h2 {{ margin: 0 0 4px; font-size: 1.45rem; }}
  .pf p {{ margin: 6px 0 0; color: {T['text2']}; font-size: 0.9rem; }}
  [class*="st-key-card_"] {{ position: relative; background: {T['surface']}; border: 1px solid {T['border']};
        border-radius: 12px; overflow: hidden; gap: 0 !important; transition: border-color .15s, transform .15s; }}
  [class*="st-key-card_"]:hover {{ border-color: #3A4250; transform: translateY(-2px); }}
  [class*="st-key-card_"] [data-testid="stElementToolbar"] {{ display: none; }}
  [class*="st-key-card_"] img {{ border-radius: 0 !important; display: block; }}
  [class*="st-key-open_"] {{ position: absolute !important; inset: 0; width: auto !important; z-index: 2; }}
  [class*="st-key-open_"] div, [class*="st-key-open_"] button {{ width: 100% !important; height: 100% !important; }}
  [class*="st-key-open_"] button {{ opacity: 0; cursor: pointer; }}
  [class*="st-key-menu_"] {{ position: absolute !important; top: 10px; right: 10px; z-index: 3; width: auto !important; }}
  [class*="st-key-menu_"] button {{ background: rgba(11,13,16,.75) !important; border-color: rgba(255,255,255,.14) !important;
        min-height: 32px; padding: 0 8px; }}
  .ov {{ position: absolute; left: 0; right: 0; bottom: 100%; aspect-ratio: 4 / 3; display: flex; align-items: center;
        justify-content: center; gap: 22px; background: rgba(5,7,10,.55); color: #fff; font-weight: 650;
        font-size: 0.95rem; opacity: 0; transition: opacity .15s; pointer-events: none; z-index: 1; }}
  [class*="st-key-card_"]:hover .ov {{ opacity: 1; }}
  .cb {{ padding: 12px 14px 10px; }}
  .cb .nm {{ font-weight: 650; color: {T['text']}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .cb .mt {{ color: {T['muted']}; font-size: 0.8rem; margin: 2px 0 8px; }}
</style>
"""


def show():
    st.html(CSS)
    slides = slide_data.list_slides(include_hidden=True)
    visible = [s for s in slides if not s.hidden]
    hidden = [s for s in slides if s.hidden]

    with_tumor = sum(s.tumor_found for s in visible)
    flagged = sum(bool(s.flags) for s in visible)
    st.html(f'<div class="pf"><img src="{ui.LOGO_URI}" alt=""><div><h2>Slide Box</h2>'
            f'<span class="stat"><b>{len(visible)}</b> slides</span>'
            f'<span class="stat"><b>{with_tumor}</b> with tumor</span>'
            f'<span class="stat"><b>{flagged}</b> with flags to check</span>'
            f'<p>Processed colon H&amp;E slides. Open one to explore its tissue map.</p></div></div>')

    find, sort = st.columns([0.7, 0.3], vertical_alignment="bottom")
    query = find.text_input("Find", placeholder="Search by name or case ID", label_visibility="collapsed",
                            icon=":material/search:")
    order = sort.segmented_control("Sort", list(SORTS), default="Newest", label_visibility="collapsed",
                                   key="box_sort", width="stretch") or "Newest"
    st.divider()

    if query:
        q = query.lower()
        visible = [s for s in visible if q in s.name.lower() or q in s.case_id.lower() or q in s.slide_id.lower()]
    visible.sort(key=SORTS[order], reverse=order == "Newest")
    if not visible:
        ui.note("No slides match. Real results go in data/results/<slide_id>/ (copied from Delta).")

    for start in range(0, len(visible), COLUMNS):
        for col, slide in zip(st.columns(COLUMNS, gap="medium"), visible[start:start + COLUMNS]):
            with col:
                card(slide)

    if not any(not s.is_demo for s in slides):
        st.html('<div class="banner" style="margin-top:14px">Real slides: the pipeline’s first run on our 5 TCGA '
                'slides (AA-3875, CM-4746, AA-3844, AH-6547, A6-3809; Delta job 22962239) saved its results on Delta '
                'in data/results/. Copy those folders into data/results/ on this laptop and they appear here, with '
                'the real scan under the map.</div>')

    if hidden:
        with st.expander(f"Removed from the box ({len(hidden)}): files are still on disk"):
            for s in hidden:
                a, b = st.columns([0.8, 0.2], vertical_alignment="center")
                a.write(f"{s.name} · {s.case_id}")
                b.button("Restore", key=f"restore_{s.key}", on_click=slide_data.restore, args=(s.key,))


def card(slide):
    hover = f"<span>Tumor {slide.tumor_fraction:.0%}</span>" if slide.tumor_found else "<span>No tumor</span>"
    hover += f"<span>{len(slide.flags)} flag{'s' if len(slide.flags) != 1 else ''}</span>"
    chips = ui.chip("Demo", "amber") if slide.is_demo else ""
    chips += ui.chip("Copy") if slide.is_copy else ""
    if slide.stage:
        chips += ui.chip(slide.stage)
    if not slide.tumor_found:
        chips += ui.chip("No tumor found")
    elif slide.flags:
        chips += ui.chip(f"{len(slide.flags)} to check", "red")
    else:
        chips += ui.chip("No flags", "blue")

    with st.container(key=f"card_{slide.key}"):
        st.image(slide_data.card_picture(slide.folder), width="stretch")
        st.html(f'<div class="ov">{hover}</div><div class="cb"><div class="nm">{ui.esc(slide.name)}</div>'
                f'<div class="mt">Case {ui.esc(slide.case_id)} · {ui.esc(slide.entered)}</div>{chips}</div>')
        st.button(f"Open {slide.name}", key=f"open_{slide.key}", on_click=ui.go,
                  args=({"name": "slide", "key": slide.key},))
        if st.button("", icon=":material/more_horiz:", key=f"menu_{slide.key}", help="Options"):
            slide_menu(slide.key)


@st.dialog("Slide options", width="medium")
def slide_menu(key):
    """The card's menu in a pop-up window. st.rerun() closes it and refreshes the box."""
    slide = slide_data.get_slide(key)
    if slide is None:
        st.warning("This slide is no longer in the box.")
        return
    st.markdown(f"**{slide.name}**")

    ui.section("Rename")
    name_col, save_col = st.columns([0.75, 0.25], vertical_alignment="bottom")
    new_name = name_col.text_input("New name", value=slide.name, key=f"rename_{key}", label_visibility="collapsed")
    if save_col.button("Save", key=f"save_{key}", width="stretch", type="primary"):
        slide_data.rename(key, new_name)
        st.rerun()
    st.code(f"{slide.name}\nCase: {slide.case_id}\nSlide: {slide.slide_id}\nEntered: {slide.entered}\n"
            f"Flags: {'; '.join(slide.flags) or 'none'}", language=None, wrap_lines=True)
    ui.note("Select and press Ctrl+C, or use the copy icon.")

    ui.section("Export")
    a, b = st.columns(2)
    a.download_button("PDF report", lambda: export.pdf_report(slide), file_name=f"{slide.slide_id}_report.pdf",
                      mime="application/pdf", key=f"exp_pdf_{key}", width="stretch", on_click="ignore",
                      icon=":material/picture_as_pdf:")
    b.download_button("Map picture", lambda: export.map_png(slide, "tissue"), file_name=f"{slide.slide_id}_map.png",
                      mime="image/png", key=f"exp_png_{key}", width="stretch", on_click="ignore",
                      icon=":material/image:")
    a.download_button("Summary (JSON)", lambda: json.dumps(slide_data.read_summary(slide.folder), indent=2),
                      file_name=f"{slide.slide_id}_summary.json", mime="application/json", key=f"exp_json_{key}",
                      width="stretch", on_click="ignore")
    b.download_button("Tiles (CSV)", lambda: (slide.folder / "tiles.csv").read_bytes(),
                      file_name=f"{slide.slide_id}_tiles.csv", mime="text/csv", key=f"exp_csv_{key}",
                      width="stretch", on_click="ignore")

    ui.section("More")
    a, b = st.columns(2)
    if a.button("Duplicate", key=f"dup_{key}", icon=":material/content_copy:", width="stretch"):
        slide_data.duplicate(key)
        st.rerun()
    sure = st.checkbox("Yes, remove it" if not slide.is_copy else "Yes, delete this copy", key=f"confirm_{key}")
    if b.button("Delete copy" if slide.is_copy else "Remove", key=f"del_{key}", icon=":material/delete:",
                disabled=not sure, width="stretch"):
        slide_data.remove(key)
        st.rerun()
    if not slide.is_copy:
        ui.note("Removing only hides the slide; its files stay on disk and you can restore it.")
