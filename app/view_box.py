"""
Level 1: the Slide Box. Every processed slide as a card; Open goes to Level 2.
Each card's ⋮ menu: rename (and copy the details), duplicate, export, remove.
(Streamlit can't catch right-clicks, so the menu is a button instead.)
"""
import json

import streamlit as st

import export
import slide_data
import ui

COLUMNS = 4
SORTS = {
    "Newest first": lambda s: (s.entered, s.name),
    "Name (A-Z)": lambda s: s.name.lower(),
    "Most risk flags": lambda s: (-len(s.flags), -s.top_risk_level, s.name.lower()),
}


def show():
    slides = slide_data.list_slides(include_hidden=True)
    visible = [s for s in slides if not s.hidden]
    hidden = [s for s in slides if s.hidden]

    head, find, sort = st.columns([0.5, 0.28, 0.22], vertical_alignment="bottom")
    head.markdown(f"#### Slide Box &nbsp; <small style='opacity:.6'>{len(visible)} slides</small>",
                  unsafe_allow_html=True)
    query = find.text_input("Find a slide", placeholder="Name or case ID", label_visibility="collapsed")
    order = sort.selectbox("Sort", list(SORTS), label_visibility="collapsed")

    if query:
        q = query.lower()
        visible = [s for s in visible if q in s.name.lower() or q in s.case_id.lower() or q in s.slide_id.lower()]
    visible.sort(key=SORTS[order], reverse=order == "Newest first")

    if not visible:
        st.info("No slides to show. Real results go in data/results/<slide_id>/ (copied from Delta); "
                "demo slides are made in app/demo_data/.")
    for start in range(0, len(visible), COLUMNS):
        cols = st.columns(COLUMNS)
        for col, slide in zip(cols, visible[start:start + COLUMNS]):
            with col:
                slide_card(slide)

    if hidden:
        with st.expander(f"Removed from the box ({len(hidden)}) — files are still on disk"):
            for s in hidden:
                a, b = st.columns([0.75, 0.25], vertical_alignment="center")
                a.write(f"{s.name} · {s.case_id}")
                b.button("Restore", key=f"restore_{s.key}", on_click=slide_data.restore, args=(s.key,))


def status_badges(slide):
    out = []
    if slide.is_demo:
        out.append(ui.badge("DEMO", "#C99A00", filled=True))
    if slide.is_copy:
        out.append(ui.badge("Copy", "#6c757d"))
    n = len(slide.flags)
    if not slide.has_tumor:
        out.append(ui.badge("No tumor found", "#6c757d"))
    elif n == 0:
        out.append(ui.badge("No high-risk flags", "#2B83BA"))
    else:
        out.append(ui.badge(f"⚠ {n} high-risk flag{'s' if n > 1 else ''}", "#E63946", filled=n >= 3))
    return "".join(out)


def slide_card(slide):
    with st.container(border=True):
        st.image(slide_data.card_picture(slide.folder), width="stretch")
        st.markdown(f"**{slide.name}**")
        st.caption(f"Case {slide.case_id} · Entered {slide.entered}")
        st.html(status_badges(slide))
        open_col, menu_col = st.columns([0.72, 0.28])
        open_col.button("Open", key=f"open_{slide.key}", type="primary", width="stretch",
                        on_click=ui.go, args=({"name": "slide", "key": slide.key},))
        if menu_col.button("⋮", key=f"menu_{slide.key}", width="stretch",
                           help="Rename, duplicate, export, remove"):
            slide_menu(slide.key)


@st.dialog("Slide options", width="medium")
def slide_menu(key):
    """The card's quick-actions menu, in a pop-up window. st.rerun() closes it and refreshes the box."""
    slide = slide_data.get_slide(key)
    if slide is None:
        st.warning("This slide is no longer in the box.")
        return
    st.markdown(f"#### {slide.name}")

    st.markdown("**Rename**")
    name_col, save_col = st.columns([0.75, 0.25], vertical_alignment="bottom")
    new_name = name_col.text_input("New name", value=slide.name, key=f"rename_{key}")
    if save_col.button("Save name", key=f"save_{key}", width="stretch"):
        slide_data.rename(key, new_name)
        st.rerun()
    st.caption("Details to copy (select and press Ctrl+C, or use the copy icon):")
    st.code(f"{slide.name}\nCase: {slide.case_id}\nSlide: {slide.slide_id}\nEntered: {slide.entered}\n"
            f"Flags: {'; '.join(slide.flags) or 'none'}", language=None, wrap_lines=True)

    st.markdown("**Duplicate**")
    if st.button("Make a copy of this slide", key=f"dup_{key}", icon="📄"):
        slide_data.duplicate(key)
        st.rerun()

    st.markdown("**Export**")
    a, b = st.columns(2)
    # files are made only when a button is clicked (the lambda runs then)
    a.download_button("PDF report (1 page)", lambda: export.pdf_report(slide),
                      file_name=f"{slide.slide_id}_report.pdf", mime="application/pdf",
                      key=f"exp_pdf_{key}", width="stretch", on_click="ignore", icon="📄")
    b.download_button("Map picture (.png)", lambda: export.map_png(slide, "tissue"),
                      file_name=f"{slide.slide_id}_tissue_map.png", mime="image/png",
                      key=f"exp_png_{key}", width="stretch", on_click="ignore", icon="🖼️")
    a.download_button("Summary (.json)", lambda: json.dumps(slide_data.read_summary(slide.folder), indent=2),
                      file_name=f"{slide.slide_id}_summary.json", mime="application/json",
                      key=f"exp_json_{key}", width="stretch", on_click="ignore")
    b.download_button("Tile table (.csv)", lambda: (slide.folder / "tiles.csv").read_bytes(),
                      file_name=f"{slide.slide_id}_tiles.csv", mime="text/csv",
                      key=f"exp_csv_{key}", width="stretch", on_click="ignore")

    st.markdown("**Delete this copy**" if slide.is_copy else "**Remove from the Slide Box**")
    if not slide.is_copy:
        st.caption("The slide's files stay on disk; you can restore it from the bottom of the Slide Box.")
    sure = st.checkbox("Yes, I'm sure", key=f"confirm_{key}")
    if st.button("Delete copy" if slide.is_copy else "Remove", key=f"del_{key}", icon="🗑️", disabled=not sure):
        slide_data.remove(key)
        st.rerun()
