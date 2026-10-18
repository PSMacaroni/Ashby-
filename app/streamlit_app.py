"""
Tumor Front Analyzer: the app. Run from the project folder:

    streamlit run app/streamlit_app.py

Screens: Level 1 Slide Box (view_box.py) -> Level 2 slide (view_slide.py). The look (dark, one blue
accent) is set in app/.streamlit/config.toml and ui.py. The app only reads result files made by the
pipeline; it never runs the big models itself. See app/README.md.
"""
import threading

import streamlit as st

import common  # noqa: F401  (first: sets up folders and imports)
import slide_data
import ui

st.set_page_config(page_title="Tumor Front Analyzer", page_icon=str(common.APP_DIR / "assets" / "icon.png"),
                   layout="wide")
ui.inject_css()


@st.cache_resource(show_spinner=False)
def make_demo_slides():
    """Once per server: make the demo slides if they aren't there yet."""
    from make_demo_data import ensure_demo_data
    ensure_demo_data()
    return True


@st.cache_resource(show_spinner=False)
def warm_up_in_background():
    """Once per server, after the first screen is drawn: load the slide screens' libraries and
    data in the background, so opening a slide is quick."""
    threading.Thread(target=slide_data.warm_up, kwargs={"delay_s": 3}, daemon=True).start()
    return True


with st.spinner("Getting the slides ready (first start only)..."):
    make_demo_slides()

view = ui.current_view()
slide = slide_data.get_slide(view["key"]) if view["name"] == "slide" else None
if view["name"] == "results":
    ui.top_bar("Model results")
    import view_results
    view_results.show()
elif view["name"] == "slide" and slide is not None:
    ui.top_bar(f"Slide Box  ›  {slide.name}")
    import view_slide
    view_slide.show(slide)
else:
    ui.top_bar("Slide Box")
    if view["name"] == "slide":
        st.warning("That slide is no longer in the Slide Box.")
    import view_box
    view_box.show()

warm_up_in_background()
