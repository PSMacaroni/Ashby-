"""
Tumor Front Analyzer: the app. Run from the project folder:

    streamlit run app/streamlit_app.py

Screens: the slide list (page_cases.py) -> one slide (page_slide.py). The look (light, clinical) is
set in app/.streamlit/config.toml and ui.py. The
app only reads result files made by the pipeline; it never runs the big models itself.
See app/README.md.
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
def warm_up_in_background():
    """Once per server, after the first screen is drawn: load the slide screens' libraries and
    data in the background, so opening a slide is quick."""
    threading.Thread(target=slide_data.warm_up, kwargs={"delay_s": 3}, daemon=True).start()
    return True


view = ui.current_view()
slide = slide_data.get_slide(view["key"]) if view["name"] == "slide" else None
if view["name"] == "slide" and slide is not None:
    import page_slide
    page_slide.show(slide)
else:
    ui.top_bar("Slides")
    if view["name"] == "slide":
        st.warning("That slide is no longer in the list.")
    import page_cases
    page_cases.show()

warm_up_in_background()
