"""
Colon Tumor Front Analyzer: the app. Run from the project folder:

    streamlit run app/streamlit_app.py

Screens: Level 1 Slide Box (view_box.py) -> Level 2 slide explorer (view_slide.py).
The app only reads result files made on Delta; it never runs the big models itself.
See app/README.md for how the pieces fit together.
"""
import streamlit as st

import common  # noqa: F401  (first: sets up folders and imports)
import make_demo_data
import slide_data
import ui
import view_box
import view_slide

st.set_page_config(page_title="Colon Tumor Front Analyzer", page_icon="🔬", layout="wide")
ui.inject_css()

if "demo_ready" not in st.session_state:
    with st.spinner("Making the demo slides (first launch only, about a minute)..."):
        make_demo_data.ensure_demo_data()
    st.session_state.demo_ready = True

view = ui.current_view()
if view["name"] == "slide":
    slide = slide_data.get_slide(view["key"])
    if slide is None:
        ui.top_bar(["Slide Box", "missing slide"])
        st.warning("This slide is no longer in the Slide Box.")
        st.button("Back to the Slide Box", on_click=ui.go, args=({"name": "box"},))
    else:
        ui.top_bar(["Slide Box", slide.name])
        view_slide.show(slide)
else:
    ui.top_bar(["Slide Box"])
    view_box.show()
