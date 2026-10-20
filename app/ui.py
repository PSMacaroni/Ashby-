"""
Shared look and pieces for every screen: page style (light, clinical), the top bar with the
navigation, and small HTML building blocks (chips, rows, list items) so all screens look the same.

Screens: "cases" (the slide list) and "slide" (one slide).
"""
import base64
import html

import streamlit as st

from common import APP_DIR, THEME

LOGO_SVG = (APP_DIR / "assets" / "logo.svg").read_text(encoding="utf-8")
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode()).decode()
T = THEME

CSS = f"""
<style>
  /* no grey fade while a screen updates: the old map stays until the new one replaces it */
  [data-stale="true"], .stale-element, .stElementContainer[data-stale="true"] > * {{ opacity: 1 !important; transition: none !important; }}

  /* page frame: no Streamlit decoration, wide content */
  [data-testid="stDecoration"] {{ display: none; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ max-width: 1480px; padding-top: 0.9rem !important; padding-bottom: 2rem; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.01em; font-weight: 650 !important; }}
  hr {{ margin: 0.6rem 0 !important; border-color: {T['border']} !important; }}
  [data-testid="stCaptionContainer"] {{ color: {T['muted']}; }}

  /* top bar */
  .tb {{ display: flex; align-items: center; gap: 12px; }}
  .tb img {{ width: 34px; height: 34px; }}
  .tb .name {{ font-weight: 650; font-size: 1.02rem; color: {T['text']}; line-height: 1.15; }}
  .tb .crumb {{ font-size: 0.82rem; color: {T['muted']}; }}
  .st-key-topbar {{ border-bottom: 1px solid {T['border']}; padding-bottom: 8px; margin-bottom: 8px; }}
  .st-key-topbar button {{ min-height: 34px; padding: 0 10px; }}

  /* chips, rows, list items */
  .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 1px 9px; border-radius: 999px;
          font-size: 0.76rem; font-weight: 600; border: 1px solid {T['border']}; color: {T['text2']};
          background: {T['surface']}; margin: 0 4px 4px 0; white-space: nowrap; }}
  .chip i {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
  .chip.blue {{ color: #0F3D5E; border-color: #B7D1E6; background: {T['blue_bg']}; }}
  .chip.red {{ color: {T['red']}; border-color: #F2B8B5; background: {T['red_bg']}; }}
  .chip.amber {{ color: {T['amber']}; border-color: #F5D9A8; background: {T['amber_bg']}; }}
  .chip.green {{ color: {T['green']}; border-color: #ABEFC6; background: {T['green_bg']}; }}
  .rows {{ border-top: 1px solid {T['border']}; }}
  .row {{ display: grid; grid-template-columns: 128px 1fr; gap: 12px; padding: 7px 2px;
          border-bottom: 1px solid {T['border']}; font-size: 0.9rem; line-height: 1.45; }}
  .row .k {{ color: {T['muted']}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
             padding-top: 2px; }}
  .row .v {{ color: {T['text']}; }}
  .row .v.strong {{ font-weight: 600; }}
  .item {{ display: flex; gap: 10px; padding: 8px 2px; border-bottom: 1px solid {T['border']};
           font-size: 0.9rem; line-height: 1.45; }}
  .item .dot {{ flex: 0 0 9px; height: 9px; border-radius: 50%; margin-top: 7px; }}
  .item .t {{ color: {T['text']}; font-weight: 500; }}
  .item .s {{ color: {T['text2']}; font-size: 0.84rem; }}
  .item.done .t, .item.done .s {{ color: {T['muted']}; text-decoration: line-through; }}
  .sec {{ font-size: 0.74rem; font-weight: 700; color: {T['muted']}; text-transform: uppercase;
          letter-spacing: 0.08em; margin: 14px 0 4px; }}
  .lead {{ font-size: 0.95rem; line-height: 1.55; color: {T['text']}; }}
  .note {{ font-size: 0.8rem; color: {T['muted']}; line-height: 1.45; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.8rem; color: {T['text2']}; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; border: 1px solid rgba(0,0,0,.12); }}
  .banner {{ font-size: 0.82rem; color: {T['amber']}; background: {T['amber_bg']}; border: 1px solid #F5D9A8;
             border-radius: 8px; padding: 6px 10px; }}
  .banner.red {{ color: {T['red']}; background: {T['red_bg']}; border-color: #F2B8B5; }}
  .banner.blue {{ color: #0F3D5E; background: {T['blue_bg']}; border-color: #B7D1E6; }}

  /* the summary strip at the top of a slide */
  .strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin: 4px 0 10px; }}
  .strip div {{ background: {T['surface']}; border: 1px solid {T['border']}; border-radius: 8px; padding: 8px 12px; }}
  .strip small {{ display: block; color: {T['muted']}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; }}
  .strip b {{ display: block; font-size: 1.02rem; font-weight: 650; color: {T['text']}; margin-top: 2px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .strip b.red {{ color: {T['red']}; }}
  .strip b.amber {{ color: {T['amber']}; }}
  .strip b.green {{ color: {T['green']}; }}

  /* text boxes: a clearer edge on the white page */
  [data-baseweb="textarea"], [data-baseweb="input"] {{ border: 1px solid #C5CCD4 !important; }}

  /* buttons that act like links ("Show on map") */
  button[kind="tertiary"] {{ color: {T['accent']} !important; padding: 0 !important; min-height: 24px !important; }}
  button[kind="tertiary"]:hover {{ text-decoration: underline; }}

  /* the map's hover box must never catch the mouse: a click goes to the square under it */
  .st-key-mapcard .hoverlayer, .st-key-mapcard .hoverlayer * {{ pointer-events: none !important; }}

  /* map + panel cards on the slide screen */
  .st-key-mapcard, .st-key-panelcard {{ background: #FFFFFF; border: 1px solid {T['border']};
          border-radius: 10px; padding: 12px 14px 10px; }}
  .st-key-mapcard [data-testid="stPlotlyChart"] {{ border-radius: 6px; overflow: hidden; border: 1px solid {T['border']}; }}
  .st-key-mapcard [data-testid="stSliderThumbValue"], .st-key-mapcard [data-testid="stSliderTickBar"] {{ display: none; }}

  /* phones */
  @media (max-width: 640px) {{
    .tb .crumb {{ display: none; }}
    .tb img {{ width: 28px; height: 28px; }}
    .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
  }}
</style>
"""


def inject_css():
    st.html(CSS)


# ---------------------------------------------------------------- navigation

def current_view():
    return st.session_state.get("view", {"name": "cases"})


def go(view):
    st.session_state["view"] = view


def top_bar(crumb, extra=None):
    """The bar on every screen. extra: a function drawing widgets in the middle (e.g. a slide picker)."""
    with st.container(key="topbar", horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="tb"><img src="{LOGO_URI}" alt="">'
                f'<div><div class="name">Tumor Front Analyzer</div><div class="crumb">{html.escape(crumb)}</div></div></div>')
        st.space("stretch")
        if extra:
            extra()
        st.button("All slides", icon=":material/list:", key="nav_cases", type="tertiary",
                  on_click=go, args=({"name": "cases"},), disabled=current_view()["name"] == "cases")


# ---------------------------------------------------------------- HTML pieces (escape inputs!)

def esc(text):
    return html.escape(str(text))


def chip(text, tone="", dot=None):
    d = f'<i style="background:{dot}"></i>' if dot else ""
    return f'<span class="chip {tone}">{d}{esc(text)}</span>'


def rows(pairs, tooltips=False):
    """Label/value rows, like a clean settings list. pairs: (label, value) or (label, value, strong);
    with tooltips=True the third element is the full sentence shown on hover."""
    body = ""
    for p in pairs:
        k, v = p[0], p[1]
        extra = p[2] if len(p) > 2 else None
        strong = " strong" if (extra and not tooltips) else ""
        title = f' title="{esc(extra)}"' if (tooltips and extra) else ""
        body += f'<div class="row"{title}><div class="k">{esc(k)}</div><div class="v{strong}">{esc(v)}</div></div>'
    st.html(f'<div class="rows">{body}</div>')


def item(title, sub="", color=None, title_html=None, done=False):
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ""
    t = title_html if title_html is not None else esc(title)
    s = f'<div class="s">{esc(sub)}</div>' if sub else ""
    st.html(f'<div class="item{" done" if done else ""}">{dot}<div><div class="t">{t}</div>{s}</div></div>')


def section(text):
    st.html(f'<div class="sec">{esc(text)}</div>')


def legend(items):
    parts = "".join(f'<span><i style="background:{c}"></i>{esc(t)}</span>' for t, c in items)
    st.html(f'<div class="legend">{parts}</div>')


def note(text):
    st.html(f'<div class="note">{esc(text)}</div>')


def banner(text, tone=""):
    st.html(f'<div class="banner {tone}">{esc(text)}</div>')


def strip(cells):
    """The summary strip: [(label, value, tone)]."""
    body = "".join(f'<div><small>{esc(k)}</small><b class="{tone}">{esc(v)}</b></div>' for k, v, tone in cells)
    st.html(f'<div class="strip">{body}</div>')
