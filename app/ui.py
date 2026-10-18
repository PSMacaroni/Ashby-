"""
Shared look and pieces for every screen: page style, the top bar with back/forward, and small
HTML building blocks (chips, rows, list items) so all screens look the same.

Navigation works like a browser's history: each screen we open is added to a list; the arrows
move back and forward through it.
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
  /* page frame: no Streamlit decoration, centered content */
  [data-testid="stDecoration"] {{ display: none; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ max-width: 1380px; padding-top: 1.1rem !important; padding-bottom: 2.5rem; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.01em; font-weight: 650 !important; }}
  hr {{ margin: 0.6rem 0 !important; border-color: {T['border']} !important; }}
  [data-testid="stCaptionContainer"] {{ color: {T['muted']}; }}

  /* top bar */
  .tb {{ display: flex; align-items: center; gap: 12px; }}
  .tb img {{ width: 34px; height: 34px; }}
  .tb .name {{ font-weight: 650; font-size: 1.02rem; color: {T['text']}; line-height: 1.15; }}
  .tb .crumb {{ font-size: 0.82rem; color: {T['muted']}; }}
  .st-key-topbar {{ border-bottom: 1px solid {T['border']}; padding-bottom: 10px; margin-bottom: 6px; }}
  .st-key-topbar button {{ min-height: 34px; padding: 0 10px; }}

  /* chips, rows, list items */
  .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 1px 9px; border-radius: 999px;
          font-size: 0.75rem; font-weight: 600; border: 1px solid {T['border']}; color: {T['text2']};
          background: {T['surface2']}; margin: 0 4px 4px 0; white-space: nowrap; }}
  .chip i {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
  .chip.blue {{ color: #CFE3FF; border-color: #24467A; background: #12213A; }}
  .chip.red {{ color: #FFD7D5; border-color: #6B2A27; background: #2A1414; }}
  .chip.amber {{ color: #FBE7B5; border-color: #6A5320; background: #251E0D; }}
  .rows {{ border-top: 1px solid {T['border']}; }}
  .row {{ display: grid; grid-template-columns: 112px 1fr; gap: 12px; padding: 9px 2px;
          border-bottom: 1px solid {T['border']}; font-size: 0.9rem; line-height: 1.45; }}
  .row .k {{ color: {T['muted']}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
             padding-top: 2px; }}
  .row .v {{ color: {T['text']}; }}
  .item {{ display: flex; gap: 10px; padding: 9px 2px; border-bottom: 1px solid {T['border']};
           font-size: 0.9rem; line-height: 1.45; }}
  .item .dot {{ flex: 0 0 8px; height: 8px; border-radius: 50%; margin-top: 7px; }}
  .item .t {{ color: {T['text']}; font-weight: 600; }}
  .item .s {{ color: {T['text2']}; font-size: 0.84rem; }}
  .sec {{ font-size: 0.74rem; font-weight: 700; color: {T['muted']}; text-transform: uppercase;
          letter-spacing: 0.08em; margin: 14px 0 4px; }}
  .lead {{ font-size: 0.95rem; line-height: 1.55; color: {T['text']}; }}
  .note {{ font-size: 0.8rem; color: {T['muted']}; line-height: 1.45; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.8rem; color: {T['text2']}; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
  .banner {{ font-size: 0.8rem; color: #FBE7B5; background: #251E0D; border: 1px solid #6A5320;
             border-radius: 8px; padding: 6px 10px; }}
  .stat b {{ color: {T['text']}; font-weight: 650; }}
  .stat {{ color: {T['text2']}; font-size: 0.92rem; margin-right: 18px; }}

  /* buttons that act like links ("Show on map") */
  button[kind="tertiary"] {{ color: {T['accent']} !important; padding: 0 !important; min-height: 26px !important; }}
  button[kind="tertiary"]:hover {{ text-decoration: underline; }}

  /* map + panel cards on the slide screen */
  .st-key-mapcard, .st-key-panelcard {{ background: {T['surface']}; border: 1px solid {T['border']};
          border-radius: 12px; padding: 14px 16px 12px; }}
  .st-key-mapcard [data-testid="stPlotlyChart"] {{ border-radius: 8px; overflow: hidden; }}
</style>
"""


def inject_css():
    st.html(CSS)


# ---------------------------------------------------------------- navigation history

def _history():
    if "history" not in st.session_state:
        st.session_state.history = [{"name": "box"}]
        st.session_state.hist_i = 0
    return st.session_state.history


def current_view():
    return _history()[st.session_state.hist_i]


def go(view):
    """Open a new screen (drops any 'forward' history, like a browser)."""
    hist = _history()[: st.session_state.hist_i + 1]
    if hist[-1] != view:
        hist.append(view)
    st.session_state.history = hist
    st.session_state.hist_i = len(hist) - 1


def back():
    st.session_state.hist_i = max(0, st.session_state.hist_i - 1)


def forward():
    st.session_state.hist_i = min(len(_history()) - 1, st.session_state.hist_i + 1)


def top_bar(crumb):
    with st.container(key="topbar", horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="tb"><img src="{LOGO_URI}" alt="">'
                f'<div><div class="name">Tumor Front Analyzer</div><div class="crumb">{html.escape(crumb)}</div></div></div>')
        st.space("stretch")
        i, n = st.session_state.hist_i, len(_history())
        st.button("", icon=":material/arrow_back:", key="nav_back", help="Back", on_click=back, disabled=i == 0)
        st.button("", icon=":material/arrow_forward:", key="nav_forward", help="Forward", on_click=forward,
                  disabled=i >= n - 1)


# ---------------------------------------------------------------- HTML pieces (escape inputs!)

def esc(text):
    return html.escape(str(text))


def chip(text, tone="", dot=None):
    d = f'<i style="background:{dot}"></i>' if dot else ""
    return f'<span class="chip {tone}">{d}{esc(text)}</span>'


def rows(pairs):
    """Label/sentence rows, like a clean settings list."""
    body = "".join(f'<div class="row"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                   for k, v in pairs)
    st.html(f'<div class="rows">{body}</div>')


def item(title, sub="", color=None, title_html=None):
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ""
    t = title_html if title_html is not None else esc(title)
    s = f'<div class="s">{esc(sub)}</div>' if sub else ""
    st.html(f'<div class="item">{dot}<div><div class="t">{t}</div>{s}</div></div>')


def section(text):
    st.html(f'<div class="sec">{esc(text)}</div>')


def legend(items):
    parts = "".join(f'<span><i style="background:{c}"></i>{esc(t)}</span>' for t, c in items)
    st.html(f'<div class="legend">{parts}</div>')


def note(text):
    st.html(f'<div class="note">{esc(text)}</div>')
