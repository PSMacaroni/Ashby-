"""
Small pieces of UI shared by every screen: page style, the top bar with the logo and the
back/forward arrows, and colored badges.

Navigation works like a web browser's history: every screen we open is added to a list, and
the arrows move back and forward through it.
"""
import base64
import html

import streamlit as st

from common import APP_DIR, RISK_LEVELS

LOGO_SVG = (APP_DIR / "assets" / "logo.svg").read_text(encoding="utf-8")
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode()).decode()

CSS = """
<style>
  .block-container { padding-top: 3.6rem !important; padding-bottom: 2rem; }
  .ctfa-title { font-size: 1.45rem; font-weight: 700; line-height: 1.2; margin: 0; }
  .ctfa-crumb { font-size: 0.9rem; opacity: 0.75; margin-top: 2px; }
  .ctfa-badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.78rem;
                font-weight: 600; margin: 0 4px 4px 0; border: 1px solid transparent; white-space: nowrap; }
  .ctfa-flag { border-left: 4px solid #E63946; padding: 6px 10px; margin: 0 0 6px 0; border-radius: 4px;
               background: rgba(230, 57, 70, 0.08); font-size: 0.92rem; }
  .ctfa-card { border-left: 5px solid #999; padding: 8px 12px; margin-bottom: 8px; border-radius: 6px;
               background: rgba(128, 128, 128, 0.07); }
  .ctfa-card h5 { margin: 0 0 4px 0; font-size: 0.98rem; }
  .ctfa-legend span.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
                         margin: 0 5px -1px 0; border: 1px solid rgba(0,0,0,0.25); }
  .ctfa-legend { font-size: 0.82rem; line-height: 1.9; }
  .ctfa-legend .item { display: inline-block; margin-right: 12px; white-space: nowrap; }
  .ctfa-demo { background: rgba(255, 193, 7, 0.16); border: 1px solid rgba(255, 193, 7, 0.6);
               border-radius: 6px; padding: 4px 10px; font-size: 0.85rem; margin-bottom: 8px; }
  .ctfa-big { font-size: 1.25rem; font-weight: 700; }
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


def top_bar(crumbs):
    """Logo, app name, breadcrumb (e.g. ['Slide Box', 'Demo 1']) and the back/forward arrows."""
    logo, title, nav = st.columns([0.06, 0.74, 0.20], vertical_alignment="center")
    logo.html(f'<img src="{LOGO_URI}" width="58" alt="Slide box logo">')
    crumb = " &rsaquo; ".join(html.escape(c) for c in crumbs)
    title.html(f'<p class="ctfa-title">Colon Tumor Front Analyzer</p><div class="ctfa-crumb">{crumb}</div>')
    b, f = nav.columns(2)
    i, n = st.session_state.hist_i, len(_history())
    b.button("←", key="nav_back", help="Back to the previous screen", on_click=back,
             disabled=i == 0, width="stretch")
    f.button("→", key="nav_forward", help="Forward again (after going back)", on_click=forward,
             disabled=i >= n - 1, width="stretch")


# ---------------------------------------------------------------- badges and cards

def badge(text, color="#6c757d", filled=False):
    """A small rounded label. Colors are hex like '#E63946'."""
    if filled:
        style = f"background:{color};color:#fff;"
    else:
        style = f"background:{color}22;border-color:{color}88;"
    return f'<span class="ctfa-badge" style="{style}">{html.escape(text)}</span>'


def risk_badge(level):
    if not level:
        return badge("No risk points", "#6c757d")
    name, color = RISK_LEVELS[level]
    return badge(f"Level {level} · {name}", color, filled=level >= 3)


def sure_badge(level):
    color = {"Sure": "#2b83ba", "Somewhat sure": "#E69F00", "Unsure": "#D55E00"}.get(level, "#6c757d")
    return badge(level, color, filled=level == "Unsure")


def flag(text):
    st.html(f'<div class="ctfa-flag">{html.escape(text)}</div>')


def card(title, body_html, color="#999"):
    """A box with a colored left edge. body_html must already be escaped."""
    st.html(f'<div class="ctfa-card" style="border-left-color:{color}"><h5>{html.escape(title)}</h5>'
            f'{body_html}</div>')


def legend(items):
    """items: list of (label, hex color)."""
    parts = [f'<span class="item"><span class="sw" style="background:{c}"></span>{html.escape(t)}</span>'
             for t, c in items]
    st.html(f'<div class="ctfa-legend">{"".join(parts)}</div>')


def demo_banner():
    st.html('<div class="ctfa-demo"><b>DEMO slide:</b> fake data made to build the app. '
            'Every number here is made up.</div>')
