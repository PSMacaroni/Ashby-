"""
Shared settings for the app: folders, the 8 tissue classes and their names, and every color the
app draws itself. Every other app file imports this first.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # so the team's shared code in src/ can be imported

from src.prompts import CLASSES  # noqa: E402  (class order = column order in tiles.csv)
from src.viz import CLASS_COLORS  # noqa: E402  (one color per class everywhere in the project)

DEMO_DIR = APP_DIR / "demo_data"              # fake slides made by make_demo_data.py
RESULTS_DIR = ROOT / "data" / "results"       # real slides copied down from Delta
BOX_STATE_FILE = APP_DIR / "slide_box.json"   # renames, copies, hidden slides (written by the app)

# Same values as .streamlit/config.toml, for the HTML, map and exports we draw ourselves
THEME = {
    "bg": "#0B0D10", "surface": "#14171C", "surface2": "#1A1E24", "border": "#262B33",
    "text": "#E6E8EB", "text2": "#9AA3AF", "muted": "#7D8590",
    "accent": "#4C9AFF", "button": "#1F6FEB",
    "mono": "Cascadia Mono, SFMono-Regular, Consolas, Menlo, monospace",
}

NAMES = {
    "tumor": "Tumor", "normal_mucosa": "Normal mucosa", "muscle": "Smooth muscle", "fat": "Fat",
    "lymphocytes": "Lymphocytes", "mucus": "Mucus", "necrosis": "Necrosis", "stroma": "Stroma",
}

# What each class typically looks like on H&E, so a pathologist knows what to check by eye.
LOOKS_LIKE = {
    "tumor": "crowded, irregular glands with large, dark, stacked nuclei",
    "normal_mucosa": "evenly spaced test-tube glands with goblet cells",
    "muscle": "long pink bundles of spindle cells with cigar-shaped nuclei",
    "fat": "large, empty-looking round cells with thin walls",
    "lymphocytes": "sheets of small round cells with dark nuclei",
    "mucus": "pale blue-grey, wispy material with few cells",
    "necrosis": "pink, structureless debris with nuclear fragments",
    "stroma": "pink fibrous tissue with thin, elongated nuclei",
}

# Mix-ups we measured zero-shot on 50 NCT-CRC tiles per class (docs/handoff_small_classifier.md)
KNOWN_MIXUPS = {
    frozenset({"stroma", "muscle"}): "Stroma vs muscle is the most common mix-up in our tests.",
    frozenset({"tumor", "normal_mucosa"}): "Tumor vs normal mucosa is a known mix-up (CONCH, 7 of 50 tumor tiles).",
    frozenset({"normal_mucosa", "mucus"}): "Normal mucosa vs mucus is a known mix-up (MUSK, 8 of 50).",
    frozenset({"muscle", "fat"}): "Muscle vs fat is a known mix-up (CONCH, 5 of 50 muscle tiles).",
    frozenset({"mucus", "necrosis"}): "Mucus vs necrosis is easy to confuse where debris sits in glands.",
}

# ---- map overlay colors (beyond the tissue classes) -----------------------------------------
# Same colors as the team's analysis figure (src/viz.py overview_figure), so the live map and the
# figure read the same way.
# Risk points: matplotlib "Reds" (white -> dark red), scaled 0..max(4, highest points) like the figure
REDS = ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"]
# Search match: one blue scale (more = darker), so it never looks like risk
RISK_RAMP = ["#B7D3F6", "#86B6EF", "#5598E7", "#2A78D6", "#1C5CAB", "#104281"]
# Zones around the invasive front (figure: core dark red, front band orange, peritumoral peach,
# other tissue light gray, the front itself black)
ZONES = {
    "core": ("Tumor core", "#8C0000"),
    "front_band": ("Invasive front band (±0.5 mm)", "#FF734D"),
    "peritumoral": ("Peritumoral (0.5-2 mm)", "#FFD9B3"),
    "outside": ("Other tissue", "#D9D9D9"),
}
FRONT_COLOR = "#000000"
GROWTH_COLOR = "#1F5BFF"         # the figure's blue arrows / circles
# Flag markers: (plotly symbol, color), from src/viz.py FLAG_MARKERS
FLAG_MARKERS = {"perineural": ("triangle-up", "#000000"), "venous": ("square", "#6A3D9A"),
                "lymphatic": ("diamond", "#FF7F00"), "budding": ("star", "#B15928"),
                "stroma_maturity": ("x", "#666666"), "adenoma": ("cross", "#33A02C")}
# Certainty: only the doubtful tiles get a color, so they stand out
CERTAINTY = {"Sure": None, "Check": "#E8B341", "Unsure": "#E5534B"}
NOT_ANALYZED = "#8B929C"

HIGH_CONF = 0.80          # top score at or above this (and no flags) = "Sure"


def risk_color(points, top=4):
    """Reds color for a number of risk points, on a 0..max(4, top) scale (like the team's figure)."""
    if not points:
        return None
    frac = min(1.0, float(points) / max(4, top))
    return REDS[min(len(REDS) - 1, int(round(frac * (len(REDS) - 1))))]


def pct(x, digits=0):
    """0.1234 -> '12%'. 'n/a' for missing values."""
    if x is None:
        return "n/a"
    return f"{100 * float(x):.{digits}f}%"


def get(d, path, default=None):
    """get(summary, 'invasion.direction.text') -> value, or default if any part is missing."""
    for part in path.split("."):
        if not isinstance(d, dict) or d.get(part) is None:
            return default
        d = d[part]
    return d
