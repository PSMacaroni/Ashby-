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
# Risk points: one hue, light -> dark (more points = darker). Index = points, capped at the end.
RISK_RAMP = ["#B7D3F6", "#86B6EF", "#5598E7", "#2A78D6", "#1C5CAB", "#104281"]
# Zones around the invasive front: ordered, so one hue from dark (core) to light (outside)
ZONES = {
    "core": ("Tumor core", "#B4471B"),
    "front_band": ("Invasive front band", "#E8753E"),
    "peritumoral": ("Peritumoral (0.5-2 mm)", "#F4B48F"),
    "outside": ("Further out", "#FBE3D3"),
}
# Certainty: only the doubtful tiles get a color, so they stand out
CERTAINTY = {"Sure": None, "Check": "#E8B341", "Unsure": "#E5534B"}
NOT_ANALYZED = "#8B929C"

HIGH_CONF = 0.80          # top score at or above this (and no flags) = "Sure"


def risk_color(points):
    return RISK_RAMP[min(int(points), len(RISK_RAMP)) - 1] if points else None


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
