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

from src.prompts import CLASSES  # noqa: E402,F401  (class order = column order in tiles.csv)
from src.viz import CLASS_COLORS  # noqa: E402,F401  (one color per class everywhere in the project)

try:
    from src.cutoffs import LOW_CONFIDENCE  # noqa: E402  ("unsure here" = top score below this)
except Exception:                            # pragma: no cover - older src without the cut-off
    LOW_CONFIDENCE = 0.9

# Where slides come from: every folder with summary.json + tiles.csv in one of these.
# showcase/results/ is in git (the 5 showcase slides); data/results/ is git-ignored (copied from Delta).
RESULTS_DIRS = [ROOT / "showcase" / "results", ROOT / "data" / "results"]
STATE_FILE = APP_DIR / "state.json"     # renames, hidden slides, copies, notes, reviewed items (written by the app)

# Same values as .streamlit/config.toml, for the HTML, map and exports we draw ourselves
THEME = {
    "bg": "#FFFFFF", "surface": "#F4F6F8", "surface2": "#EAEEF2", "border": "#D9DEE4",
    "text": "#1B1F24", "text2": "#424A55", "muted": "#6B7480",
    "accent": "#1F5F8B", "button": "#1F5F8B",
    "red": "#B42318", "red_bg": "#FEF3F2", "amber": "#B54708", "amber_bg": "#FFFAEB",
    "green": "#1E7A4A", "green_bg": "#ECFDF3", "blue_bg": "#EAF2F9",
    "mono": "Cascadia Mono, SFMono-Regular, Consolas, Menlo, monospace",
}

NAMES = {
    "tumor": "Tumor", "normal_mucosa": "Normal mucosa", "muscle": "Smooth muscle", "fat": "Fat",
    "lymphocytes": "Lymphocytes", "mucus": "Mucus", "necrosis": "Necrosis", "stroma": "Stroma",
}
SITES = {"TCGA-COAD": "Colon", "TCGA-READ": "Rectum"}

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
MATCH_RAMP = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"]
# Zones around the invasive front (figure: core dark red, front band orange, peritumoral peach,
# other tissue light gray, the front itself black)
ZONES = {
    "core": ("Tumor core", "#8C0000"),
    "front_band": ("Invasive front band (±0.5 mm)", "#FF734D"),
    "peritumoral": ("Peritumoral (0.5-2 mm)", "#FFD9B3"),
    "outside": ("Other tissue", "#D9D9D9"),
}
ZONE_SHORT = {"core": "Tumor core", "front_band": "Front band", "peritumoral": "Peritumoral", "outside": "Other tissue",
              "": "—"}
FRONT_COLOR = "#000000"
GROWTH_COLOR = "#1F5BFF"         # the figure's blue arrows / circles
# Flag markers: (plotly symbol, color), from src/viz.py FLAG_MARKERS
FLAG_MARKERS = {"perineural": ("triangle-up", "#000000"), "venous": ("square", "#6A3D9A"),
                "lymphatic": ("diamond", "#FF7F00"), "budding": ("star", "#B15928"),
                "stroma_maturity": ("x", "#666666"), "adenoma": ("cross", "#33A02C")}
# Certainty: only the doubtful tiles get a color, so they stand out
UNSURE_COLOR = "#E8A33D"
NOT_ANALYZED = "#9AA1AA"
# Pointer colors per worklist group
POINTER_COLORS = {"flags": "#B42318", "risk": "#7A1C1C", "unsure": "#B54708", "mucin": "#0E7490",
                  "islands": "#1F5BFF", "nodes": "#5B21B6", "search": "#1F5F8B", "other": "#1F5F8B"}


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


def mm(um):
    """Micrometers -> 'x.x mm' (or 'n/a')."""
    if um is None:
        return "n/a"
    return f"{float(um) / 1000:.1f} mm"


def get(d, path, default=None):
    """get(summary, 'invasion.direction.text') -> value, or default if any part is missing."""
    for part in path.split("."):
        if not isinstance(d, dict) or d.get(part) is None:
            return default
        d = d[part]
    return d


def site_name(summary):
    return SITES.get(summary.get("project") or "", summary.get("project") or "")


def stage_text(summary):
    """'pT4a N2b · Stage IVA' from what the pipeline recorded (TCGA report), or ''."""
    parts = []
    if summary.get("pT"):
        parts.append(f"p{summary['pT']}" if not str(summary["pT"]).startswith("p") else summary["pT"])
    if summary.get("pN"):
        parts.append(str(summary["pN"]))
    text = " ".join(parts)
    if summary.get("stage"):
        text = f"{text} · {summary['stage']}" if text else str(summary["stage"])
    return text
