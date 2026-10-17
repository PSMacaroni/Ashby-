"""
Shared settings for the app: folders, the 8 tissue classes, their colors and plain-English text.
Every other app file imports this first.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # so the team's shared code in src/ can be imported

from src.prompts import CLASSES  # noqa: E402  (class order = column order in tiles.csv)

DEMO_DIR = APP_DIR / "demo_data"              # fake slides made by make_demo_data.py
RESULTS_DIR = ROOT / "data" / "results"       # real slides copied down from Delta
BOX_STATE_FILE = APP_DIR / "slide_box.json"   # renames, copies, hidden slides (written by the app)

NAMES = {
    "tumor": "Tumor",
    "normal_mucosa": "Normal mucosa",
    "muscle": "Smooth muscle",
    "fat": "Fat",
    "lymphocytes": "Lymphocytes (immune cells)",
    "mucus": "Mucus",
    "necrosis": "Necrosis (dead tissue)",
    "stroma": "Stroma (connective tissue)",
}

SHORT_NAMES = {
    "tumor": "Tumor", "normal_mucosa": "Normal mucosa", "muscle": "Muscle", "fat": "Fat",
    "lymphocytes": "Lymphocytes", "mucus": "Mucus", "necrosis": "Necrosis", "stroma": "Stroma",
}

# Map overlay colors: picked to stand out on pink/purple H&E and from each other
COLORS = {
    "tumor": "#E6194B",
    "normal_mucosa": "#3CB44B",
    "muscle": "#F58231",
    "fat": "#FFE119",
    "lymphocytes": "#4363D8",
    "mucus": "#42D4F4",
    "necrosis": "#3A3A3A",
    "stroma": "#B15FD0",
}
NOT_ANALYZED_COLOR = "#9E9E9E"   # tiles that failed quality checks (blur, pen marks)

# What each class typically looks like on H&E, so a pathologist knows what to check by eye.
# (General histology, not something the model told us.)
LOOKS_LIKE = {
    "tumor": "crowded, irregular glands lined by tall cells with large, dark, stacked nuclei",
    "normal_mucosa": "evenly spaced, test-tube-shaped glands with goblet cells and small nuclei at the base",
    "muscle": "long pink bundles of spindle cells with cigar-shaped nuclei",
    "fat": "large, empty-looking round cells with thin walls",
    "lymphocytes": "sheets of small round cells with dark round nuclei and little cytoplasm",
    "mucus": "pale blue-grey, wispy material with few cells",
    "necrosis": "pink, structureless debris with broken nuclear fragments",
    "stroma": "pink fibrous tissue with scattered thin, elongated nuclei",
}

# Mix-ups we measured zero-shot on 50 NCT-CRC tiles per class (docs/handoff_small_classifier.md)
KNOWN_MIXUPS = {
    frozenset({"stroma", "muscle"}): "Stroma vs muscle is the most common mix-up in our tests "
                                     "(CONCH called 11 of 50 stroma tiles muscle; MUSK called 6 of 50 "
                                     "muscle tiles stroma).",
    frozenset({"tumor", "normal_mucosa"}): "Tumor vs normal mucosa is a known mix-up "
                                           "(CONCH called 7 of 50 tumor tiles normal mucosa).",
    frozenset({"normal_mucosa", "mucus"}): "Normal mucosa vs mucus is a known mix-up "
                                           "(MUSK called 8 of 50 normal mucosa tiles mucus).",
    frozenset({"muscle", "fat"}): "Muscle vs fat is a known mix-up (CONCH called 5 of 50 muscle tiles fat).",
    frozenset({"fat", "mucus"}): "Fat vs mucus is a known mix-up (MUSK called 4 of 50 fat tiles mucus).",
    frozenset({"stroma", "necrosis"}): "Stroma vs necrosis is a known mix-up (CONCH called 3 of 50 stroma "
                                       "tiles necrosis).",
    frozenset({"mucus", "necrosis"}): "Mucus vs necrosis is easy to confuse where debris sits inside glands.",
}

# Confidence bands used everywhere in the app
HIGH_CONF = 0.80
LOW_CONF = 0.60

# Risk levels (points from map features near each tile; see app/README.md)
RISK_LEVELS = {
    1: ("Low", "#F9D84A"),
    2: ("Moderate", "#F8961E"),
    3: ("High", "#E63946"),
    4: ("Very high", "#7B0828"),
}

CAVEATS = [
    "Decision support only, not a diagnosis.",
    "One slide is a 2D snapshot: 'direction' means where the tumor is invading in this slice, "
    "not where it will grow next.",
    "T stage needs more than one slide; ours is a 'consistent with' hint.",
    "H&E can't tell CD4 from CD8 T cells; we only measure where lymphocytes are.",
    "The map gives one answer per square (many cells across), so single cells are invisible to it.",
]


def pct(x, digits=0):
    """0.1234 -> '12%'. Returns 'n/a' for missing values."""
    if x is None:
        return "n/a"
    return f"{100 * float(x):.{digits}f}%"
