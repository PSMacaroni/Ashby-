"""
Text descriptions ("prompts") for each tissue class, used for zero-shot classification.

Each class gets several names x several sentence templates. The model's embeddings for all
of a class's prompts are averaged ("prompt ensemble"), which is usually more reliable than
any single phrasing. If you change these, bump PROMPTS_VERSION so results stay traceable.
"""

PROMPTS_VERSION = "v1"

# Order matters: it is the column order of the probabilities (p_tumor, p_normal_mucosa, ...).
CLASSES = [
    "tumor",
    "normal_mucosa",
    "muscle",
    "fat",
    "lymphocytes",
    "mucus",
    "necrosis",
    "stroma",
]

CLASS_NAMES = {
    "tumor": ["colorectal adenocarcinoma", "adenocarcinoma epithelium", "invasive carcinoma",
              "colon cancer tumor epithelium"],
    "normal_mucosa": ["normal colon mucosa", "normal colonic glands", "benign colonic epithelium"],
    "muscle": ["smooth muscle", "muscularis propria", "smooth muscle bundles"],
    "fat": ["adipose tissue", "adipocytes", "fat cells"],
    "lymphocytes": ["lymphocytes", "dense lymphocytic infiltrate", "lymphoid aggregate"],
    "mucus": ["mucus", "extracellular mucin", "pools of mucin"],
    "necrosis": ["necrosis", "necrotic debris", "tumor necrosis"],
    "stroma": ["fibrous stroma", "desmoplastic stroma", "cancer-associated stroma",
               "connective tissue"],
}

TEMPLATES = [
    "{}.",
    "an H&E image of {}.",
    "a photomicrograph showing {}.",
    "an image of {}.",
    "{} in an H&E stained section.",
]


def build_prompts():
    """Returns {class: [prompt, prompt, ...]} for every class in CLASSES."""
    return {c: [t.format(name) for name in CLASS_NAMES[c] for t in TEMPLATES] for c in CLASSES}
