"""
Readers for the datasets preloaded on Delta (under DATA_DIR = /projects/bfqi/shared/datasets).

- NCT-CRC-HE-100K: 100,000 labeled 224 x 224 px colon tiles (0.5 microns/px), one folder per class.
- PathGen 2k sample: ~2,100 patches cut from TCGA slides, as <file_id>_<x>_<y>.png + .txt caption.
"""
from pathlib import Path

import numpy as np
from PIL import Image

# NCT-CRC folder name -> our class name (see src/prompts.py CLASSES).
# BACK (background glass) has no class of ours: our tissue mask removes background first.
NCT_TO_CLASS = {
    "TUM": "tumor",
    "NORM": "normal_mucosa",
    "MUS": "muscle",
    "ADI": "fat",
    "LYM": "lymphocytes",
    "MUC": "mucus",
    "DEB": "necrosis",   # "debris": necrosis + debris
    "STR": "stroma",
}


def sample_nct(nct_dir, n_per_class=50, seed=0):
    """
    Pick n random tiles per class. Returns a list of (path, our_class_name).
    Note: NCT-CRC has no patient IDs, so this is a quick check, not a patient-level test.
    """
    rng = np.random.default_rng(seed)
    items = []
    for folder, cls in NCT_TO_CLASS.items():
        files = sorted((Path(nct_dir) / folder).glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"no .tif files in {Path(nct_dir) / folder}")
        for i in rng.choice(len(files), min(n_per_class, len(files)), replace=False):
            items.append((files[i], cls))
    return items


# The PathGen 2k sample comes from 10 TCGA slides of many cancer types; only these are colorectal
# (GDC file_id -> slide). Our tissue classes are for colon, so tests default to these.
PATHGEN_COLORECTAL_FILE_IDS = {
    "bffacf34-4942-496d-9c5d-d36294d80a9d": "TCGA-AA-3844 (COAD)",
    "ed7c20f8-e2a0-4a07-baaf-5002433d8c0c": "TCGA-CM-4746 (COAD)",
    "32abfee5-daea-4816-8c24-122dbf593aff": "TCGA-AH-6547 (READ)",
}


def sample_pathgen(pathgen_dir, n=16, seed=0, file_ids=PATHGEN_COLORECTAL_FILE_IDS):
    """
    Pick n random PathGen patches that have a caption, from slides in `file_ids`
    (None = any slide). Returns a list of dicts: {path, file_id, x, y, caption}.
    x, y are full-resolution pixel positions in the TCGA slide.
    """
    pngs = sorted(p for p in Path(pathgen_dir).glob("*.png") if p.with_suffix(".txt").exists()
                  and (file_ids is None or p.stem.rsplit("_", 2)[0] in file_ids))
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(pngs), min(n, len(pngs)), replace=False):
        p = pngs[i]
        file_id, x, y = p.stem.rsplit("_", 2)
        out.append({"path": p, "file_id": file_id, "x": int(x), "y": int(y),
                    "caption": p.with_suffix(".txt").read_text().strip()})
    return out


def load_rgb(path):
    return Image.open(path).convert("RGB")
