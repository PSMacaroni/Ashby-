"""
Test our trained models on the hackathon's own datasets, which were NOT used for training:

  A. NCT-CRC-HE-100K (~89k labeled colon tiles): 8-class tissue accuracy of the STARC-9-trained classifier
     (data/models/conch_lr_starc9_lr.joblib) vs zero-shot CONCH.
  B. Same NCT tiles through the nerve / vessel side classifier (data/models/conch_structures_v1.joblib):
     NCT has no nerve/vessel labels, so we report how often each tissue gets flagged + example images.
  C. PathGen patches (TCGA, AI-written captions): weak check of both models against the captions:
     does the caption mention nerves / vessels / tumor where the models see them?

Needs nct_100k.npz from embed_tiles.py (default labels). PathGen patches are embedded here (GPU if available).

    python scripts/eval_provided.py --run provided_v1
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config  # first, so HF_HOME from .env is applied

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, PngImagePlugin
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.classifier import BLOOD, LABEL_NAMES, predict
from src.datasets import load_rgb
from src.models import load_conch
from src.prompts import CLASSES
from src.structures import STRUCTURE_LABELS, load_structures, predict_structures
from train_classifier import save_confusion, score, zero_shot_probs
from train_structures import montage

N = len(CLASSES)
PngImagePlugin.MAX_TEXT_CHUNK = 64 * 2**20  # some PathGen PNGs carry big text metadata
NERVE_RE = re.compile(r"\bnerv|perineural|\bneural\b|ganglion", re.I)
VESSEL_RE = re.compile(r"vessel|vascular|capillar|arteri|\barter|\bvein|venule|endotheli", re.I)
TUMOR_RE = re.compile(r"carcinoma|adenocarcinoma|tumou?r|malignan|neoplas|cancer", re.I)


def pathgen_tiles(img, mpp, out_px=224, target_mpp=0.5):
    """
    Cover a PathGen patch with the 224 px / 0.5 microns-per-px tiles our models expect (overlapping if the
    patch isn't a whole number of tiles: 672 px at 0.25 microns/px -> 2 x 2 windows of 448 px).
    """
    src = min(int(round(out_px * target_mpp / mpp)), img.width)
    k = -(-img.width // src)  # ceil
    starts = np.linspace(0, img.width - src, k).round().astype(int)
    return [img.crop((x, y, x + src, y + src)).resize((out_px, out_px), Image.BILINEAR) for y in starts for x in starts]


@torch.inference_mode()
def embed(model, preprocess, images, device, batch=128):
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([preprocess(im) for im in images[i:i + batch]]).to(device)
        out.append(model.encode_image(x, proj_contrast=True, normalize=True).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def auroc(y, s):
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--nct-emb", default="data/embeddings/nct_100k.npz")
    ap.add_argument("--nct-root", default=None, help="NCT folder (for example images); default DATA_DIR/NCT-CRC-HE-100K/data-samples")
    ap.add_argument("--tissue-model", default="data/models/conch_lr_starc9_lr.joblib")
    ap.add_argument("--structures-model", default="data/models/conch_structures_v1.joblib")
    ap.add_argument("--pathgen-dir", default=None, help="default DATA_DIR/pathgen_dataset/pathgen_2k_output")
    ap.add_argument("--pathgen-mpp", type=float, default=0.25, help="microns per pixel of PathGen patches (40x: red cells ~30 px)")
    ap.add_argument("--pathgen-limit", type=int, default=None, help="max patches (quick tests)")
    args = ap.parse_args()
    out = Path("data/results/eval_provided") / args.run
    out.mkdir(parents=True, exist_ok=True)
    nct_root = Path(args.nct_root or Path(config.DATA_DIR) / "NCT-CRC-HE-100K" / "data-samples")
    pathgen_dir = Path(args.pathgen_dir or Path(config.DATA_DIR) / "pathgen_dataset" / "pathgen_2k_output")
    tissue_clf = joblib.load(args.tissue_model)
    bundle = load_structures(args.structures_model)
    results = {}

    # A. 8 tissue classes on NCT-CRC-HE-100K
    d = np.load(args.nct_emb)
    assert list(d["label_names"]) == LABEL_NAMES
    X, y, paths = d["embeddings"], d["labels"], d["paths"]
    zs = zero_shot_probs(X, d)
    lr8, is_blood = predict(tissue_clf, X)
    lr_pred = np.where(is_blood, BLOOD, lr8.argmax(1))
    results["nct_tissue"] = {"zero_shot": score(y, zs.argmax(1), zs), "classifier": score(y, lr_pred, lr8)}
    save_confusion(y, zs.argmax(1), list(range(N)), list(range(N)), out / "confusion_nct_zero_shot", "zero-shot CONCH on NCT-CRC-HE-100K")
    save_confusion(y, lr_pred, list(range(N)), list(range(N + 1)), out / "confusion_nct_classifier",
                   "STARC-9 classifier on NCT-CRC-HE-100K")
    print("A done:", {k: round(v["accuracy"], 3) for k, v in results["nct_tissue"].items()}, flush=True)

    # B. nerve / vessel flags on the same NCT tiles
    s = predict_structures(bundle, X)
    flag = {c: {"n": int((y == i).sum()),
                "as_nerve": float((s["structure"][y == i] == "nerve").mean()),
                "as_vessel": float((s["structure"][y == i] == "vessel").mean())} for i, c in enumerate(CLASSES)}
    results["nct_structures"] = {"flagged_by_tissue": flag,
                                 "overall_nerve": float((s["structure"] == "nerve").mean()),
                                 "overall_vessel": float((s["structure"] == "vessel").mean())}
    for key in ("nerve", "vessel"):
        top = np.argsort(-s[f"p_{key}"])[:12]
        montage([nct_root / paths[i] for i in top],
                [f"{CLASSES[y[i]]} p_{key} {s[f'p_{key}'][i]:.2f}" for i in top], out / f"nct_top_{key}.png")
    pd.DataFrame({"path": paths, "true": np.array(LABEL_NAMES)[y], "pred": np.array(LABEL_NAMES)[lr_pred],
                  **{f"p_{c}": lr8[:, i] for i, c in enumerate(CLASSES)}, "is_blood": is_blood,
                  "p_nerve": s["p_nerve"], "p_vessel": s["p_vessel"]}).to_csv(out / "predictions_nct.csv.gz", index=False)
    print("B done", flush=True)

    # C. PathGen patches vs their captions
    pngs = sorted(p for p in pathgen_dir.glob("*.png") if p.with_suffix(".txt").exists())[:args.pathgen_limit]
    device = config.get_device()
    model, preprocess = load_conch(device=device, image_size=224)
    rows = []
    skipped = []
    for p in tqdm(pngs, desc="PathGen"):
        try:
            tiles = pathgen_tiles(load_rgb(p), args.pathgen_mpp)
        except Exception as e:  # unreadable image: count it and move on
            skipped.append((p.name, repr(e)[:120]))
            continue
        e = embed(model, preprocess, tiles, device)
        t8, tb = predict(tissue_clf, e)
        st = predict_structures(bundle, e)
        cap = p.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore")
        file_id = p.stem.rsplit("_", 2)[0]
        rows.append({"patch": p.name, "file_id": file_id, "n_tiles": len(tiles),
                     "caption_nerve": bool(NERVE_RE.search(cap)), "caption_vessel": bool(VESSEL_RE.search(cap)),
                     "caption_tumor": bool(TUMOR_RE.search(cap)),
                     "max_p_nerve": float(st["p_nerve"].max()), "max_p_vessel": float(st["p_vessel"].max()),
                     "any_nerve": bool((st["structure"] == "nerve").any()), "any_vessel": bool((st["structure"] == "vessel").any()),
                     "tumor_fraction": float(((t8.argmax(1) == CLASSES.index("tumor")) & ~tb).mean()),
                     "main_tissue": CLASSES[int(np.bincount(t8.argmax(1), minlength=N).argmax())],
                     "caption": cap.strip().replace("\n", " ")[:400]})
    pg = pd.DataFrame(rows)
    pg.to_csv(out / "predictions_pathgen.csv", index=False)
    from src.datasets import PATHGEN_COLORECTAL_FILE_IDS
    res_pg = {}
    for name, sub in [("all", pg), ("colorectal", pg[pg.file_id.isin(PATHGEN_COLORECTAL_FILE_IDS)])]:
        r = {"n_patches": int(len(sub))}
        for key in ("nerve", "vessel"):
            m = sub[f"caption_{key}"]
            r[key] = {"caption_mentions": int(m.sum()), "auroc_max_p": auroc(m, sub[f"max_p_{key}"]),
                      "flagged_when_mentioned": float(sub[f"any_{key}"][m].mean()) if m.any() else None,
                      "flagged_when_not_mentioned": float(sub[f"any_{key}"][~m].mean()) if (~m).any() else None}
        m = sub["caption_tumor"]
        r["tumor"] = {"caption_mentions": int(m.sum()), "auroc_tumor_fraction": auroc(m, sub["tumor_fraction"]),
                      "mean_tumor_fraction_mentioned": float(sub.tumor_fraction[m].mean()) if m.any() else None,
                      "mean_tumor_fraction_not_mentioned": float(sub.tumor_fraction[~m].mean()) if (~m).any() else None}
        res_pg[name] = r
    results["pathgen"] = res_pg
    results["pathgen_skipped"] = skipped
    for key in ("nerve", "vessel"):
        top = pg.sort_values(f"max_p_{key}", ascending=False).head(12)
        montage([pathgen_dir / f for f in top.patch], [f"p_{key} {v:.2f} cap:{'Y' if c else 'n'}"
                for v, c in zip(top[f"max_p_{key}"], top[f"caption_{key}"])], out / f"pathgen_top_{key}.png")
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
