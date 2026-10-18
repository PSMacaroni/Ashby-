"""
Re-test the two fixes (src/detectors.py: background guard + immune detector) on datasets we used EARLIER but that were
NOT used to design or train either fix (Kather 2016 is left out on purpose: the fixes were tuned on it).

  A. PathGen patches (TCGA, AI-written captions): does the immune detector follow caption mentions of lymphocytes /
     inflammation better than the tissue model's P(lymphocytes)? Negated mentions ("no inflammation") are left out.
  B. NCT-CRC-HE-100K (89k labeled tissue tiles): "no harm" check: how many real tissue tiles would the guard remove?

Runs on Delta (GPU for the PathGen embeddings; NCT needs nct_100k.npz from embed_tiles.py).

    python scripts/eval_fixes_existing.py --run fixes_existing_v1
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config  # first, so HF_HOME from .env is applied

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from eval_provided import embed, pathgen_tiles
from src.datasets import NCT_TO_CLASS, PATHGEN_COLORECTAL_FILE_IDS, load_rgb
from src.detectors import colorful_fraction, is_background, load_immune, predict_immune
from src.models import load_conch
from src.prompts import CLASSES

L = CLASSES.index("lymphocytes")
LYMPH_RE = re.compile(r"lymphocyt|inflammat|immune cell|infiltrat|plasma cell|lymphoid", re.I)
NEG_RE = re.compile(r"\b(no|without|absence of|absent|lack of|minimal|scant|rare)\b[^.]{0,40}"
                    r"(lymphocyt|inflammat|immune cell|infiltrat|plasma cell|lymphoid)", re.I)


def colorful_of(path):
    return colorful_fraction(load_rgb(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tissue-model", default="models/tissue8_split_v1.joblib")
    ap.add_argument("--immune-model", default="models/immune_v1.joblib")
    ap.add_argument("--nct-emb", default="data/embeddings/nct_100k.npz")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    out = Path("data/results/fixes_existing") / args.run
    out.mkdir(parents=True, exist_ok=True)
    t8, imm = joblib.load(args.tissue_model), load_immune(args.immune_model)
    res = {}

    # ---- A. PathGen: immune detector vs tissue model P(lymphocytes), against caption mentions ----
    pg_dir = Path(config.DATA_DIR) / "pathgen_dataset" / "pathgen_2k_output"
    pngs = sorted(p for p in pg_dir.glob("*.png") if p.with_suffix(".txt").exists())
    device = config.get_device()
    model, preprocess = load_conch(device=device, image_size=224)
    rows = []
    for p in tqdm(pngs, desc="PathGen"):
        try:
            tiles = pathgen_tiles(load_rgb(p), 0.25)
        except Exception:
            continue
        e = embed(model, preprocess, tiles, device)
        pl = t8.predict_proba(e)[:, list(t8.classes_).index(L)]
        pi = predict_immune(imm, e)
        cap = p.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore")
        mention = bool(LYMPH_RE.search(cap))
        rows.append({"patch": p.name, "file_id": p.stem.rsplit("_", 2)[0], "mention": mention,
                     "negated": bool(NEG_RE.search(cap)), "t8_max": pl.max(), "t8_mean": pl.mean(),
                     "imm_max": pi.max(), "imm_mean": pi.mean(), "caption": cap.strip().replace("\n", " ")[:300]})
    pg = pd.DataFrame(rows)
    pg.to_csv(out / "pathgen_scores.csv", index=False)
    usable = pg[~pg.negated]
    res["pathgen"] = {"patches": int(len(pg)), "negated_left_out": int(pg.negated.sum())}
    for name, sub in [("all", usable), ("colorectal", usable[usable.file_id.isin(PATHGEN_COLORECTAL_FILE_IDS)])]:
        r = {"n": int(len(sub)), "mentions": int(sub.mention.sum())}
        if 0 < sub.mention.sum() < len(sub):
            for c in ["t8_max", "t8_mean", "imm_max", "imm_mean"]:
                r[c] = {"auroc": float(roc_auc_score(sub.mention, sub[c])),
                        "mean_mentioned": float(sub[c][sub.mention].mean()), "mean_not": float(sub[c][~sub.mention].mean()),
                        "flag_mentioned": float((sub[c][sub.mention] >= 0.5).mean()), "flag_not": float((sub[c][~sub.mention] >= 0.5).mean())}
        res["pathgen"][name] = r
    print(json.dumps(res["pathgen"], indent=1), flush=True)

    # ---- B. NCT-CRC-HE-100K: how many real tissue tiles would the guard remove? ----
    d = np.load(args.nct_emb)
    nct_root = Path(config.DATA_DIR) / "NCT-CRC-HE-100K" / "data-samples"
    paths = [nct_root / p for p in d["paths"]]
    with ProcessPoolExecutor(args.workers) as pool:
        colorful = np.array(list(tqdm(pool.map(colorful_of, paths, chunksize=256), total=len(paths), desc="NCT colorful")))
    label = np.array(CLASSES)[t8.predict(d["embeddings"])]
    true = np.array(d["label_names"])[d["labels"]]
    bg = is_background(colorful, label)
    res["nct_guard"] = {"tiles": int(len(bg)), "removed": int(bg.sum()),
                        "removed_by_true_class": {c: int(bg[true == c].sum()) for c in CLASSES},
                        "removed_correctly_labeled": int((bg & (label == true)).sum()),
                        "colorful_5th_pct_by_class": {c: float(np.percentile(colorful[true == c], 5)) for c in CLASSES}}
    back = sorted((nct_root / "BACK").glob("*.tif"))[:2000]
    with ProcessPoolExecutor(args.workers) as pool:
        bc = np.array(list(pool.map(colorful_of, back, chunksize=64)))
    res["nct_back"] = {"tiles_checked": int(len(bc)), "colorful_median": float(np.median(bc)),
                       "would_pass_guard_colorful<0.10": float((bc < 0.10).mean())}
    pd.DataFrame({"path": d["paths"], "true": true, "label": label, "colorful": colorful, "background": bg}).to_csv(
        out / "nct_guard.csv.gz", index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("nct_guard", "nct_back")}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
