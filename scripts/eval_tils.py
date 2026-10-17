"""
Test lymphocyte scores on TCGA-TILs (Abousamra et al. 2022 / Saltz et al. 2018; Zenodo 6604094, CC-BY 4.0):
raw (not color-normalized) TCGA colon (coad) and rectum (read) patches labeled TIL-positive (>= 2 lymphocytes) or
TIL-negative. Never used for training any of our models.

Compares, per patch: tissue8 P(lymphocytes), zero-shot CONCH P(lymphocytes), immune detector v0 and v1.
Needs embeddings from embed_tiles.py named tils_<cancer>_<split>.npz (labels til-negative / til-positive).

    python scripts/eval_tils.py --run tils_v1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.prompts import CLASSES
from train_classifier import zero_shot_probs

L = CLASSES.index("lymphocytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--models", nargs="+", default=["tissue8=models/tissue8_split_v1.joblib",
                                                    "immune_v0=data/results/fixes/immune_detector_v0.joblib",
                                                    "immune_v1=models/immune_v1.joblib"])
    args = ap.parse_args()
    out = Path("data/results/tils") / args.run
    out.mkdir(parents=True, exist_ok=True)
    models = {k: joblib.load(v) for k, v in (m.split("=", 1) for m in args.models) if Path(v).exists()}

    rows = []
    for f in sorted(Path(args.emb_dir).glob("tils_*.npz")):
        d = np.load(f)
        _, cancer, split = f.stem.split("_", 2)
        names = list(d["label_names"])
        y = (np.array(names)[d["labels"]] == "til-positive").astype(int)
        scores = {"zero_shot": zero_shot_probs(d["embeddings"], d)[:, L]}
        for k, m in models.items():
            p = m.predict_proba(d["embeddings"])
            scores[k] = p[:, list(m.classes_).index(L)] if k == "tissue8" else p[:, 1]
        patients = [Path(p).name.split("_")[0] for p in d["paths"]]
        rows.append(pd.DataFrame({"cancer": cancer, "split": split, "patient": patients, "til_positive": y, **scores}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "scores.csv.gz", index=False)
    score_cols = [c for c in df.columns if c in ["zero_shot", *models]]
    res = {"n_patches": int(len(df)), "n_positive": int(df.til_positive.sum()), "n_patients": int(df.patient.nunique())}
    rng = np.random.default_rng(0)
    pts = df.patient.unique()
    for name, sub in [("all", df), *[(c, df[df.cancer == c]) for c in sorted(df.cancer.unique())]]:
        r = {"n": int(len(sub)), "positive": int(sub.til_positive.sum())}
        for c in score_cols:
            auc = roc_auc_score(sub.til_positive, sub[c])
            boots = []
            sp = sub.patient.unique()
            for _ in range(300):
                pick = rng.choice(sp, len(sp))
                b = pd.concat([sub[sub.patient == q] for q in pick])
                if 0 < b.til_positive.sum() < len(b):
                    boots.append(roc_auc_score(b.til_positive, b[c]))
            pos, neg = sub[c][sub.til_positive == 1], sub[c][sub.til_positive == 0]
            r[c] = {"auroc": float(auc), "auroc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "mean_pos": float(pos.mean()), "mean_neg": float(neg.mean()),
                    "flag_pos": float((pos >= 0.5).mean()), "flag_neg": float((neg >= 0.5).mean())}
        res[name] = r
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{res['n_patches']} patches ({res['n_positive']} TIL-positive) from {res['n_patients']} patients")
    for name in ["all", *sorted(df.cancer.unique())]:
        print(f"== {name} ({res[name]['n']} patches, {res[name]['positive']} positive)")
        for c in score_cols:
            m = res[name][c]
            print(f"  {c:10} AUROC {m['auroc']:.3f} [{m['auroc_ci'][0]:.3f}-{m['auroc_ci'][1]:.3f}] | mean pos {m['mean_pos']:.3f} neg {m['mean_neg']:.3f} | "
                  f">=0.5: pos {m['flag_pos']:.1%} neg {m['flag_neg']:.1%}")


if __name__ == "__main__":
    main()
