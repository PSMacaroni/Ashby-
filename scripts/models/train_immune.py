"""
Train the immune (lymphocyte-presence) detector on CONCH embeddings and test it. CPU, ~1 minute.
See src/detectors.py and docs/fixes_background_lymph.md.

Positive = lymphocytes / inflammatory cells present:
  NCT-CRC-HE-100K LYM, STARC-9 LYM, all SPIDER "Inflammation" tiles from TRAIN slides (colon + skin).
Negative = everything else: NCT / STARC-9 other tissue (incl. blood), SPIDER other classes (train slides).

Tests (never used for training): Kather 2016 (scattered-lymphocyte tiles the tissue model missed), SPIDER test slides,
and per-class sanity checks on CRC-VAL-HE-7K, STARC-9 VAL-LARGE and STARC-9 TCGA.

    python scripts/models/train_immune.py --run immune_v1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.prompts import CLASSES
from train_classifier import zero_shot_probs
from train_split_models import spider

L = CLASSES.index("lymphocytes")


def npz(emb_dir, name):
    return np.load(Path(emb_dir) / f"{name}.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--colon-manifest", default="data/datasets/spider_colorectal/manifest.csv")
    ap.add_argument("--skin-manifest", default="data/datasets/spider_skin/manifest.csv")
    ap.add_argument("--neg-per-dataset", type=int, default=15000)
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    args = ap.parse_args()
    E, rng = args.emb_dir, np.random.default_rng(0)
    out = Path("data/results/immune") / args.run
    out.mkdir(parents=True, exist_ok=True)

    X, y, g = [], [], []
    def add(emb, label, groups):
        X.append(emb); y.append(np.full(len(emb), label)); g.append(groups)
    for name in ["nct_100k", "starc9_train"]:
        d = npz(E, name)
        lym = d["labels"] == L
        add(d["embeddings"][lym], 1, np.array([f"{name}:{i}" for i in np.where(lym)[0]]))
        idx = rng.choice(np.where(~lym)[0], args.neg_per_dataset, replace=False)
        add(d["embeddings"][idx], 0, np.array([f"{name}:{i}" for i in idx]))
    for organ, man in [("colon", args.colon_manifest), ("skin", args.skin_manifest)]:
        inf = npz(E, f"spider_inflam_{organ}_train")
        imeta = pd.read_csv(Path(man).parent.parent / f"spider_inflam_{'colorectal' if organ == 'colon' else 'skin'}" / "manifest.csv").set_index("image")
        add(inf["embeddings"], 1, (organ + ":" + imeta.loc[[Path(p).name for p in inf["paths"]], "slide_id"].astype(str)).to_numpy())
        s = spider(E, organ, "train", man)
        keep = (s["y3"] == 0) & (s["source"] != "Inflammation")      # inflammation already added above
        add(s["X"][keep], 0, s["slide"][keep])
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    print(f"training: {int(y.sum())} positive, {int((1 - y).sum())} negative tiles", flush=True)

    best = None
    for C in args.Cs:
        accs = [balanced_accuracy_score(y[va], LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
                                        .fit(X[tr], y[tr]).predict(X[va])) for tr, va in GroupKFold(5).split(X, y, g)]
        print(f"  C={C:<5} grouped CV balanced accuracy {np.mean(accs):.3f}", flush=True)
        if best is None or np.mean(accs) > best[0]:
            best = (np.mean(accs), C)
    det = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(X, y)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(det, f"data/models/{args.run}.joblib")
    res = {"C": best[1], "train_pos": int(y.sum()), "train_neg": int((1 - y).sum())}

    # Kather 2016: compare with the tissue model's P(lymphocytes), zero-shot and the v0 prototype
    k = npz(E, "kather2016")
    df = pd.read_csv("data/results/fixes/kather_lymph_features.csv")
    df["p_immune_v1"] = det.predict_proba(k["embeddings"])[:, 1]
    v0 = Path("data/results/fixes/immune_detector_v0.joblib")
    if v0.exists():
        df["p_immune_v0"] = joblib.load(v0).predict_proba(k["embeddings"])[:, 1]
    cols = [c for c in ["p_lymph_t8", "p_lymph_zs", "p_immune_v0", "p_immune_v1"] if c in df]
    groups = ["LYMPHO-found", "LYMPHO-missed", "STROMA", "TUMOR", "COMPLEX", "MUCOSA", "DEBRIS", "ADIPOSE", "EMPTY"]
    res["kather"] = {}
    for c in cols:
        def auc(a, b):
            m = df.group.isin([a, b]); return float(roc_auc_score(df.group[m] == a, df[c][m]))
        res["kather"][c] = {"mean": {gr: float(df[c][df.group == gr].mean()) for gr in groups},
                            "flag_rate_0.5": {gr: float((df[c][df.group == gr] >= 0.5).mean()) for gr in groups},
                            "auroc_missed_vs": {b: auc("LYMPHO-missed", b) for b in ["STROMA", "TUMOR", "COMPLEX"]}}
    # SPIDER test slides: inflammation vs other classes
    res["spider_test"] = {}
    for organ, man in [("colon", args.colon_manifest), ("skin", args.skin_manifest)]:
        inf = npz(E, f"spider_inflam_{organ}_test")["embeddings"]
        s = spider(E, organ, "test", man)
        oth = s["X"][(s["y3"] == 0) & (s["source"] != "Inflammation")]
        p_inf, p_oth = det.predict_proba(inf)[:, 1], det.predict_proba(oth)[:, 1]
        res["spider_test"][organ] = {"n_inflammation": int(len(inf)), "n_other": int(len(oth)),
                                     "auroc": float(roc_auc_score(np.r_[np.ones(len(inf)), np.zeros(len(oth))], np.r_[p_inf, p_oth])),
                                     "inflammation_flagged": float((p_inf >= 0.5).mean()), "other_flagged": float((p_oth >= 0.5).mean())}
    # sanity: mean p_immune and flag rate per tissue class on the labeled test sets
    res["by_class"] = {}
    for name in ["crc_val_7k", "starc9_val_large", "starc9_val_tcga"]:
        d = npz(E, name)
        p = det.predict_proba(d["embeddings"])[:, 1]
        names = list(d["label_names"])
        res["by_class"][name] = {names[c]: {"mean": float(p[d["labels"] == c].mean()), "flagged": float((p[d["labels"] == c] >= 0.5).mean())}
                                 for c in sorted(set(d["labels"]))}
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    df.to_csv(out / "kather_scores.csv", index=False)
    print(json.dumps(res, indent=1))
    print(f"saved data/models/{args.run}.joblib and {out}")


if __name__ == "__main__":
    main()
