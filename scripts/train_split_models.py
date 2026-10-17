"""
Two separate models instead of one:

  Model 1, tissue8: ONLY our 8 tissue classes (no blood, no structures), trained on NCT-CRC-HE-100K + STARC-9
           together, so it learns both hospitals' way of drawing muscle vs stroma.
  Model 2, structures: nerve / vessel / blood / other (blood moved here from the tissue model; it is still
           "identify and ignore"). Trained on SPIDER (nerve, vessel, other) + STARC-9 blood (+ some STARC-9 tissue
           as other).

Both run on the same CONCH embeddings. CPU only, a few minutes. Needs these files in --emb-dir (embed_tiles.py):
  nct_100k, starc9_train, starc9_val_small, starc9_val_large, starc9_val_tcga, crc_val_7k,
  spider_colon_train, spider_colon_test, spider_skin_train, spider_skin_test
and the SPIDER manifests (--spider-colon-manifest / --spider-skin-manifest) for slide ids and source classes.

Model 1 is compared on hospitals it was NOT trained on with: zero-shot CONCH, the STARC-only model
(models/conch_lr_starc9_lr.joblib) and an NCT-only model trained here the same way.

    python scripts/train_split_models.py --run split_v1
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.classifier import LABEL_NAMES, predict
from src.prompts import CLASSES
from train_classifier import save_confusion, score, zero_shot_probs

N = len(CLASSES)
STRUCT = ["other", "nerve", "vessel", "blood"]
# SPIDER colon classes that map cleanly onto ours (adenoma / polyps / inflammation have no clean match: left out).
# SPIDER "Mucus" tiles are normal crypts full of goblet cells (checked by eye), i.e. our normal_mucosa, not mucus pools.
SPIDER_TO_CLASS = {"Adenocarcinoma high grade": "tumor", "Adenocarcinoma low grade": "tumor", "Fat": "fat",
                   "Mucus": "normal_mucosa", "Muscle": "muscle", "Necrosis": "necrosis", "Stroma healthy": "stroma"}


def npz(emb_dir, name):
    return np.load(Path(emb_dir) / f"{name}.npz")


def tissue_only(d):
    """Embeddings + labels of a tissue npz, without blood tiles."""
    keep = d["labels"] < N
    return d["embeddings"][keep], d["labels"][keep]


def fit(X, y, C):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)


def proba(clf, X, n):
    p = np.zeros((len(X), n))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


def spider(emb_dir, organ, split, manifest):
    d = npz(emb_dir, f"spider_{organ}_{split}")
    man = pd.read_csv(manifest).set_index("image")
    meta = man.loc[[Path(p).name for p in d["paths"]]]
    return {"X": d["embeddings"], "y3": d["labels"], "source": meta["source_class"].to_numpy(),
            "slide": (organ + ":" + meta["slide_id"].astype(str)).to_numpy()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--spider-colon-manifest", default="data/datasets/spider_colorectal/manifest.csv")
    ap.add_argument("--spider-skin-manifest", default="data/datasets/spider_skin/manifest.csv")
    ap.add_argument("--old-tissue-model", default="models/conch_lr_starc9_lr.joblib")
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path("data/results/split_models") / args.run
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    E = args.emb_dir
    results = {}

    # ---------------- Model 1: tissue8 (NCT + STARC-9) ----------------
    nct = npz(E, "nct_100k")
    Xn, yn = nct["embeddings"], nct["labels"]
    Xs, ys = tissue_only(npz(E, "starc9_train"))
    Xvs, yvs = tissue_only(npz(E, "starc9_val_small"))
    # pick C on two tuning sets, one per hospital: STARC-9 VAL-SMALL (other Stanford slides) and a random 10% of
    # NCT (NCT has no patient ids; this hold-out is only for choosing C, never for the reported test numbers)
    hold = rng.random(len(yn)) < 0.10
    best = None
    for C in args.Cs:
        clf = fit(np.concatenate([Xn[~hold], Xs]), np.concatenate([yn[~hold], ys]), C)
        b = np.mean([balanced_accuracy_score(yvs, clf.predict(Xvs)), balanced_accuracy_score(yn[hold], clf.predict(Xn[hold]))])
        print(f"tissue8 C={C:<5} tuning balanced accuracy (mean of Stanford + NCT hold-out) {b:.3f}", flush=True)
        if best is None or b > best[0]:
            best = (b, C)
    C1 = best[1]
    tissue8 = fit(np.concatenate([Xn, Xs]), np.concatenate([yn, ys]), C1)
    nct_only = fit(Xn, yn, C1)
    old = joblib.load(args.old_tissue_model)
    print(f"tissue8: C = {C1}, trained on {len(yn) + len(ys)} tiles", flush=True)

    tests = {}
    for name in ["crc_val_7k", "starc9_val_large", "starc9_val_tcga"]:
        d = npz(E, name)
        X, y = tissue_only(d)
        tests[name] = (X, y, d)
    # SPIDER colon (a third hospital, never used for tissue training): mapped classes from train + test splits
    sx, sy = [], []
    for split in ["train", "test"]:
        s = spider(E, "colon", split, args.spider_colon_manifest)
        m = np.array([src in SPIDER_TO_CLASS for src in s["source"]]) & (s["y3"] == 0)
        sx.append(s["X"][m])
        sy.append(np.array([CLASSES.index(SPIDER_TO_CLASS[src]) for src in s["source"][m]]))
    tests["spider_colon"] = (np.concatenate(sx), np.concatenate(sy), tests["crc_val_7k"][2])

    res1 = {}
    for name, (X, y, d) in tests.items():
        r = {}
        zs = zero_shot_probs(X, d)
        r["zero_shot"] = score(y, zs.argmax(1), zs)
        p_old, b_old = predict(old, X)
        r["starc_only"] = score(y, np.where(b_old, 8, p_old.argmax(1)), p_old)
        p = proba(nct_only, X, N)
        r["nct_only"] = score(y, p.argmax(1), p)
        p = proba(tissue8, X, N)
        r["tissue8_combined"] = score(y, p.argmax(1), p)
        save_confusion(y, p.argmax(1), sorted(set(y)), list(range(N)), out / f"confusion_tissue8_{name}",
                       f"tissue8 (NCT+STARC) on {name}")
        res1[name] = r
        print(name, {k: round(v["accuracy"], 3) for k, v in r.items()}, flush=True)
    results["tissue8"] = {"C": C1, "train_tiles": int(len(yn) + len(ys)), "tests": res1,
                          "note": "nct_only is trained on NCT, so it is in-hospital on crc_val_7k (different patients)"}

    # ---------------- Model 2: structures (nerve / vessel / blood / other) ----------------
    tr = [spider(E, o, "train", m) for o, m in [("colon", args.spider_colon_manifest), ("skin", args.spider_skin_manifest)]]
    st = npz(E, "starc9_train")
    blood = st["labels"] == 8
    other_idx = np.concatenate([rng.choice(np.where(st["labels"] == c)[0], 1000, replace=False) for c in range(N)])
    X2 = np.concatenate([t["X"] for t in tr] + [st["embeddings"][blood], st["embeddings"][other_idx]])
    y2 = np.concatenate([t["y3"] for t in tr] + [np.full(blood.sum(), 3), np.zeros(len(other_idx), int)])
    g2 = np.concatenate([t["slide"] for t in tr] + [np.array([f"starc:{i}" for i in np.where(blood)[0]]),
                                                    np.array([f"starc:{i}" for i in other_idx])])
    best = None
    for C in args.Cs:
        accs = [balanced_accuracy_score(y2[va], fit(X2[tr_], y2[tr_], C).predict(X2[va]))
                for tr_, va in GroupKFold(5).split(X2, y2, g2)]
        print(f"structures C={C:<5} grouped CV balanced accuracy {np.mean(accs):.3f}", flush=True)
        if best is None or np.mean(accs) > best[0]:
            best = (np.mean(accs), C)
    C2 = best[1]
    struct = fit(X2, y2, C2)
    print(f"structures: C = {C2}, trained on {len(y2)} tiles", flush=True)

    res2 = {}
    for organ, man in [("colon", args.spider_colon_manifest), ("skin", args.spider_skin_manifest)]:
        t = spider(E, organ, "test", man)
        p = proba(struct, t["X"], 4)
        pred = p.argmax(1)
        r = {"n": int(len(t["y3"])), "balanced_accuracy": float(balanced_accuracy_score(t["y3"], pred))}
        for c in sorted(set(t["y3"])):
            r[f"recall_{STRUCT[c]}"] = float((pred[t["y3"] == c] == c).mean())
        for c in (1, 2):
            if c in set(t["y3"]):
                r[f"precision_{STRUCT[c]}"] = float((t["y3"][pred == c] == c).mean()) if (pred == c).any() else 0.0
                r[f"auroc_{STRUCT[c]}"] = float(roc_auc_score(t["y3"] == c, p[:, c]))
        r["other_called_blood"] = float((pred[t["y3"] == 0] == 3).mean())
        res2[f"spider_{organ}_test"] = r
    for name in ["starc9_val_large", "starc9_val_tcga", "crc_val_7k"]:
        d = npz(E, name)
        p = proba(struct, d["embeddings"], 4)
        pred, y = p.argmax(1), d["labels"]
        r = {"blood_tiles": int((y == 8).sum())}
        if (y == 8).any():
            r["blood_found"] = float((pred[y == 8] == 3).mean())
            r["blood_auroc"] = float(roc_auc_score(y == 8, p[:, 3]))
        r["tissue_called_blood"] = {CLASSES[c]: float((pred[y == c] == 3).mean()) for c in range(N) if (y == c).any()}
        r["tissue_called_nerve"] = float((pred[y < N] == 1).mean())
        r["tissue_called_vessel"] = float((pred[y < N] == 2).mean())
        res2[name] = r
    p = proba(struct, Xn, 4)
    pred = p.argmax(1)
    res2["nct_100k_flags"] = {CLASSES[c]: {s: float((pred[yn == c] == i).mean()) for i, s in enumerate(STRUCT) if i}
                              for c in range(N)}
    results["structures"] = {"C": C2, "train_tiles": int(len(y2)), "tests": res2}

    # ---------------- save ----------------
    Path("data/models").mkdir(parents=True, exist_ok=True)
    v1 = joblib.load("models/conch_structures_v1.joblib")
    joblib.dump(tissue8, f"data/models/tissue8_{args.run}.joblib")
    joblib.dump({"clf": struct, "labels": STRUCT, "vessel_type_text": v1["vessel_type_text"],
                 "vessel_type_names": v1["vessel_type_names"], "logit_scale": v1["logit_scale"],
                 "embedding": v1["embedding"], "trained_on": ["SPIDER colon", "SPIDER skin", "STARC-9 blood + tissue"],
                 "C": C2, "date": str(date.today())}, f"data/models/structures_{args.run}.joblib")
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [f"# {args.run}: split models", "", f"## Model 1: tissue8 (8 classes only), NCT + STARC-9, C = {C1}", "",
             "Accuracy on tissue tiles (balanced accuracy in brackets):", "",
             "| test set | zero-shot | STARC-only | NCT-only | **tissue8 (NCT+STARC)** |", "|---|---|---|---|---|"]
    for name, r in res1.items():
        cells = [f"{r[k]['accuracy']:.1%} ({r[k]['balanced_accuracy']:.3f})" for k in ["zero_shot", "starc_only", "nct_only", "tissue8_combined"]]
        lines.append(f"| {name} ({r['zero_shot']['n_tiles']}) | " + " | ".join(cells) + " |")
    lines += ["", "Per class, tissue8:", "", "| class | " + " | ".join(res1) + " |", "|---|" + "---|" * len(res1)]
    for c in CLASSES:
        lines.append(f"| {c} | " + " | ".join(f"{res1[n]['tissue8_combined']['per_class'].get(c, float('nan')):.1%}" for n in res1) + " |")
    lines += ["", f"## Model 2: structures (nerve / vessel / blood / other), C = {C2}", "", "```",
              json.dumps(res2, indent=1), "```"]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"saved {out}, data/models/tissue8_{args.run}.joblib, data/models/structures_{args.run}.joblib")


if __name__ == "__main__":
    main()
