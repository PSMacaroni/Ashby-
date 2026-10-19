"""
Train the small tissue classifier (logistic regression on CONCH embeddings) and compare it with
zero-shot CONCH on held-out test sets. CPU only, runs in a minute or two.

Uses .npz files from scripts/models/embed_tiles.py (in --emb-dir). Labels: our 8 CLASSES + "blood".
Blood is only recognized so it can be ignored (see src/classifier.py); it is not a map class.

How it's scored, on each test set:
  - tissue tiles (the 8 classes): accuracy, balanced accuracy, per-class accuracy, log-loss,
    calibration error (ECE). If the classifier calls a tissue tile "blood", that counts as wrong
    (the tile would be ignored by mistake).
  - blood tiles: how often the classifier flags them as blood, and what zero-shot calls them.

    python scripts/models/train_classifier.py --train starc9_train --tune starc9_val_small \
        --test starc9_val_large starc9_val_tcga crc_val_7k --run starc9_lr
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss

from src.classifier import BLOOD, LABEL_NAMES, predict, predict_all
from src.prompts import CLASSES

N = len(CLASSES)


def load(emb_dir, names):
    """Stack one or more embedding files. Returns (X, y, data of the first file)."""
    parts = [np.load(Path(emb_dir) / f"{n}.npz") for n in names]
    for p in parts:
        assert list(p["label_names"]) == LABEL_NAMES, "label order differs; re-run embed_tiles.py"
    return (np.concatenate([p["embeddings"] for p in parts]),
            np.concatenate([p["labels"] for p in parts]), parts[0])


def zero_shot_probs(X, data):
    logits = float(data["logit_scale"]) * X @ data["text_emb"].T
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def ece(probs, y, bins=15):
    """Expected calibration error: average gap between confidence and accuracy (0 = perfect)."""
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def score(y, pred, probs8):
    """Metrics on tissue tiles. pred may contain BLOOD (= wrongly ignored)."""
    present = sorted(set(y))
    per_class = {CLASSES[c]: float((pred[y == c] == c).mean()) for c in present}
    return {"n_tiles": int(len(y)),
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "log_loss": float(log_loss(y, probs8.clip(1e-7, 1), labels=list(range(N)))),
            "ece": ece(probs8, y),
            "per_class": per_class,
            "flagged_as_blood": float((pred == BLOOD).mean())}


def save_confusion(y, pred, rows, cols, path, title):
    cm = confusion_matrix(y, pred, labels=list(range(len(LABEL_NAMES))))
    cm = cm[np.ix_(rows, cols)]
    pd.DataFrame(cm, index=[LABEL_NAMES[r] for r in rows], columns=[LABEL_NAMES[c] for c in cols]
                 ).to_csv(path.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [LABEL_NAMES[c] for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [LABEL_NAMES[r] for r in rows])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="embedding file names to train on")
    ap.add_argument("--tune", required=True, help="embedding file used to pick C (never a test set)")
    ap.add_argument("--test", nargs="+", required=True, help="embedding files to test on, once")
    ap.add_argument("--run", required=True, help="name for this run (results + model file)")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    args = ap.parse_args()

    out = Path("data/results/classifier") / args.run
    out.mkdir(parents=True, exist_ok=True)
    X, y, _ = load(args.emb_dir, args.train)
    print(f"training on {len(y)} tiles: " + ", ".join(
        f"{n} {c}" for c, n in zip(LABEL_NAMES, np.bincount(y, minlength=len(LABEL_NAMES))) if n))

    # 1. pick C on the tuning set (balanced accuracy over all its labels, blood included)
    Xv, yv, _ = load(args.emb_dir, [args.tune])
    best = None
    for C in args.Cs:
        t0 = time.time()
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
        bacc = balanced_accuracy_score(yv, predict_all(clf, Xv).argmax(1))
        print(f"  C={C:<6} tuning balanced accuracy {bacc:.3f}  ({time.time() - t0:.0f} s)")
        if best is None or bacc > best[0]:
            best = (bacc, C, clf)
    _, C, clf = best
    print(f"best C = {C}")

    # 2. test once on each test set, zero-shot vs classifier
    results = {}
    for name in args.test:
        Xt, yt, data = load(args.emb_dir, [name])
        zs = zero_shot_probs(Xt, data)
        lr8, is_blood = predict(clf, Xt)
        lr_pred = np.where(is_blood, BLOOD, lr8.argmax(1))
        zs_pred = zs.argmax(1)
        tissue, blood = yt < N, yt == BLOOD
        r = {"zero_shot": score(yt[tissue], zs_pred[tissue], zs[tissue]),
             "classifier": score(yt[tissue], lr_pred[tissue], lr8[tissue])}
        if blood.any():
            r["blood_tiles"] = {
                "n_tiles": int(blood.sum()),
                "classifier_flags_blood": float(is_blood[blood].mean()),
                "zero_shot_calls_them": {CLASSES[c]: float((zs_pred[blood] == c).mean())
                                         for c in range(N) if (zs_pred[blood] == c).any()}}
        results[name] = r
        rows = sorted(set(yt))
        save_confusion(yt, zs_pred, rows, list(range(N)), out / f"confusion_{name}_zero_shot",
                       f"zero-shot CONCH on {name}")
        save_confusion(yt, lr_pred, rows, sorted(set(rows) | {BLOOD}), out / f"confusion_{name}_classifier",
                       f"CONCH + logistic regression on {name}")

    # 3. save model, settings, results
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / f"conch_lr_{args.run}.joblib")
    settings = {"run": args.run, "date": str(date.today()), "train": args.train, "tune": args.tune,
                "test": args.test, "C": C, "Cs_tried": args.Cs, "class_weight": "balanced",
                "label_names": LABEL_NAMES, "image_size": int(data["image_size"]),
                "prompts_version": str(data["prompts_version"]),
                "model_file": str(model_dir / f"conch_lr_{args.run}.joblib")}
    (out / "metrics.json").write_text(json.dumps({"settings": settings, "results": results}, indent=2), encoding="utf-8")

    # 4. a readable table
    lines = [f"# {args.run}: zero-shot CONCH vs CONCH + logistic regression", "",
             f"Trained on {', '.join(args.train)} ({len(y)} tiles), C = {C} picked on {args.tune}.", ""]
    for name, r in results.items():
        zs_r, lr_r = r["zero_shot"], r["classifier"]
        lines += [f"## {name} ({zs_r['n_tiles']} tissue tiles)", "",
                  "| | zero-shot | classifier |", "|---|---|---|"]
        for k in ["accuracy", "balanced_accuracy", "log_loss", "ece"]:
            lines.append(f"| {k} | {zs_r[k]:.3f} | {lr_r[k]:.3f} |")
        for c in zs_r["per_class"]:
            lines.append(f"| {c} | {zs_r['per_class'][c]:.1%} | {lr_r['per_class'][c]:.1%} |")
        lines.append(f"| tissue wrongly flagged as blood | n/a | {lr_r['flagged_as_blood']:.1%} |")
        if "blood_tiles" in r:
            b = r["blood_tiles"]
            calls = ", ".join(f"{c} {v:.0%}" for c, v in sorted(b["zero_shot_calls_them"].items(),
                                                                 key=lambda kv: -kv[1]))
            lines += ["", f"Blood tiles ({b['n_tiles']}): classifier flags {b['classifier_flags_blood']:.1%} "
                          f"as blood; zero-shot calls them: {calls}."]
        lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved results to {out} and model to {settings['model_file']}")


if __name__ == "__main__":
    main()
