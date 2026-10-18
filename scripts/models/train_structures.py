"""
Train and test the nerve / vessel side classifier (src/structures.py) on CONCH embeddings of SPIDER tiles,
and compare it with zero-shot CONCH. Also tries zero-shot artery vs vein on vessel tiles (no labels exist,
so that part is only described, plus example images to look at). CPU only.

Inputs (from get_spider.py + embed_tiles.py --labels other nerve vessel):
    data/embeddings/spider_<key>_<train|test>.npz  and  <dataset dir>/manifest.csv (slide ids, source classes)

    python scripts/models/train_structures.py --data colon=/work/hdd/bfqi/$USER/datasets/spider_colorectal \
        skin=/work/hdd/bfqi/$USER/datasets/spider_skin --run conch_structures_v1

Output: data/results/structures/<run>/ (results.md, metrics.json, predictions_<key>.csv, confusion + example PNGs)
        data/models/<run>.joblib (classifier + artery/vein text embeddings; load with src/structures.py)
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # first, so HF_HOME from .env is applied

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models import load_conch
from src.structures import (STRUCTURE_LABELS, VESSEL_TYPE_NAMES, encode_prompt_sets, prompt_sets, softmax,
                            zero_shot_sets, zero_shot_structures)

NERVE, VESSEL = STRUCTURE_LABELS.index("nerve"), STRUCTURE_LABELS.index("vessel")


def load(emb_dir, key, split, data_dir):
    d = np.load(Path(emb_dir) / f"spider_{key}_{split}.npz")
    assert list(d["label_names"]) == STRUCTURE_LABELS, d["label_names"]
    man = pd.read_csv(Path(data_dir) / "manifest.csv").set_index("image")
    images = [Path(p).name for p in d["paths"]]
    meta = man.loc[images]
    return {"X": d["embeddings"], "y": d["labels"], "images": np.array(images),
            "paths": np.array([str(Path(data_dir) / split / p) for p in d["paths"]]),
            "slide": (key + ":" + meta["slide_id"].astype(str)).to_numpy(),
            "source": meta["source_class"].to_numpy(), "logit_scale": float(d["logit_scale"])}


def scores(y, p):
    pred = p.argmax(1)
    r = {"n_tiles": int(len(y)), "accuracy": float((pred == y).mean()),
         "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
         "log_loss": float(log_loss(y, p.clip(1e-7, 1), labels=list(range(len(STRUCTURE_LABELS)))))}
    for c in sorted(set(y)):
        r[f"recall_{STRUCTURE_LABELS[c]}"] = float((pred[y == c] == c).mean())
    for c in (NERVE, VESSEL):
        name = STRUCTURE_LABELS[c]
        flagged = pred == c
        r[f"flagged_{name}"] = float(flagged.mean())
        if c in set(y):
            r[f"precision_{name}"] = float((y[flagged] == c).mean()) if flagged.any() else 0.0
            r[f"auroc_{name}"] = float(roc_auc_score(y == c, p[:, c]))
    return r, pred


def false_alarms(source, y, pred):
    """For 'other' tiles: share called nerve / vessel, per original SPIDER class."""
    rows = {}
    for s in sorted(set(source[y == 0])):
        m = (y == 0) & (source == s)
        rows[s] = {"n": int(m.sum()), "as_nerve": float((pred[m] == NERVE).mean()),
                   "as_vessel": float((pred[m] == VESSEL).mean())}
    return rows


def save_confusion(y, pred, path, title):
    cm = confusion_matrix(y, pred, labels=range(len(STRUCTURE_LABELS)))
    rows = sorted(set(y))
    cm = cm[rows]
    pd.DataFrame(cm, index=[STRUCTURE_LABELS[r] for r in rows], columns=STRUCTURE_LABELS).to_csv(path.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), STRUCTURE_LABELS)
    ax.set_yticks(range(len(rows)), [STRUCTURE_LABELS[r] for r in rows])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    for i in range(len(rows)):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)


def montage(paths, captions, path, cols=6):
    """Grid of tiles with a caption under each, to eyeball what the model picked."""
    n, size, cap = len(paths), 224, 18
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, c) in enumerate(zip(paths, captions)):
        x, y = (i % cols) * size, (i // cols) * (size + cap)
        sheet.paste(Image.open(p).convert("RGB").resize((size, size)), (x, y))
        draw.text((x + 4, y + size + 3), c, fill="black")
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="key=dataset_dir, e.g. colon=/work/.../spider_colorectal")
    ap.add_argument("--run", required=True)
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    args = ap.parse_args()
    data = dict(kv.split("=", 1) for kv in args.data)
    out = Path("data/results/structures") / args.run
    out.mkdir(parents=True, exist_ok=True)

    train = [load(args.emb_dir, k, "train", d) for k, d in data.items()]
    X = np.concatenate([t["X"] for t in train])
    y = np.concatenate([t["y"] for t in train])
    groups = np.concatenate([t["slide"] for t in train])
    print(f"training tiles: {len(y)} ({', '.join(f'{n} {c}' for c, n in zip(STRUCTURE_LABELS, np.bincount(y, minlength=3)))}), "
          f"{len(set(groups))} slides")

    # 1. pick C with 5-fold cross-validation split by slide (never by tile)
    best = None
    for C in args.Cs:
        accs = []
        for tr, va in GroupKFold(5).split(X, y, groups):
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[va], clf.predict(X[va])))
        print(f"  C={C:<5} slide-grouped CV balanced accuracy {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        if best is None or np.mean(accs) > best[0]:
            best = (np.mean(accs), C)
    C = best[1]
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
    print(f"best C = {C}")

    # 2. zero-shot text embeddings (CONCH text encoder on CPU is enough)
    model, _ = load_conch(device="cpu", image_size=224)
    zs_sets = zero_shot_sets()
    zs_names = list(zs_sets)
    zs_text = encode_prompt_sets(model, zs_sets)
    vt_names = list(VESSEL_TYPE_NAMES)
    vt_text = encode_prompt_sets(model, prompt_sets(VESSEL_TYPE_NAMES))
    scale = train[0]["logit_scale"]

    # 3. test once per organ
    results = {}
    for key, d in data.items():
        t = load(args.emb_dir, key, "test", d)
        p_lr = np.zeros((len(t["y"]), 3))
        p_lr[:, clf.classes_] = clf.predict_proba(t["X"])
        p_zs = zero_shot_structures(t["X"], zs_text, zs_names, scale)
        p_art = softmax(scale * t["X"] @ vt_text.T)[:, vt_names.index("artery")]
        r = {}
        for method, p in [("zero_shot", p_zs), ("classifier", p_lr)]:
            r[method], pred = scores(t["y"], p)
            r[method]["false_alarms_by_class"] = false_alarms(t["source"], t["y"], pred)
            save_confusion(t["y"], pred, out / f"confusion_{key}_{method}", f"{method} on SPIDER-{key} test")
        vessels = t["y"] == VESSEL
        r["vessel_type_zero_shot"] = {"n_vessel_tiles": int(vessels.sum()),
                                      "share_called_artery": float((p_art[vessels] > 0.5).mean()),
                                      "median_p_artery": float(np.median(p_art[vessels]))}
        order = np.argsort(p_art[vessels])
        vp, va = t["paths"][vessels], p_art[vessels]
        montage(vp[order[-12:][::-1]], [f"p_artery {v:.2f}" for v in va[order[-12:][::-1]]], out / f"most_artery_like_{key}.png")
        montage(vp[order[:12]], [f"p_artery {v:.2f}" for v in va[order[:12]]], out / f"most_vein_like_{key}.png")
        nerves = np.where(t["y"] == NERVE)[0]
        if len(nerves):
            pick = nerves[np.random.default_rng(0).choice(len(nerves), min(12, len(nerves)), replace=False)]
            montage(t["paths"][pick], [f"p_nerve {v:.2f}" for v in p_lr[pick, NERVE]], out / f"nerve_examples_{key}.png")
        pd.DataFrame({"image": t["images"], "slide": t["slide"], "true": np.array(STRUCTURE_LABELS)[t["y"]],
                      "source_class": t["source"], "p_other": p_lr[:, 0], "p_nerve": p_lr[:, 1], "p_vessel": p_lr[:, 2],
                      "zs_p_nerve": p_zs[:, 1], "zs_p_vessel": p_zs[:, 2], "p_artery_zero_shot": p_art}
                     ).to_csv(out / f"predictions_{key}.csv", index=False)
        results[key] = r

    # 4. save the bundle the pipeline will load, plus settings and a readable summary
    Path("data/models").mkdir(parents=True, exist_ok=True)
    bundle = {"clf": clf, "labels": STRUCTURE_LABELS, "vessel_type_text": vt_text, "vessel_type_names": vt_names,
              "logit_scale": scale, "embedding": "CONCH ViT-B/16, 224 px, proj_contrast + normalize",
              "trained_on": list(data), "C": C, "date": str(date.today())}
    joblib.dump(bundle, Path("data/models") / f"{args.run}.joblib")
    settings = {"run": args.run, "C": C, "Cs_tried": args.Cs, "cv": "GroupKFold(5) by slide",
                "train_tiles": int(len(y)), "train_slides": int(len(set(groups))), "data": data}
    (out / "metrics.json").write_text(json.dumps({"settings": settings, "results": results}, indent=2), encoding="utf-8")

    lines = [f"# {args.run}: nerve / vessel side classifier (CONCH embeddings)", "",
             f"Trained on {len(y)} SPIDER tiles from {len(set(groups))} slides ({', '.join(data)}), C = {C}.", ""]
    for key, r in results.items():
        lines += [f"## SPIDER-{key} test ({r['classifier']['n_tiles']} tiles, slides not seen in training)", "",
                  "| | zero-shot | classifier |", "|---|---|---|"]
        for k in r["classifier"]:
            if k in ("false_alarms_by_class", "n_tiles"):
                continue
            z = r["zero_shot"].get(k)
            lines.append(f"| {k} | {z:.3f} | {r['classifier'][k]:.3f} |" if z is not None else
                         f"| {k} | n/a | {r['classifier'][k]:.3f} |")
        fa = r["classifier"]["false_alarms_by_class"]
        worst = sorted(fa.items(), key=lambda kv: -(kv[1]["as_nerve"] + kv[1]["as_vessel"]))[:5]
        lines += ["", "Other tissue most often mistaken (classifier): " +
                  "; ".join(f"{s} → nerve {v['as_nerve']:.0%}, vessel {v['as_vessel']:.0%}" for s, v in worst), "",
                  f"Artery vs vein (zero-shot, unvalidated): {r['vessel_type_zero_shot']['share_called_artery']:.0%} of "
                  f"{r['vessel_type_zero_shot']['n_vessel_tiles']} vessel tiles called artery.", ""]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {out} and data/models/{args.run}.joblib")


if __name__ == "__main__":
    main()
