"""
Is MUSK worth it? CONCH vs MUSK vs CONCH+MUSK for our two small classifiers, trained and tested exactly
like scripts/train_split_models.py (split_v1), on the SAME tiles for every backbone.

  tissue8     8 tissue classes, trained on NCT-CRC-HE-100K + STARC-9 train (no blood);
              C picked on STARC-9 VAL-SMALL + a random 10% NCT hold-out (same seed as split_v1)
  structures  other / nerve / vessel / blood, trained on SPIDER colon+skin train + STARC-9 blood
              (+ 1,000 STARC-9 tissue tiles per class as "other"); C by 5-fold CV grouped by slide

Feature sets: conch (512 numbers per tile), musk (1024), both (the two side by side, 1536).
Zero-shot rows (no training) are included for reference: CONCH, MUSK, and their average.

Tests, all from patients never used for training:
  crc_val_7k        Heidelberg, 50 patients, color-normalized
  starc9_val_large  Stanford, 50 slides, color-normalized
  starc9_val_tcga   TCGA, 30 slides, color-normalized. MUSK may have seen TCGA images during pretraining,
                    so this one is reported but left out of the decision.
  spider_colon      a third hospital (SPIDER), mapped classes (needs the SPIDER embeddings)
  kather2016        Mannheim, 10 patients, NOT color-normalized (closest to raw slides); coarse classes,
                    scored like docs/split_models.md; error bars by resampling patients

Needs data/embeddings/<backbone>/<set>.npz from scripts/embed_many.py. CPU only.

    python scripts/compare_backbones.py --run musk_v1                     # everything
    python scripts/compare_backbones.py --run musk_v1 --parts tissue      # the two halves can run as
    python scripts/compare_backbones.py --run musk_v1 --parts structures  #   separate jobs at the same time
    python scripts/compare_backbones.py --run musk_v1 --parts report      # combine what's there (seconds)

Each part saves metrics_<part>.json; every run then rebuilds metrics.json + results.md from whatever parts exist.
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.prompts import CLASSES
from train_classifier import save_confusion, score, zero_shot_probs
from train_split_models import SPIDER_TO_CLASS, STRUCT, fit, proba

N = len(CLASSES)
DECISION_TESTS = ("crc_val_7k", "starc9_val_large", "spider_colon", "kather2016")
KATHER_ALLOWED = {"01_TUMOR": ["tumor"], "02_STROMA": ["stroma", "muscle"], "04_LYMPHO": ["lymphocytes"],
                  "05_DEBRIS": ["necrosis", "mucus"], "06_MUCOSA": ["normal_mucosa"], "07_ADIPOSE": ["fat"]}
KATHER_STRICT = {**KATHER_ALLOWED, "02_STROMA": ["stroma"], "05_DEBRIS": ["necrosis"]}
SWITCH_GAIN = 0.02     # decision rule: adopt a new feature set only if it beats CONCH by >= 2 points on average


class Embeddings:
    """data/embeddings/<backbone>/<set>.npz, and feature sets built from them (checking the tiles line up)."""

    def __init__(self, root, backbones):
        self.root, self.backbones, self.cache = Path(root), list(backbones), {}

    def has(self, name):
        return all((self.root / b / f"{name}.npz").exists() for b in self.backbones)

    def npz(self, backbone, name):
        if (backbone, name) not in self.cache:
            self.cache[backbone, name] = dict(np.load(self.root / backbone / f"{name}.npz"))
        return self.cache[backbone, name]

    def meta(self, name):
        return self.npz(self.backbones[0], name)

    def X(self, features, name):
        if features == "both":
            a, b = self.npz("conch", name), self.npz("musk", name)
            if not (a["paths"] == b["paths"]).all():
                sys.exit(f"{name}: CONCH and MUSK embeddings are not in the same tile order")
            return np.hstack([a["embeddings"], b["embeddings"]])
        return self.npz(features, name)["embeddings"]


def spider_set(E, features, organ, split, manifest):
    name = f"spider_{organ}_{split}"
    meta = pd.read_csv(manifest).set_index("image").loc[[Path(p).name for p in E.meta(name)["paths"]]]
    return {"X": E.X(features, name), "y3": E.meta(name)["labels"], "source": meta["source_class"].to_numpy(),
            "slide": (organ + ":" + meta["slide_id"].astype(str)).to_numpy()}


def kather_patients(paths):
    """Kather 2016 file names carry the patient: ..._CRC-Prim-HE-07_... -> '07'."""
    out = []
    for p in paths:
        m = re.search(r"CRC-Prim-HE-(\d+)", str(p))
        out.append(m.group(1) if m else "unknown")
    return np.array(out)


def kather_score(folder_labels, pred, patients, rng, n_boot=2000):
    """Coarse Kather 2016 scoring (see docs/split_models.md); 95% CI by resampling patients."""
    scored = np.isin(folder_labels, list(KATHER_ALLOWED))
    labels, guess, pts = folder_labels[scored], pred[scored], patients[scored]
    ok = np.array([CLASSES[g] in KATHER_ALLOWED[l] for l, g in zip(labels, guess)])
    strict = np.array([CLASSES[g] in KATHER_STRICT[l] for l, g in zip(labels, guess)])
    unique = np.unique(pts)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(unique, len(unique), replace=True)
        boots.append(np.concatenate([ok[pts == p] for p in pick]).mean())
    empty = folder_labels == "08_EMPTY"
    return {"n_tiles": int(scored.sum()), "accuracy": float(ok.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "strict_accuracy": float(strict.mean()),
            "per_class": {l: float(ok[labels == l].mean()) for l in KATHER_ALLOWED if (labels == l).any()},
            "empty_called_necrosis": float((pred[empty] == CLASSES.index("necrosis")).mean()) if empty.any() else None}


# --- tissue8 --------------------------------------------------------------------------------

def train_tissue8(E, features, hold, Cs):
    Xn, yn = E.X(features, "nct_100k"), E.meta("nct_100k")["labels"]
    ys_all = E.meta("starc9_train")["labels"]
    Xs, ys = E.X(features, "starc9_train")[ys_all < N], ys_all[ys_all < N]
    yv_all = E.meta("starc9_val_small")["labels"]
    Xv, yv = E.X(features, "starc9_val_small")[yv_all < N], yv_all[yv_all < N]
    best = None
    for C in Cs:
        clf = fit(np.concatenate([Xn[~hold], Xs]), np.concatenate([yn[~hold], ys]), C)
        b = np.mean([balanced_accuracy_score(yv, clf.predict(Xv)), balanced_accuracy_score(yn[hold], clf.predict(Xn[hold]))])
        print(f"  tissue8[{features}] C={C:<5} tuning balanced accuracy {b:.3f}", flush=True)
        if best is None or b > best[0]:
            best = (b, C)
    return fit(np.concatenate([Xn, Xs]), np.concatenate([yn, ys]), best[1]), best[1]


def tissue_tests(E, spider_manifest):
    """{test name: (function features -> X, y)} for every tissue test whose embeddings exist."""
    tests = {}
    for name in ("crc_val_7k", "starc9_val_large", "starc9_val_tcga"):
        if E.has(name):
            y_all = E.meta(name)["labels"]
            keep = y_all < N
            tests[name] = (lambda f, n=name, k=keep: E.X(f, n)[k], y_all[keep], name)
    if E.has("spider_colon_train") and E.has("spider_colon_test") and Path(spider_manifest).exists():
        parts = []
        for split in ("train", "test"):
            s = spider_set(E, E.backbones[0], "colon", split, spider_manifest)
            m = np.array([src in SPIDER_TO_CLASS for src in s["source"]]) & (s["y3"] == 0)
            parts.append((f"spider_colon_{split}", m, [CLASSES.index(SPIDER_TO_CLASS[src]) for src in s["source"][m]]))
        tests["spider_colon"] = (lambda f, p=parts: np.concatenate([E.X(f, n)[m] for n, m, _ in p]),
                                 np.concatenate([np.array(y) for _, _, y in parts]), None)
    return tests


def zero_shot(E, backbone, test_name, X_fn, y):
    """Zero-shot probabilities from the backbone's own text classifier (stored in its npz)."""
    ref = test_name if test_name else "nct_100k"        # spider_colon has no single npz: any npz's text_emb works
    return zero_shot_probs(X_fn(backbone), E.npz(backbone, ref))


# --- structures -----------------------------------------------------------------------------

def train_structures(E, features, other_idx, manifests, Cs):
    tr = [spider_set(E, features, organ, "train", manifests[organ]) for organ in ("colon", "skin")]
    st_y = E.meta("starc9_train")["labels"]
    Xst = E.X(features, "starc9_train")
    blood = np.flatnonzero(st_y == 8)
    X2 = np.concatenate([t["X"] for t in tr] + [Xst[blood], Xst[other_idx]])
    y2 = np.concatenate([t["y3"] for t in tr] + [np.full(len(blood), 3), np.zeros(len(other_idx), int)])
    g2 = np.concatenate([t["slide"] for t in tr] + [np.array([f"starc:{i}" for i in blood]),
                                                    np.array([f"starc:{i}" for i in other_idx])])
    best = None
    for C in Cs:
        accs = [balanced_accuracy_score(y2[va], fit(X2[tr_], y2[tr_], C).predict(X2[va]))
                for tr_, va in GroupKFold(5).split(X2, y2, g2)]
        print(f"  structures[{features}] C={C:<5} grouped CV balanced accuracy {np.mean(accs):.3f}", flush=True)
        if best is None or np.mean(accs) > best[0]:
            best = (np.mean(accs), C)
    return fit(X2, y2, best[1]), best[1]


def test_structures(E, features, clf, manifests):
    res = {}
    for organ in ("colon", "skin"):
        if not E.has(f"spider_{organ}_test"):
            continue
        t = spider_set(E, features, organ, "test", manifests[organ])
        p = proba(clf, t["X"], 4)
        pred = p.argmax(1)
        r = {"n": int(len(t["y3"])), "balanced_accuracy": float(balanced_accuracy_score(t["y3"], pred))}
        for c in sorted(set(t["y3"])):
            r[f"recall_{STRUCT[c]}"] = float((pred[t["y3"] == c] == c).mean())
        for c in (1, 2):
            if c in set(t["y3"]):
                r[f"precision_{STRUCT[c]}"] = float((t["y3"][pred == c] == c).mean()) if (pred == c).any() else 0.0
                r[f"auroc_{STRUCT[c]}"] = float(roc_auc_score(t["y3"] == c, p[:, c]))
        res[f"spider_{organ}_test"] = r
    for name in ("starc9_val_large", "starc9_val_tcga", "crc_val_7k"):
        if not E.has(name):
            continue
        y = E.meta(name)["labels"]
        p = proba(clf, E.X(features, name), 4)
        pred = p.argmax(1)
        r = {"blood_tiles": int((y == 8).sum())}
        if (y == 8).any():
            r["blood_found"] = float((pred[y == 8] == 3).mean())
            r["blood_auroc"] = float(roc_auc_score(y == 8, p[:, 3]))
        r["tissue_called_blood"] = {CLASSES[c]: float((pred[y == c] == 3).mean()) for c in range(N) if (y == c).any()}
        res[name] = r
    yn = E.meta("nct_100k")["labels"]
    pred = proba(clf, E.X(features, "nct_100k"), 4).argmax(1)
    res["nct_muscle_flagged"] = {s: float((pred[yn == CLASSES.index("muscle")] == i).mean())
                                 for i, s in enumerate(STRUCT) if i}
    return res


# --- report ---------------------------------------------------------------------------------

def pct(v):
    return "–" if v is None else f"{v:.1%}"


def report(results, feature_sets, run):
    t8, zs = results["tissue8"], results["zero_shot"]
    tests = [t for t in ["crc_val_7k", "starc9_val_large", "starc9_val_tcga", "spider_colon", "kather2016"]
             if any(t in t8[f]["tests"] for f in feature_sets)]
    cols = [(f"zero-shot {k}", zs[k]) for k in zs] + [(f"**{f}**", t8[f]["tests"]) for f in feature_sets]
    lines = [f"# {run}: CONCH vs MUSK vs CONCH+MUSK", "", f"Written {date.today()} by scripts/compare_backbones.py.",
             "", "## tissue8 accuracy (balanced accuracy in brackets for the normalized sets)", "",
             "| test | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for t in tests:
        cells = []
        for _, r in cols:
            v = r.get(t)
            if v is None:
                cells.append("–")
            elif t == "kather2016":
                cells.append(f"{v['accuracy']:.1%} [{v['ci95'][0]:.0%}–{v['ci95'][1]:.0%}]")
            else:
                cells.append(f"{v['accuracy']:.1%} ({v['balanced_accuracy']:.3f})")
        lines.append(f"| {t}{' (not in decision)' if t not in DECISION_TESTS else ''} | " + " | ".join(cells) + " |")
    lines.append("| **average of decision tests** | " + " | ".join(
        pct(results["decision"]["average"].get(c.strip("*").replace("zero-shot ", "zs_"))) for c, _ in cols) + " |")

    lines += ["", "## Known weak spots (higher is better unless marked)", "",
              "| weak spot | " + " | ".join(feature_sets) + " |", "|---|" + "---|" * len(feature_sets)]
    for label, getter in WEAK_SPOTS:
        lines.append(f"| {label} | " + " | ".join(pct(getter(results, f)) for f in feature_sets) + " |")

    if results.get("structures"):
        lines += ["", "## structures (nerve / vessel / blood / other)", "",
                  "| metric | " + " | ".join(feature_sets) + " |", "|---|" + "---|" * len(feature_sets)]
        for label, path in STRUCTURE_ROWS:
            lines.append(f"| {label} | " + " | ".join(pct(dig(results["structures"].get(f, {}), path)) for f in feature_sets) + " |")
    d = results["decision"]
    lines += ["", "## Decision", "", d["text"], ""]
    return "\n".join(lines)


def dig(d, path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


WEAK_SPOTS = [
    ("CRC-VAL-7K muscle", lambda r, f: dig(r, ["tissue8", f, "tests", "crc_val_7k", "per_class", "muscle"])),
    ("CRC-VAL-7K stroma", lambda r, f: dig(r, ["tissue8", f, "tests", "crc_val_7k", "per_class", "stroma"])),
    ("SPIDER colon tumor", lambda r, f: dig(r, ["tissue8", f, "tests", "spider_colon", "per_class", "tumor"])),
    ("Kather 2016 (raw tiles) lymphocytes", lambda r, f: dig(r, ["tissue8", f, "tests", "kather2016", "per_class", "04_LYMPHO"])),
    ("Kather 2016 debris", lambda r, f: dig(r, ["tissue8", f, "tests", "kather2016", "per_class", "05_DEBRIS"])),
    ("Kather 2016 glass called necrosis (lower is better)",
     lambda r, f: dig(r, ["tissue8", f, "tests", "kather2016", "empty_called_necrosis"])),
    ("TCGA fat called blood (lower is better)",
     lambda r, f: dig(r, ["structures", f, "starc9_val_tcga", "tissue_called_blood", "fat"])),
]
STRUCTURE_ROWS = [
    ("SPIDER colon balanced accuracy", ["spider_colon_test", "balanced_accuracy"]),
    ("SPIDER colon vessel recall", ["spider_colon_test", "recall_vessel"]),
    ("SPIDER colon vessel precision", ["spider_colon_test", "precision_vessel"]),
    ("SPIDER skin balanced accuracy", ["spider_skin_test", "balanced_accuracy"]),
    ("SPIDER skin nerve recall", ["spider_skin_test", "recall_nerve"]),
    ("SPIDER skin nerve precision", ["spider_skin_test", "precision_nerve"]),
    ("blood found, Stanford", ["starc9_val_large", "blood_found"]),
    ("blood found, TCGA", ["starc9_val_tcga", "blood_found"]),
    ("TCGA fat called blood (lower is better)", ["starc9_val_tcga", "tissue_called_blood", "fat"]),
    ("NCT muscle called vessel (lower is better)", ["nct_muscle_flagged", "vessel"]),
    ("NCT muscle called nerve (lower is better)", ["nct_muscle_flagged", "nerve"]),
]


def decide(results, feature_sets):
    """Average accuracy over the decision tests that every feature set has; pick by the 2-point rule."""
    avg = {}
    rows = {f"zs_{k}": v for k, v in results["zero_shot"].items()}
    rows.update({f: results["tissue8"][f]["tests"] for f in feature_sets})
    common = [t for t in DECISION_TESTS if all(t in r for r in rows.values())]
    for key, r in rows.items():
        avg[key] = float(np.mean([r[t]["accuracy"] for t in common])) if common else None
    base = avg.get("conch")
    text = f"Average over {', '.join(common)}: " + ", ".join(f"{k} {pct(v)}" for k, v in avg.items()) + "."
    trained = {f: avg[f] for f in feature_sets if avg.get(f) is not None}
    if base is not None and len(trained) > 1:
        best = max(trained, key=trained.get)
        gain = trained[best] - base
        if best != "conch" and gain >= SWITCH_GAIN:
            text += f" **{best} beats CONCH by {gain:.1%}: worth switching** (rule: at least {SWITCH_GAIN:.0%})."
        else:
            text += (f" Best is {best} ({gain:+.1%} vs CONCH): below the {SWITCH_GAIN:.0%} rule, **keep CONCH** "
                     "unless a weak-spot row above changes a lot.")
    return {"average": avg, "tests_used": common, "text": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--emb-root", default="data/embeddings")
    ap.add_argument("--backbones", nargs="+", default=["conch", "musk"], choices=["conch", "musk"])
    ap.add_argument("--spider-colon-manifest", default="data/datasets/spider_colorectal/manifest.csv")
    ap.add_argument("--spider-skin-manifest", default="data/datasets/spider_skin/manifest.csv")
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", nargs="+", choices=["tissue", "structures", "report"], default=["tissue", "structures"],
                    help="tissue / structures can run as separate jobs; 'report' only combines saved parts")
    args = ap.parse_args()

    feature_sets = list(args.backbones) + (["both"] if len(args.backbones) == 2 else [])
    out = Path("data/results/compare_backbones") / args.run
    out.mkdir(parents=True, exist_ok=True)
    if args.parts != ["report"]:
        run_parts(args, feature_sets, out)
    write_report(out, feature_sets, args.run)


def write_report(out, feature_sets, run):
    """Combine metrics_tissue.json + metrics_structures.json (whichever exist) into metrics.json + results.md."""
    results = {"tissue8": {}, "zero_shot": {}, "structures": {}}
    for part in ("tissue", "structures"):
        path = out / f"metrics_{part}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            for key in ("tissue8", "zero_shot", "structures"):
                results[key].update(saved.get(key, {}))
            results.setdefault("date", saved.get("date"))
    if not results["tissue8"]:
        print("no tissue results yet: report not written")
        return
    feature_sets = [f for f in feature_sets if f in results["tissue8"]]
    results["decision"] = decide(results, feature_sets)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    text = report(results, feature_sets, run)
    (out / "results.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nsaved {out}")


def run_parts(args, feature_sets, out):
    E = Embeddings(args.emb_root, args.backbones)
    for need in ("nct_100k", "starc9_train", "starc9_val_small"):
        if not E.has(need):
            sys.exit(f"missing {need} embeddings for {args.backbones} in {args.emb_root} (run scripts/embed_many.py)")
    rng = np.random.default_rng(args.seed)

    # same random choices for every feature set (and the same order of draws as split_v1)
    hold = rng.random(len(E.meta("nct_100k")["labels"])) < 0.10
    st_y = E.meta("starc9_train")["labels"]
    other_idx = np.concatenate([rng.choice(np.flatnonzero(st_y == c), 1000, replace=False) for c in range(N)])

    tests = tissue_tests(E, args.spider_colon_manifest)
    kather = None
    if E.has("kather2016"):
        km = E.meta("kather2016")
        kather = {"labels": np.array(km["label_names"])[km["labels"]], "patients": kather_patients(km["paths"])}
        print(f"Kather 2016 patients: {sorted(set(kather['patients']))}")

    meta = {"date": str(date.today()), "backbones": args.backbones}

    if "tissue" in args.parts:
        results = {**meta, "tissue8": {}, "zero_shot": {}}
        # zero-shot reference rows
        zs_probs = {}
        for b in args.backbones:
            zs_probs[b] = {t: zero_shot(E, b, src, X_fn, y) for t, (X_fn, y, src) in tests.items()}
            if kather:
                zs_probs[b]["kather2016"] = zero_shot_probs(E.X(b, "kather2016"), E.npz(b, "kather2016"))
        if len(args.backbones) == 2:
            zs_probs["avg"] = {t: (zs_probs["conch"][t] + zs_probs["musk"][t]) / 2 for t in zs_probs["conch"]}
        for key, by_test in zs_probs.items():
            r = {}
            for t, p in by_test.items():
                if t == "kather2016":
                    r[t] = kather_score(kather["labels"], p.argmax(1), kather["patients"], np.random.default_rng(1))
                else:
                    r[t] = score(tests[t][1], p.argmax(1), p)
            results["zero_shot"][key] = r

        # trained tissue models
        for f in feature_sets:
            print(f"\n== tissue8 [{f}]", flush=True)
            clf, C = train_tissue8(E, f, hold, args.Cs)
            joblib.dump({"clf": clf, "features": f, "classes": CLASSES, "C": C, "date": str(date.today())},
                        out / f"tissue8_{f}.joblib")
            r = {}
            for t, (X_fn, y, _) in tests.items():
                p = proba(clf, X_fn(f), N)
                r[t] = score(y, p.argmax(1), p)
                save_confusion(y, p.argmax(1), sorted(set(y)), list(range(N)), out / f"confusion_{f}_{t}",
                               f"tissue8 [{f}] on {t}")
            if kather:
                p = proba(clf, E.X(f, "kather2016"), N)
                r["kather2016"] = kather_score(kather["labels"], p.argmax(1), kather["patients"], np.random.default_rng(1))
            results["tissue8"][f] = {"C": C, "tests": r}
            print(f"  tissue8[{f}] C={C}: " + ", ".join(f"{t} {v['accuracy']:.3f}" for t, v in r.items()), flush=True)
        (out / "metrics_tissue.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if "structures" in args.parts:
        manifests = {"colon": args.spider_colon_manifest, "skin": args.spider_skin_manifest}
        have_spider = (all(E.has(f"spider_{o}_train") for o in manifests)
                       and all(Path(m).exists() for m in manifests.values()))
        if not have_spider:
            print("structures skipped: SPIDER embeddings or manifests are missing")
            return
        results = {**meta, "structures": {}}
        for f in feature_sets:
            print(f"\n== structures [{f}]", flush=True)
            sclf, sC = train_structures(E, f, other_idx, manifests, args.Cs)
            joblib.dump({"clf": sclf, "labels": STRUCT, "features": f, "C": sC, "date": str(date.today())},
                        out / f"structures_{f}.joblib")
            results["structures"][f] = {"C": sC, **test_structures(E, f, sclf, manifests)}
        (out / "metrics_structures.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
