"""
Calibrate the analysis cut-offs that have a known answer to check against: the depth hint and the serosal flag,
scored against each slide's pathologic T stage (pT, from TCGA). Re-run whenever more labeled slides have been
processed: the cut-offs are re-tuned on the bigger set, so they get better as labeled slides accumulate.

pT is the deepest invasion in the WHOLE surgical specimen; a slide is one section, so it can only show as deep or
shallower. Scoring follows that:
  depth level 1 = muscle wall present but not reached, 2 = reaches the muscle wall (>= pT2), 3 = touches fat (>= pT3)
  correct   predicted level == min(pT, 3)
  overcall  predicted level > pT (claims deeper than the truth): a real error, weighted x2
  undercall predicted level < min(pT, 3): often just sampling (deepest point not in this section), weighted x0.5
  not_assessable (no wall / fat in the section): no credit, no penalty; coverage is reported
Serosal flag: positive = pT4 (T4 / T4a / T4b); tuned to avoid false alarms first (each false alarm costs 2).

Honest accuracy comes from leave-one-slide-out: for every slide, choose the cut-offs on all the OTHER slides, then
score that slide. The chosen values are saved to calibration/calibration_<version>.json; --apply also writes them
into src/cutoffs.py.

    python scripts/calibrate_analysis.py --results data/results --version v1 [--apply]
"""
import argparse
import itertools
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import cutoffs
from src.features import DEPTH_PARAMS, SEROSA_PARAMS, depth_and_fat, outer_edge
from src.front import find_front
from src.tilemap import TileMap

DEPTH_GRID = {"WALL_MUSCLE_MIN_MM2": [0.5, 1.0, 2.0, 4.0, 8.0],
              "WALL_FAT_MIN_MM2": [0.25, 0.5, 1.0, 2.0],
              "DEPTH_CONTACT_MIN_TILES": [1, 2, 3, 5, 8],
              "DEPTH_CONTACT_RADIUS_TILES": [1, 2]}
SEROSA_GRID = {"SEROSA_NEEDS_WALL": [True, False],
               "SEROSA_MIN_RUN_UM": [0, 250, 500, 1000, 2000],
               "SEROSA_MAX_FROM_WALL_UM": [500, 1000, 1500, 3000, 1e9]}


def t_level(pt):
    """'T3a' -> 3, 'Tis' -> 1 (treated with T1: both stop above the muscle wall), None if unknown."""
    if not isinstance(pt, str):
        return None
    if pt.startswith("Tis"):
        return 1
    m = re.match(r"T(\d)", pt)
    return int(m.group(1)) if m else None


def grid(spec):
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(spec[k] for k in keys))]


def depth_outcome(pred, truth):
    if pred is None:
        return "not_assessable"
    want = min(truth, 3)
    return "correct" if pred == want else "overcall" if pred > truth else "undercall"


def depth_score(outcomes):
    w = {"correct": 1.0, "overcall": -2.0, "undercall": -0.5, "not_assessable": 0.0}
    return sum(w[o] for o in outcomes)


def serosa_score(flags, truths):
    tp = sum(f and t for f, t in zip(flags, truths))
    fp = sum(f and not t for f, t in zip(flags, truths))
    return tp - 2.0 * fp


def summarize_depth(outcomes):
    n = len(outcomes)
    k = {o: sum(x == o for x in outcomes) for o in ("correct", "overcall", "undercall", "not_assessable")}
    assessable = n - k["not_assessable"]
    return {**k, "slides": n, "assessable": assessable,
            "correct_of_assessable": round(k["correct"] / assessable, 3) if assessable else None,
            "overcall_of_assessable": round(k["overcall"] / assessable, 3) if assessable else None}


def summarize_serosa(flags, truths):
    tp = sum(f and t for f, t in zip(flags, truths)); fp = sum(f and not t for f, t in zip(flags, truths))
    pos = sum(truths); neg = len(truths) - pos
    return {"flagged": int(sum(flags)), "true_pT4_flagged": int(tp), "false_alarms": int(fp),
            "sensitivity": round(tp / pos, 3) if pos else None, "specificity": round((neg - fp) / neg, 3) if neg else None,
            "pT4_slides": int(pos), "other_slides": int(neg)}


def learning_curve(outcome, slides, combos, png, n_orders=50, seed=0):
    """
    "Predict first, then learn", in many random slide orders: slide k is predicted with the cut-offs tuned on the
    k-1 slides before it (it has never been seen), then added to the pool. Returns, for each pool size, the share of
    assessable predictions that were correct / overcalls, averaged over the orders, and draws it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(slides)
    rng = np.random.default_rng(seed)
    w = {"correct": 1.0, "overcall": -2.0, "undercall": -0.5, "not_assessable": 0.0}
    table = np.array([[w[o] for o in row] for row in outcome])          # [combos, slides] score per slide
    right = np.zeros(n); over = np.zeros(n); seen = np.zeros(n)
    for _ in range(n_orders):
        order = rng.permutation(n)
        running = np.zeros(len(combos))
        for k, j in enumerate(order):
            if k:                                                        # slide 0 has no history to tune on
                o = outcome[int(running.argmax())][j]
                if o != "not_assessable":
                    seen[k] += 1; right[k] += o == "correct"; over[k] += o == "overcall"
            running += table[:, j]
    ks = np.arange(1, n)
    acc = np.divide(right[1:], seen[1:], out=np.full(n - 1, np.nan), where=seen[1:] > 0)
    ovr = np.divide(over[1:], seen[1:], out=np.full(n - 1, np.nan), where=seen[1:] > 0)
    window = max(3, n // 10)                                             # smooth for display
    smooth = lambda a: np.convolve(np.nan_to_num(a), np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, acc, color="#1f77b4", alpha=0.25)
    ax.plot(ks, ovr, color="#d62728", alpha=0.25)
    ax.plot(ks[window - 1:], smooth(acc), color="#1f77b4", label="correct (of assessable slides)")
    ax.plot(ks[window - 1:], smooth(ovr), color="#d62728", label="overcall: deeper than pT")
    ax.set_xlabel("slides already used for tuning")
    ax.set_ylabel(f"share of the next slide's predictions\n(averaged over {n_orders} random orders)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Depth hint: predict first, then learn ({n} TCGA slides)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    png.parent.mkdir(exist_ok=True)
    fig.savefig(png, dpi=120)
    plt.close(fig)
    step = max(1, (n - 1) // 10)
    return {"pool_size": ks[::step].tolist(), "correct": np.round(acc[::step], 3).tolist(),
            "overcall": np.round(ovr[::step], 3).tolist(), "orders": n_orders, "figure": str(png)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="data/results", help="folder of slide result folders (tiles.csv + summary.json)")
    ap.add_argument("--version", required=True)
    ap.add_argument("--apply", action="store_true", help="write the chosen values into src/cutoffs.py")
    args = ap.parse_args()

    slides = []
    for d in sorted(Path(args.results).glob("*")):
        if not (d / "tiles.csv").exists() or not (d / "summary.json").exists():
            continue
        meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        truth = t_level(meta.get("pT"))
        if truth is None:
            continue
        tm = TileMap.from_csv(d / "tiles.csv", mpp=meta.get("mpp", 0.5), shape=tuple(meta["grid_shape"]))
        fr = find_front(tm)
        if fr is None:
            continue
        slides.append({"id": meta["slide_id"], "pT": meta["pT"], "truth": truth, "tm": tm, "fr": fr})
    print(f"{len(slides)} slides with known pT: " +
          ", ".join(f"T{t}: {sum(s['truth'] == t for s in slides)}" for t in (1, 2, 3, 4)), flush=True)

    # 1. depth: every setting combination, every slide (CPU, seconds)
    combos = grid(DEPTH_GRID)
    pred = np.empty((len(combos), len(slides)), dtype=object)
    for i, p in enumerate(combos):
        for j, s in enumerate(slides):
            pred[i, j] = depth_and_fat(s["tm"], s["fr"], p)[0]["depth_level"]
    outcome = [[depth_outcome(pred[i, j], s["truth"]) for j, s in enumerate(slides)] for i in range(len(combos))]
    scores = np.array([depth_score(o) for o in outcome])
    best = int(scores.argmax())
    current = {k: getattr(cutoffs, k) for k in DEPTH_PARAMS}
    cur_i = next((i for i, p in enumerate(combos) if p == current), None)
    loo = []
    for j in range(len(slides)):       # leave one slide out: tune on the others, score this one
        others = [depth_score([o for jj, o in enumerate(outcome[i]) if jj != j]) for i in range(len(combos))]
        loo.append(outcome[int(np.argmax(others))][j])

    # 2. serosa: with the chosen depth settings (they define the muscle wall / fat)
    depth_best = combos[best]
    truths4 = [s["truth"] == 4 for s in slides]
    scombos = grid(SEROSA_GRID)
    flags = [[outer_edge(s["tm"], s["fr"], {**depth_best, **p})[0]["touches"] for s in slides] for p in scombos]
    sscores = np.array([serosa_score(f, truths4) for f in flags])
    sbest = int(sscores.argmax())
    cur_s = {k: getattr(cutoffs, k) for k in SEROSA_PARAMS}
    cur_flags = [outer_edge(s["tm"], s["fr"], {**current, **cur_s})[0]["touches"] for s in slides]
    sloo = []
    for j in range(len(slides)):
        others = [serosa_score([f for jj, f in enumerate(fl) if jj != j], [t for jj, t in enumerate(truths4) if jj != j])
                  for fl in flags]
        sloo.append(flags[int(np.argmax(others))][j])

    curve = learning_curve(outcome, slides, combos, Path("calibration") / f"learning_curve_{args.version}.png")

    res = {"version": args.version, "date": str(date.today()), "slides": [s["id"] for s in slides],
           "learning_curve": curve,
           "depth": {"chosen": depth_best, "on_all_slides": summarize_depth(outcome[best]),
                     "leave_one_out": summarize_depth(loo),
                     "current_values": current,
                     "current_on_all_slides": summarize_depth(outcome[cur_i]) if cur_i is not None else None,
                     "by_slide": [{"slide": s["id"], "pT": s["pT"], "predicted_level": pred[best, j],
                                   "outcome": outcome[best][j], "leave_one_out": loo[j]} for j, s in enumerate(slides)]},
           "serosa": {"chosen": scombos[sbest], "on_all_slides": summarize_serosa(flags[sbest], truths4),
                      "leave_one_out": summarize_serosa(sloo, truths4),
                      "current_values": cur_s, "current_on_all_slides": summarize_serosa(cur_flags, truths4)}}
    out = Path("calibration") / f"calibration_{args.version}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    d = res["depth"]
    print("\nDEPTH chosen:", d["chosen"])
    print("  current values on all slides:", d["current_on_all_slides"])
    print("  chosen values on all slides: ", d["on_all_slides"])
    print("  honest estimate (leave-one-slide-out):", d["leave_one_out"])
    print("\nSEROSA chosen:", res["serosa"]["chosen"])
    print("  current values:", res["serosa"]["current_on_all_slides"])
    print("  chosen values: ", res["serosa"]["on_all_slides"])
    print("  honest estimate (leave-one-slide-out):", res["serosa"]["leave_one_out"])
    print(f"\nsaved {out}")

    if args.apply:
        path = Path("src/cutoffs.py")
        text = path.read_text(encoding="utf-8")
        for k, v in {**depth_best, **scombos[sbest]}.items():
            text, n = re.subn(rf"^{k} = [^#\n]*", f"{k} = {v!r}".ljust(35), text, flags=re.M)
            assert n == 1, k
        text = re.sub(r"#     values below are .*\n",
                      f"#     values below: calibration {args.version} ({date.today()}, {len(slides)} TCGA slides; calibration/)\n",
                      text)
        path.write_text(text, encoding="utf-8")
        print("wrote the chosen values into src/cutoffs.py")


if __name__ == "__main__":
    main()
