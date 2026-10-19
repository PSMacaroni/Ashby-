"""
Re-derive the four cut-offs set in reports/cutoffs_v1 from the saved labels (no pathologist on this project: the
pattern / tile labels are Claude's, by eye; mucinous is TCGA's diagnosis). Each part picks the cut-off on the
calibration slides (results_v4) and reports it on the held-out slides (results_holdout_v1).

  mucin       mucin.fraction_in_tumor vs TCGA "mucinous"          -> cutoffs.MUCIN_FLAG_FRACTION
  confidence  map accuracy by top probability (tile labels)       -> cutoffs.LOW_CONFIDENCE
  pattern     fingers + satellites / solidity vs pattern labels   -> cutoffs.INFILTRATIVE_MIN_PROTRUSIONS,
                                                                     cutoffs.SOLIDITY_INFILTRATIVE
  blood       what tiles in each p_blood band are, by eye         -> cutoffs.BLOOD_PROB

    python scripts/calibrate_cutoffs.py --results-calib data/results_v4 --results-holdout data/results_holdout_v1
    (--results-* are only needed for "mucin"; the other parts read reports/)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parent.parent / "reports"
TISSUE = {"tumor", "normal_mucosa", "muscle", "fat", "lymphocytes", "mucus", "necrosis", "stroma"}


def balanced(pred, truth):
    """(sensitivity + specificity) / 2 for two boolean arrays."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    return (pred[truth].mean() + (~pred[~truth]).mean()) / 2


def mucin(calib_dir, holdout_dir, labels):
    lab = pd.read_csv(labels)[["case", "mucinous"]]
    rows = []
    for split, root in (("calib", calib_dir), ("holdout", holdout_dir)):
        for f in sorted(Path(root).glob("TCGA-*/summary.json")):
            s = json.loads(f.read_text())
            rows.append({"split": split, "case": s["slide_id"][:12],
                         "mucin": (s.get("mucin") or {}).get("fraction_in_tumor")})
    df = pd.DataFrame(rows).merge(lab, on="case", how="left").dropna(subset=["mucin"])
    df["mucinous"] = df.mucinous.fillna(False).astype(bool)
    cal, ho = df[df.split == "calib"], df[df.split == "holdout"]
    print(f"\n== mucin: {len(cal)} calibration slides ({cal.mucinous.sum()} mucinous), {len(ho)} held-out "
          f"({ho.mucinous.sum()} mucinous)")
    for t in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5):
        line = []
        for d in (cal, ho):
            p = d.mucin >= t
            line.append(f"flags {p.sum():2d}, {(p & d.mucinous).sum():2d} mucinous, balanced acc "
                        f"{balanced(p, d.mucinous):.2f}")
        print(f"  >= {t:<5}  calib: {line[0]}  |  held-out: {line[1]}")


def confidence(calib_labels, holdout_labels):
    print("\n== LOW_CONFIDENCE: map label vs Claude's tile label (class sets, 'unclear' left out)")
    for name, path in (("calib", calib_labels), ("held-out", holdout_labels)):
        d = pd.read_csv(path)
        d = d[d.set.str.startswith("class_") & (d.mine != "unclear")]
        right = d.label == d.mine
        print(f"  {name}: {len(d)} tiles")
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            lo = d.confidence < t
            print(f"    below {t}: {lo.sum():3d} tiles, {right[lo].mean():.0%} right | at or above: {(~lo).sum():3d} "
                  f"tiles, {right[~lo].mean():.0%} right | wrong tiles marked unsure {(~right & lo).sum()}/{(~right).sum()}")


def pattern(labels):
    d = pd.read_csv(labels)
    d = d[d.mine != "cant_tell"].assign(protrusions=lambda x: x.n_fingers + x.n_satellites)
    cal, ho = d[d.split == "calib"], d[d.split == "holdout"]
    inf_c, inf_h = cal.mine == "infiltrative", ho.mine == "infiltrative"

    def rule(x, s, p):
        return (x.solidity < s) | (x.protrusions >= p)
    print(f"\n== growth pattern: {len(cal)} calibration slides ({inf_c.sum()} infiltrative), {len(ho)} held-out "
          f"({inf_h.sum()} infiltrative); 'can't tell' left out")
    grid = [(balanced(rule(cal, s, p), inf_c), s, p) for s in np.round(np.arange(0, 0.91, 0.05), 2) for p in range(1, 21)]
    best = max(g[0] for g in grid)
    print(f"  best calibration balanced accuracy {best:.2f} at (solidity, protrusions): "
          f"{[(float(s), p) for b, s, p in grid if b == best]}")
    for s, p in ((0.85, 3), (0.5, 7), (0.5, 6), (0.5, 8), (0.7, 7)):
        print(f"  solidity < {s} or protrusions >= {p}:  calib {balanced(rule(cal, s, p), inf_c):.2f} "
              f"({(rule(cal, s, p) == inf_c).sum()}/{len(cal)} right)  |  held-out {balanced(rule(ho, s, p), inf_h):.2f} "
              f"({(rule(ho, s, p) == inf_h).sum()}/{len(ho)} right)")


def blood(new_labels, old_labels, bands_csv):
    def kind(m):
        return ("blood" if m == "blood" else "tissue" if m in TISSUE or m == "tissue"
                else "ink/glass/blur" if m in ("ink", "artifact", "background") else "mixed/unclear")
    new = pd.read_csv(new_labels)
    old = pd.read_csv(old_labels)
    old = old[old.set.isin(["blood", "blood_border"])]
    d = pd.concat([new[["p_blood", "mine"]], old[["p_blood", "mine"]]], ignore_index=True)
    d["kind"] = d.mine.map(kind)
    d["band"] = pd.cut(d.p_blood, [0.3, 0.5, 0.7, 0.9, 1.01], right=False, labels=["0.3", "0.5", "0.7", "0.9"])
    share = pd.crosstab(d.band, d.kind, normalize="index")
    print(f"\n== BLOOD_PROB: {len(d)} labeled tiles (calibration slides); share of each p_blood band:")
    print((share * 100).round(0).astype(int).to_string())
    bands = pd.read_csv(bands_csv)
    for split, b in bands.groupby("split"):
        n = {k: b[f"band_{k}"].sum() for k in ("0.3", "0.5", "0.7", "0.9")}
        print(f"  {split} ({b.n_nonfat.sum():,} non-fat tiles), estimated from the shares above:")
        for t in n:
            above = [k for k in n if k >= t]
            below = [k for k in n if k < t]
            skip = sum(n[k] for k in above)
            est = {c: sum(n[k] * share.loc[k, c] for k in above) for c in share.columns}
            kept = sum(n[k] * share.loc[k, "blood"] for k in below)
            print(f"    BLOOD_PROB {t}: skip {skip:6,.0f} | blood {est['blood']:5,.0f}, real tissue {est['tissue']:5,.0f}, "
                  f"ink/glass/blur {est['ink/glass/blur']:5,.0f} | blood kept (p 0.3+) {kept:5,.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="mucin,confidence,pattern,blood")
    ap.add_argument("--results-calib", help="results_v4 folder (summary.json per slide), for mucin")
    ap.add_argument("--results-holdout", help="results_holdout_v1 folder, for mucin")
    ap.add_argument("--labels", default="calibration/clinical_labels.csv")
    args = ap.parse_args()
    parts = args.parts.split(",")
    if "mucin" in parts and args.results_calib and args.results_holdout:
        mucin(args.results_calib, args.results_holdout, args.labels)
    if "confidence" in parts:
        confidence(REPORTS / "overnight_v4/labels_claude_v4.csv", REPORTS / "overnight_v4/holdout/labels_claude_holdout.csv")
    if "pattern" in parts:
        pattern(REPORTS / "cutoffs_v1/pattern_labels_claude.csv")
    if "blood" in parts:
        blood(REPORTS / "cutoffs_v1/blood_labels_claude.csv", REPORTS / "overnight_v4/labels_claude_v4.csv",
              REPORTS / "cutoffs_v1/blood_bands.csv")


if __name__ == "__main__":
    main()
