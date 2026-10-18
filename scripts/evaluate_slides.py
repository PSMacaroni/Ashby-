"""
Evaluate the analysis on processed TCGA slides with the CURRENT settings (src/cutoffs.py). Nothing is tuned here
(that's scripts/calibrate_analysis.py). CPU only.

Per slide (from summary.json, written by scripts/analyze_slide.py):
  depth hint vs real pT (exact / shallower / deeper than pT / can't judge), serosal flag vs pT4, quality-check counts,
  tissue mix, necrosis, mucin, immune, unsure share.
With --ablation, the same tiles are analyzed again WITHOUT the MUSK parts (blood tiles kept, no MUSK overrule), to show
what the MUSK blood check changed on identical maps.
With --labels (calibration/clinical_labels.csv, from scripts/pick_gdc_slides.py): mucin fraction vs TCGA's mucinous
diagnosis and the immune score vs MSI status (descriptive only: these are known associations, not tuning targets).

    python scripts/evaluate_slides.py --results data/results_v4 --labels calibration/clinical_labels.csv \
        --out reports/eval_v4 --ablation
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.analysis import analyze
from src.tilemap import TileMap


def t_level(pt):
    if not isinstance(pt, str):
        return None
    if pt.startswith("Tis"):
        return 1
    m = re.match(r"T(\d)", pt)
    return int(m.group(1)) if m else None


def outcome(level, truth):
    if level is None or truth is None or pd.isna(level) or pd.isna(truth):   # NaN after a DataFrame round trip
        return "not_assessable"
    return "exact" if level == min(truth, 3) else "deeper" if level > truth else "shallower"


def numbers(s, prefix=""):
    """The fields we compare, from one analysis summary."""
    inv, tf = s.get("invasion", {}), s.get("tissue_fractions", {})
    return {f"{prefix}depth_level": inv.get("depth_level"), f"{prefix}deepest_layer": inv.get("deepest_layer"),
            f"{prefix}wall_in_section": inv.get("wall_in_section"),
            f"{prefix}fat_contact_without_wall": inv.get("fat_contact_without_wall"),
            f"{prefix}serosa_flag": s.get("outer_edge", {}).get("touches"),
            f"{prefix}tumor_frac": tf.get("tumor"), f"{prefix}necrosis_frac": tf.get("necrosis"),
            f"{prefix}necrosis_in_tumor": s.get("necrosis", {}).get("fraction_in_tumor"),
            f"{prefix}mucin_in_tumor": s.get("mucin", {}).get("fraction_in_tumor"),
            f"{prefix}immune_margin": s.get("immune", {}).get("margin_fraction"),
            f"{prefix}immune_core": s.get("immune", {}).get("core_fraction"),
            f"{prefix}unsure": s.get("uncertainty", {}).get("fraction_unsure"),
            f"{prefix}n_flags": len(s.get("high_risk_flags", []))}


def without_musk(folder, meta):
    """Re-analyze the same tiles as if the MUSK blood check and overrule didn't exist."""
    df = pd.read_csv(folder / "tiles.csv")
    df["qc_reason"] = df.qc_reason.fillna("").astype(str)       # all-empty columns load as float
    blood = df.qc_reason == "blood"
    df.loc[blood, "qc_ok"], df.loc[blood, "qc_reason"] = True, ""
    df = df.drop(columns=[c for c in ("alt_label", "p_blood") if c in df])
    tm = TileMap.from_dataframe(df, mpp=meta.get("mpp", 0.5), shape=tuple(meta["grid_shape"]))
    return analyze(tm, None, {k: meta[k] for k in ("slide_id", "mpp", "tile_size_px", "grid_shape") if k in meta})[0]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def depth_table(df, col="depth_level"):
    out = df.apply(lambda r: outcome(r[col], r.T_level), axis=1)
    judged = out != "not_assessable"
    t34 = df.T_level >= 3
    return {"slides": int(len(df)), "judged": int(judged.sum()), "exact": int((out == "exact").sum()),
            "shallower": int((out == "shallower").sum()), "deeper_than_pT": int((out == "deeper").sum()),
            "claim_true": f"{int(judged.sum() - (out == 'deeper').sum())}/{int(judged.sum())}",
            "T3T4_reached_level3": f"{int(((df[col] == 3) & t34).sum())}/{int(t34.sum())}"}


def serosa_table(df, col="serosa_flag"):
    t4, flag = df.T_level == 4, df[col].fillna(False).astype(bool)
    return {"pT4_found": f"{int((flag & t4).sum())}/{int(t4.sum())}",
            "false_alarms": f"{int((flag & ~t4).sum())}/{int((~t4).sum())}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder of slide result folders")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in sorted(p for p in Path(args.results).iterdir() if (p / "summary.json").exists()):
        s = json.loads((folder / "summary.json").read_text())
        if "invasion" not in s:
            print(f"{folder.name}: not analyzed yet, skipped")
            continue
        qc = s.get("qc", {})
        row = {"slide": s["slide_id"], "case": s["slide_id"][:12], "pT": s.get("pT"), "T_level": t_level(s.get("pT")),
               "n_tiles_read": s.get("n_tiles_read"), "n_tissue_tiles": s.get("n_tissue_tiles"),
               "qc_background": qc.get("background"), "qc_blood": qc.get("blood"), "qc_blur": qc.get("blur"),
               **numbers(s)}
        if args.ablation:
            row.update(numbers(without_musk(folder, s), "nomusk_"))
        rows.append(row)
        print(f"{s['slide_id']}: pT {s.get('pT')}, depth level {row['depth_level']}", flush=True)
    df = pd.DataFrame(rows)
    df["outcome"] = df.apply(lambda r: outcome(r.depth_level, r.T_level), axis=1)

    result = {"depth": depth_table(df), "serosa": serosa_table(df),
              "depth_by_pT": pd.crosstab(df.pT.map(lambda p: f"T{t_level(p)}"), df.depth_level.fillna(0).astype(int)
                                         ).rename(columns={0: "cant_judge"}).to_dict(),
              "qc_totals": {k: int(df[k].fillna(0).sum()) for k in ("n_tiles_read", "n_tissue_tiles", "qc_background",
                                                                       "qc_blood", "qc_blur")},
              "means": {k: float(df[k].astype(float).mean()) for k in ("tumor_frac", "necrosis_frac", "necrosis_in_tumor",
                                                                      "mucin_in_tumor", "unsure")}}
    if args.ablation:
        df["nomusk_outcome"] = df.apply(lambda r: outcome(r.nomusk_depth_level, r.T_level), axis=1)
        result["without_musk"] = {"depth": depth_table(df, "nomusk_depth_level"),
                                  "serosa": serosa_table(df, "nomusk_serosa_flag"),
                                  "means": {k: float(df["nomusk_" + k].astype(float).mean())
                                            for k in ("necrosis_frac", "necrosis_in_tumor", "unsure")},
                                  "slides_where_depth_changed": df.slide[df.depth_level.fillna(0)
                                                                         != df.nomusk_depth_level.fillna(0)].tolist()}
    if args.labels:
        lab = pd.read_csv(args.labels)
        df = df.merge(lab[["case", "mucinous", "msi_mantis", "primary_diagnosis"]], on="case", how="left")
        muc = df.mucinous.fillna(False).astype(bool)
        msi_h = pd.to_numeric(df.msi_mantis, errors="coerce") > 0.4
        mss = pd.to_numeric(df.msi_mantis, errors="coerce") <= 0.4
        result["mucin_vs_tcga_mucinous"] = {
            "n_mucinous": int(muc.sum()), "n_other": int((~muc).sum()),
            "median_mucin_in_tumor_mucinous": float(df.mucin_in_tumor[muc].median()) if muc.any() else None,
            "median_mucin_in_tumor_other": float(df.mucin_in_tumor[~muc].median()),
            "auroc": auroc(df.mucin_in_tumor[muc], df.mucin_in_tumor[~muc])}
        result["immune_vs_msi"] = {
            "n_msi_high": int(msi_h.sum()), "n_mss": int(mss.sum()),
            "auroc_margin": auroc(df.immune_margin[msi_h], df.immune_margin[mss]),
            "auroc_core": auroc(df.immune_core[msi_h], df.immune_core[mss])}
    df.to_csv(out / "per_slide.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
