"""Before (results_v4 / results_holdout_v1, cutoffs_v1 code) vs after (results_v5 / results_holdout_v2, + glass/ink
texture check). Prints a JSON summary. Run from ~/Ashby-v5 after rerun_v5.sbatch."""
import json
from pathlib import Path

import pandas as pd

R = Path("/work/hdd/bfqi/natmakur")
SETS = {"calib": ("results_v4", "results_v5", "reports/overnight_v4/labels_claude_v4.csv"),
        "holdout": ("results_holdout_v1", "results_holdout_v2", "reports/overnight_v4/holdout/labels_claude_holdout.csv")}
PAT = pd.read_csv("reports/cutoffs_v1/pattern_labels_claude.csv")
GLASS = {"background"}                       # by-eye label for glass / ink / empty (128 px previews)
TISSUE = {"tumor", "stroma", "muscle", "lymphocytes", "mucus", "normal_mucosa", "fat", "necrosis"}


def slide_row(s):
    qc, b, inv, m = s.get("qc", {}), s.get("border", {}), s.get("invasion", {}), s.get("mucin", {})
    return {"pattern": b.get("pattern"), "depth": inv.get("deepest_layer"), "t_hint": inv.get("t_stage_hint"),
            "mucin_flag": bool(m.get("check_mucinous")), "qc_background": qc.get("background"),
            "qc_blood": qc.get("blood"), "qc_blur": qc.get("blur"), "tumor_frac": s.get("tissue_fractions", {}).get("tumor"),
            "n_flags": len(s.get("high_risk_flags", []) or [])}


out = {}
for name, (old, new, labels) in SETS.items():
    rows = []
    for f in sorted((R / new).glob("TCGA-*/summary.json")):
        o = R / old / f.parent.name / "summary.json"
        a, b = slide_row(json.loads(o.read_text())), slide_row(json.loads(f.read_text()))
        rows.append({"slide": f.parent.name, **{f"old_{k}": v for k, v in a.items()}, **{f"new_{k}": v for k, v in b.items()}})
    d = pd.DataFrame(rows)
    res = {"slides": len(d)}
    for k in ("qc_background", "qc_blood", "qc_blur"):
        res[k] = [int(d[f"old_{k}"].fillna(0).sum()), int(d[f"new_{k}"].fillna(0).sum())]
    res["pattern_counts"] = {"old": d.old_pattern.value_counts(dropna=False).to_dict(),
                             "new": d.new_pattern.value_counts(dropna=False).to_dict()}
    lab = PAT[(PAT.split == name) & (PAT.mine != "cant_tell")].merge(d, on="slide")
    judged_o, judged_n = lab.old_pattern.isin(["pushing", "infiltrative"]), lab.new_pattern.isin(["pushing", "infiltrative"])
    res["pattern_vs_eye"] = {"old": f"{int((lab.old_pattern == lab.mine).sum())}/{int(judged_o.sum())} judged",
                             "new": f"{int((lab.new_pattern == lab.mine).sum())}/{int(judged_n.sum())} judged",
                             "labeled_slides": len(lab)}
    res["mucin_flag"] = [int(d.old_mucin_flag.sum()), int(d.new_mucin_flag.sum())]
    res["changed"] = {k: d.slide[d[f"old_{k}"].astype(str) != d[f"new_{k}"].astype(str)].str[:12].tolist()
                      for k in ("pattern", "depth", "t_hint", "mucin_flag")}
    res["tumor_frac_mean"] = [round(float(d.old_tumor_frac.mean()), 4), round(float(d.new_tumor_frac.mean()), 4)]

    # tile level: by-eye labeled tiles, old vs new quality check
    t = pd.read_csv(labels)
    t["mine"] = t.mine.astype(str)
    new_qc = []
    for slide, g in t.groupby("slide"):
        p = R / new / slide / "tiles.csv"
        if not p.exists():
            continue
        tl = pd.read_csv(p, usecols=["row", "col", "qc_reason"])
        new_qc.append(g.merge(tl.rename(columns={"qc_reason": "new_qc"}), on=["row", "col"], how="left"))
    t = pd.concat(new_qc)
    old_bg = t.qc_reason.fillna("").ne("")
    new_bg = t.new_qc.fillna("").ne("")
    gl, ti, bl = t.mine.isin(GLASS), t.mine.isin(TISSUE), t.mine.eq("blood")
    res["tiles_by_eye"] = {
        "glass_or_ink_removed": [f"{int((old_bg & gl).sum())}/{int(gl.sum())}", f"{int((new_bg & gl).sum())}/{int(gl.sum())}"],
        "tissue_removed": [f"{int((old_bg & ti).sum())}/{int(ti.sum())}", f"{int((new_bg & ti).sum())}/{int(ti.sum())}"],
        "blood_removed": [f"{int((old_bg & bl).sum())}/{int(bl.sum())}", f"{int((new_bg & bl).sum())}/{int(bl.sum())}"],
        "new_reason_counts": t.new_qc.fillna("ok").value_counts().to_dict(),
        "tissue_newly_removed": t[ti & new_bg & ~old_bg][["slide", "row", "col", "mine", "label", "new_qc"]]
        .assign(slide=lambda x: x.slide.str[:12]).values.tolist()}
    d.to_csv(f"data/compare_v5_{name}.csv", index=False)
    out[name] = res
print(json.dumps(out, indent=1, default=str))
