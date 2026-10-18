import json
from pathlib import Path
import pandas as pd
R = Path("/work/hdd/bfqi/natmakur")
CL = ["tumor","normal_mucosa","muscle","fat","lymphocytes","mucus","necrosis","stroma"]
SETS = {"calib": ("results_v4","results_v5","reports/overnight_v4/labels_claude_v4.csv"),
        "holdout": ("results_holdout_v1","results_holdout_v2","reports/overnight_v4/holdout/labels_claude_holdout.csv")}
out = {}
for name,(old,new,lab) in SETS.items():
    fr = {"old": [], "new": []}
    for f in sorted((R/new).glob("TCGA-*/summary.json")):
        for k,base in (("old",old),("new",new)):
            s = json.loads((R/base/f.parent.name/"summary.json").read_text())
            fr[k].append({c: s.get("tissue_fractions",{}).get(c) for c in CL})
    res = {"fractions_mean": {c: [round(pd.DataFrame(fr["old"])[c].mean()*100,1), round(pd.DataFrame(fr["new"])[c].mean()*100,1)] for c in CL}}
    t = pd.read_csv(lab); t["mine"] = t.mine.astype(str)
    parts = []
    for slide,g in t.groupby("slide"):
        p = R/new/slide/"tiles.csv"
        if p.exists():
            parts.append(g.merge(pd.read_csv(p, usecols=["row","col","qc_ok"]).rename(columns={"qc_ok":"new_ok"}), on=["row","col"]))
    t = pd.concat(parts); t = t[t.mine != "unclear"]
    acc = {}
    for c in CL:
        for k,ok in (("old", t.qc_ok.astype(bool)), ("new", t.new_ok.astype(bool))):
            d = t[ok & (t.label == c)]
            acc.setdefault(c, []).append(f"{int((d.mine==c).sum())}/{len(d)}")
    for k,ok in (("old", t.qc_ok.astype(bool)), ("new", t.new_ok.astype(bool))):
        d = t[ok]; acc.setdefault("ALL kept tiles", []).append(f"{int((d.mine==d.label).sum())}/{len(d)}")
    res["map_label_right"] = acc
    out[name] = res
print(json.dumps(out, indent=1))
