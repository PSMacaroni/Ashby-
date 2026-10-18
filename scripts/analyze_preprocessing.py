"""
Score the preprocessing test (scripts/eval_preprocessing.py) against labels by eye.

    python scripts/analyze_preprocessing.py data/preproc_test/features.csv [--extra-labels labels_fullres.csv]

--extra-labels: a CSV with tile_id, mine_full (labels made at full resolution) that replace / add to the 128 px ones.
Truth groups: "glass" = background (glass, ink, artifacts); "tissue" = any of the 8 tissue classes or blood.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TISSUE = ["tumor", "normal_mucosa", "muscle", "fat", "lymphocytes", "mucus", "necrosis", "stroma"]


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--extra-labels")
    args = ap.parse_args()
    df = pd.read_csv(args.features)
    df["truth"] = df["mine"].fillna("")
    if args.extra_labels:
        extra = pd.read_csv(args.extra_labels)[["tile_id", "mine_full"]]
        df = df.merge(extra, on="tile_id", how="left")
        df["truth"] = df.mine_full.where(df.mine_full.notna() & (df.mine_full != ""), df.truth)
    df = df[df.truth.isin(TISSUE + ["blood", "background"])].copy()
    df["glass"] = df.truth == "background"
    fat = df.truth == "fat"
    print(f"{len(df)} labeled tiles: glass {int(df.glass.sum())}, tissue {int((~df.glass).sum())} (real fat {int(fat.sum())})")

    # 1. how well does each measure separate glass from tissue, and glass from real fat?
    print("\n1. Glass vs tissue (AUROC; 0.5 = useless, 1 = perfect). Low values mean 'low = glass'.")
    for f in ("colorful", "tissue_px", "edges", "blur"):
        a_all = roc_auc_score(~df.glass, df[f])
        a_fat = roc_auc_score(~df.glass[df.glass | fat], df[f][df.glass | fat])
        print(f"   {f:10s} glass vs all tissue {a_all:.2f} | glass vs real fat {a_fat:.2f}")

    # 2. candidate rules: what they catch and what they cost
    print("\n2. Rules: glass caught / tissue lost / real fat lost")
    rules = {"current pipeline (qc_ok on these tiles)": None,
             "colorful < 0.3": df.colorful < 0.3}
    for tp in (0.2, 0.3, 0.4, 0.5):
        rules[f"tissue_px < {tp}"] = df.tissue_px < tp
    for e in (0.01, 0.02, 0.03, 0.05):
        rules[f"edges < {e}"] = df.edges < e
    for tp in (0.3, 0.4, 0.5):
        for e in (0.02, 0.03):
            rules[f"tissue_px < {tp} AND edges < {e}"] = (df.tissue_px < tp) & (df.edges < e)
    rules["ink (blue/green or black > 20%)"] = (df.ink_bluegreen > 0.2) | (df.ink_black > 0.2)
    for name, r in rules.items():
        if r is None:
            continue
        print(f"   {name:42s} glass {pct(r[df.glass].mean()):>4s} | tissue {pct(r[~df.glass].mean()):>4s}"
              f" | fat {int(r[fat].sum())}/{int(fat.sum())}")

    # 3. color normalization: does the tissue model agree with the eye more often?
    t = df[df.truth.isin(TISSUE)]
    print(f"\n3. Color normalization (tissue tiles by eye: {len(t)})")
    print(f"   re-run reproduces the pipeline label: {pct((t.label_raw == t.label).mean())}")
    for tag in ("raw", "norm"):
        print(f"   CONCH tissue model {tag:4s}: agree {pct((t[f'label_{tag}'] == t.truth).mean())} | "
              f"MUSK second opinion {pct((t[f'alt_{tag}'] == t.truth).mean())}")
    rows = []
    for c in TISSUE:
        s = t[t.truth == c]
        if len(s):
            rows.append({"class (eye)": c, "n": len(s), "raw": pct((s.label_raw == c).mean()),
                         "normalized": pct((s.label_norm == c).mean())})
    print(pd.DataFrame(rows).to_string(index=False))
    changed = t[t.label_raw != t.label_norm]
    print(f"   label changed on {len(changed)} tiles: fixed {int((changed.label_norm == changed.truth).sum())}, "
          f"broken {int((changed.label_raw == changed.truth).sum())}")

    # 4. blood check vs muscle
    mus, bl = df[df.truth == "muscle"], df[df.truth == "blood"]
    print(f"\n4. MUSK blood check (p_blood >= 0.5): muscle by eye called blood / blood by eye caught")
    for tag in ("raw", "norm"):
        print(f"   {tag:4s}: muscle {int((mus[f'p_blood_{tag}'] >= 0.5).sum())}/{len(mus)} | "
              f"blood {int((bl[f'p_blood_{tag}'] >= 0.5).sum())}/{len(bl)}")
    print(f"   rule 'never blood if the tissue model says muscle': muscle saved "
          f"{int(((mus.p_blood_raw >= 0.5) & (mus.label_raw == 'muscle')).sum())}, blood lost "
          f"{int(((bl.p_blood_raw >= 0.5) & (bl.label_raw == 'muscle')).sum())}")


if __name__ == "__main__":
    main()
