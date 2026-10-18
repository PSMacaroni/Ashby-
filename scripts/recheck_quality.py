"""
Re-apply the quality check (glass / blood / blur) to already-processed slides with the CURRENT rules in src/cutoffs.py
and src/detectors.py, using the numbers saved in tiles.csv (colorful, blur, label, p_blood, and the texture measures
edges / structure_cov / ink_* when present: older folders get them from scripts/add_texture_measures.py). CPU only,
no model, no slide file needed. Updates qc_ok / qc_reason in tiles.csv and the qc counts in summary.json; run
scripts/analyze_slide.py afterwards.

    python scripts/recheck_quality.py data/results/<slide_id> [more folders...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import cutoffs
from src.detectors import is_blood, quality_check


def recheck(folder):
    tiles = pd.read_csv(folder / "tiles.csv")
    old = tiles.qc_reason.fillna("").astype(str)
    background, blurry, blur_ref = quality_check(tiles.colorful, tiles.blur, tiles.label, texture=tiles)
    blood = (is_blood(tiles.p_blood.to_numpy(), tiles.label.to_numpy()) & ~background) if "p_blood" in tiles \
        else np.zeros(len(tiles), bool)
    blurry = blurry & ~blood
    tiles["qc_ok"] = ~(background | blood | blurry)
    tiles["qc_reason"] = np.where(background, "background", np.where(blood, "blood", np.where(blurry, "blur", "")))
    tiles.to_csv(folder / "tiles.csv", index=False)

    s = json.loads((folder / "summary.json").read_text())
    s["n_tissue_tiles"] = int(tiles.qc_ok.sum())
    s.setdefault("qc", {}).update({"background": int(background.sum()), "blood": int(blood.sum()),
                                   "blur": int(blurry.sum()), "blur_reference": round(blur_ref, 6)})
    if "blood_check" in s:
        s["blood_check"]["rule"] = (f"MUSK p_blood >= {cutoffs.BLOOD_PROB} and not labeled "
                                    f"{' or '.join(cutoffs.BLOOD_SPARE_LABELS)}")
    (folder / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    changed = int((old != tiles.qc_reason).sum())
    print(f"{folder.name}: {changed} tiles changed; now background {int(background.sum())}, blood {int(blood.sum())}, "
          f"blur {int(blurry.sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    for f in ap.parse_args().folders:
        recheck(Path(f))


if __name__ == "__main__":
    main()
