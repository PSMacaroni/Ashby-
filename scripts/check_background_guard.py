"""
Check the background guard (src/detectors.py) on real whole slides, where partly-glass tiles at tissue edges, tears and
holes are what actually leaks through the thumbnail tissue mask. No labels needed: we count what the guard removes, what
the tissue model called those tiles, and save image sheets to check by eye.

Per slide: thumbnail tissue mask (same rule as src/slides.py), grid of 224 px tiles at 0.5 µm/px, keep cells that
are >= 50% tissue in the thumbnail (like sample_tissue_coords), then take ALL edge tiles (next to non-tissue, up to
--max-edge) plus a random sample of interior tiles. Each tile: colorful fraction, CONCH embedding -> tissue8 label,
immune detector score, guard decision.

    python scripts/check_background_guard.py --slides /work/hdd/bfqi/natmakur/tcga_slides/*.svs --run guard_v1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # first, so HF_HOME from .env is applied

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.detectors import colorful_fraction, is_background, load_immune, predict_immune
from src.models import load_conch
from src.prompts import CLASSES
from src.slides import get_mpp, get_thumbnail, open_slide, tissue_mask

DOWNSAMPLE, OUT_PX, MPP = 64, 224, 0.5


def grid(slide_path, max_edge, max_interior, seed, mask_sat=None):
    slide = open_slide(slide_path)
    mpp = get_mpp(slide) or 0.25
    read_px = int(round(OUT_PX * MPP / mpp))
    thumb = get_thumbnail(slide, DOWNSAMPLE)
    if mask_sat is None:
        mask = tissue_mask(thumb)                       # current rule: saturation > max(Otsu, 20)
    else:                                               # looser fixed rule (keeps pale tissue: submucosa, fat)
        mask = np.asarray(thumb.convert("HSV"))[..., 1].astype(float) > mask_sat
        cur = tissue_mask(thumb)
    cell = max(1, round(read_px / DOWNSAMPLE))
    R, C = mask.shape[0] // cell, mask.shape[1] // cell
    cov = mask[:R * cell, :C * cell].reshape(R, cell, C, cell).mean(axis=(1, 3))
    keep = cov >= 0.5
    cur_cov = (cur[:R * cell, :C * cell].reshape(R, cell, C, cell).mean(axis=(1, 3)) >= 0.5) if mask_sat is not None else keep
    pad = np.pad(keep, 1)
    all_nb = np.ones_like(keep)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            all_nb &= pad[1 + dr:1 + dr + R, 1 + dc:1 + dc + C]
    edge = keep & ~all_nb
    rng = np.random.default_rng(seed)
    rows = []
    for name, m, cap in [("edge", edge, max_edge), ("interior", keep & ~edge, max_interior)]:
        rr, cc = np.nonzero(m)
        idx = rng.choice(len(rr), min(cap, len(rr)), replace=False) if len(rr) else []
        rows += [{"row": int(rr[i]), "col": int(cc[i]), "x": int(cc[i] * cell * DOWNSAMPLE), "y": int(rr[i] * cell * DOWNSAMPLE),
                  "zone": name, "coverage": float(cov[rr[i], cc[i]]),
                  "new_in_loose_mask": bool(mask_sat is not None and not cur_cov[rr[i], cc[i]])} for i in idx]
    info = {"mask": "current" if mask_sat is None else f"sat>{mask_sat}", "mpp": mpp, "read_px": read_px, "grid": [R, C], "tissue_cells": int(keep.sum()), "edge_cells": int(edge.sum())}
    return pd.DataFrame(rows), info, read_px


class Tiles(Dataset):
    def __init__(self, path, df, read_px, preprocess):
        self.path, self.df, self.read_px, self.preprocess, self.slide = str(path), df, read_px, preprocess, None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self.slide is None:
            self.slide = open_slide(self.path)          # one handle per DataLoader worker
        r = self.df.iloc[i]
        tile = self.slide.read_region((int(r.x), int(r.y)), 0, (self.read_px, self.read_px)).convert("RGB")
        tile = tile.resize((OUT_PX, OUT_PX), Image.BILINEAR)
        thumb = np.asarray(tile.resize((112, 112)))
        return self.preprocess(tile), colorful_fraction(tile), torch.from_numpy(thumb.copy())


def sheet(thumbs, captions, path, cols=8):
    n = len(thumbs)
    if not n:
        return
    s = Image.new("RGB", (cols * 112, ((n + cols - 1) // cols) * 126), "white")
    d = ImageDraw.Draw(s)
    for k, (t, c) in enumerate(zip(thumbs, captions)):
        x, y = (k % cols) * 112, (k // cols) * 126
        s.paste(Image.fromarray(t), (x, y))
        d.text((x + 2, y + 113), c, fill="black")
    s.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tissue-model", default="models/tissue8_split_v1.joblib")
    ap.add_argument("--immune-model", default="models/immune_v1.joblib")
    ap.add_argument("--max-edge", type=int, default=1500)
    ap.add_argument("--max-interior", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--mask-sat", type=float, default=None, help="use a fixed saturation cut-off for the tissue mask")
    args = ap.parse_args()
    out = Path("data/results/guard_check") / args.run
    out.mkdir(parents=True, exist_ok=True)
    device = config.get_device()
    model, preprocess = load_conch(device=device, image_size=224)
    t8, imm = joblib.load(args.tissue_model), load_immune(args.immune_model)
    summary = {}
    for sp in args.slides:
        name = Path(sp).name.split(".")[0]
        df, info, read_px = grid(sp, args.max_edge, args.max_interior, seed=0, mask_sat=args.mask_sat)
        print(f"{name}: {info}, testing {len(df)} tiles", flush=True)
        loader = DataLoader(Tiles(sp, df, read_px, preprocess), batch_size=128, num_workers=args.workers)
        embs, fracs, thumbs = [], [], []
        with torch.inference_mode():
            for x, f, th in tqdm(loader, desc=name):
                embs.append(model.encode_image(x.to(device), proj_contrast=True, normalize=True).float().cpu().numpy())
                fracs.append(f.numpy())
                thumbs.append(th.numpy())
        E, df["colorful"], thumbs = np.concatenate(embs), np.concatenate(fracs), np.concatenate(thumbs)
        p = t8.predict_proba(E)
        df["label"] = np.array(CLASSES)[p.argmax(1)]
        df["confidence"] = p.max(1)
        df["p_immune"] = predict_immune(imm, E)
        df["background"] = is_background(df.colorful.to_numpy(), df.label.to_numpy())
        df.to_csv(out / f"{name}.csv", index=False)
        s = {"info": info, "tested": int(len(df)),
             "flagged_background": {w: int(df.background[df.zone == w].sum()) for w in ("edge", "interior")},
             "flagged_by_label": df[df.background].label.value_counts().to_dict(),
             "necrosis_tiles": int((df.label == "necrosis").sum()),
             "necrosis_flagged": int((df.background & (df.label == "necrosis")).sum()),
             "fat_tiles_low_color": int(((df.label == "fat") & (df.colorful < 0.10)).sum()),
             "labels_edge": df[df.zone == "edge"].label.value_counts().to_dict(),
             "labels_interior": df[df.zone == "interior"].label.value_counts().to_dict(),
             "new_tiles": int(df.new_in_loose_mask.sum()),
             "new_tiles_labels": df[df.new_in_loose_mask].label.value_counts().to_dict(),
             "new_tiles_flagged_background": int((df.new_in_loose_mask & df.background).sum())}
        summary[name] = s
        print(json.dumps(s), flush=True)
        rng = np.random.default_rng(0)
        def pick(mask, n=32):
            idx = np.where(mask)[0]
            return idx[rng.choice(len(idx), min(n, len(idx)), replace=False)] if len(idx) else idx
        for tag, idx in [("flagged", pick(df.background.to_numpy())),
                         ("near_miss", pick((~df.background & (df.colorful < 0.25) & (df.label != "fat")).to_numpy())),
                         ("necrosis_kept", pick((~df.background & (df.label == "necrosis")).to_numpy())),
                         ("fat_low_color", pick(((df.label == "fat") & (df.colorful < 0.10)).to_numpy())),
                         ("new_kept", pick((df.new_in_loose_mask & ~df.background).to_numpy())),
                         ("new_flagged", pick((df.new_in_loose_mask & df.background).to_numpy()))]:
            sheet(thumbs[idx], [f"{df.label[i][:6]} c{df.colorful[i]:.2f}" for i in idx], out / f"{name}_{tag}.jpg")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
