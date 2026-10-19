"""
Blood check, part 2: is it color or zoom? (Delta job; GPU if the job has one.)

Part 1 (blood_check.py) normalized each slide's colors with ONE set of stain colors per slide. STARC-9 may have
normalized each tile on its own, which can turn blood purple (a tile of pure blood has one stain, so a per-tile fit
splits it into fake "H" and "E"). STARC-9 tiles were also shrunk to 128 px (64 um) and then enlarged to 224 px by
CONCH's preprocessing, so the detector saw blood cells 1.75x bigger than in our 224 px / 112 um tiles.

For a chosen subset of tiles (subset.csv: slide, i, x, y, read_px, group) this embeds four new versions:
  tilenorm        Macenko fitted on the tile itself (STARC-9 style), to STARC-9's colors
  raw_crop        center 64 um of the raw tile, enlarged like STARC-9 tiles
  slidenorm_crop  center 64 um, slide-level normalization (part 1)
  tilenorm_crop   center 64 um, per-tile normalization
and runs the same models. Writes <out>/subset_results.csv and <out>/subset_previews.npz.

    ASHBY_REPO=~/Ashby- python blood_check2.py --run <part 1 out dir> --subset subset.csv --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blood_check import (REPO, TILE_PX, StainNormalizer, config, fit_stains, jpeg, load_lr,  # noqa: E402
                         predict, read_tile, stained, to_od, zero_shot_text)

import numpy as np  # noqa: E402
import openslide  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

VARIANTS = ["tilenorm", "raw_crop", "slidenorm_crop", "tilenorm_crop"]
CROP = 128   # center 128 px of a 224 px tile at 0.5 um/px = 64 um, the size of a STARC-9 tile


def center_crop(img):
    off = (TILE_PX - CROP) // 2
    return img.crop((off, off, off + CROP, off + CROP))


class Subset(Dataset):
    def __init__(self, rows, slides_dir, slide_norm, ref, preprocess):
        self.rows, self.slides_dir, self.slide_norm = rows, slides_dir, slide_norm
        self.ref, self.preprocess, self.open = ref, preprocess, {}

    def __len__(self):
        return len(self.rows)

    def tile_norm(self, raw, fallback):
        """Macenko fitted on this tile alone; falls back to the slide's fit if the tile has too little stain."""
        od = to_od(raw)
        if stained(od).sum() < 2000:
            return fallback(raw), False
        try:
            he, mx = fit_stains(raw.reshape(-1, 3))
            if not np.all(np.isfinite(he)) or np.any(mx <= 0):
                return fallback(raw), False
            return StainNormalizer(he, mx, self.ref["he"], self.ref["max_c"])(raw), True
        except (np.linalg.LinAlgError, ValueError):
            return fallback(raw), False

    def __getitem__(self, k):
        r = self.rows[k]
        if r["slide"] not in self.open:
            self.open[r["slide"]] = openslide.OpenSlide(str(next(self.slides_dir.glob(r["slide"] + "*.svs"))))
        raw_img = read_tile(self.open[r["slide"]], int(r["x"]), int(r["y"]), int(r["read_px"]))
        raw = np.asarray(raw_img)
        slide_norm = self.slide_norm[r["slide"]]
        tn, fitted = self.tile_norm(raw, slide_norm)
        tn_img = Image.fromarray(tn)
        imgs = {"tilenorm": tn_img, "raw_crop": center_crop(raw_img),
                "slidenorm_crop": center_crop(Image.fromarray(slide_norm(raw))), "tilenorm_crop": center_crop(tn_img)}
        x = torch.stack([self.preprocess(imgs[v]) for v in VARIANTS])
        return x, fitted, jpeg(tn_img), jpeg(imgs["raw_crop"])


def collate(batch):
    x, fitted, j1, j2 = zip(*batch)
    return torch.stack(x), list(fitted), list(j1), list(j2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="part 1 output folder (ref.json, summary.json)")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slides-dir", default="/work/hdd/bfqi/natmakur/tcga_slides")
    ap.add_argument("--models", default=str(REPO / "models"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    run, out, models_dir = Path(args.run), Path(args.out), Path(args.models)
    out.mkdir(parents=True, exist_ok=True)

    ref = json.loads((run / "ref.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    slide_norm = {s["slide_id"][:12]: StainNormalizer(s["stains"]["he"], s["stains"]["max_c"], ref["he"], ref["max_c"])
                  for s in summary["slides"]}
    sub = pd.read_csv(args.subset)
    rows = sub.to_dict("records")

    from src.models import load_conch

    device = config.get_device()
    model, preprocess = load_conch(device=device, image_size=TILE_PX)
    models = {"tissue8": load_lr(models_dir / "tissue8_split_v1.joblib"),
              "structures": load_lr(models_dir / "structures_split_v1.joblib"),
              "starc9": load_lr(models_dir / "conch_lr_starc9_lr.joblib"),
              "zs_text": zero_shot_text(model), "zs_scale": float(model.logit_scale.exp())}

    loader = DataLoader(Subset(rows, Path(args.slides_dir), slide_norm, ref, preprocess),
                        batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    embs, fitted, prev_tn, prev_crop = [], [], [], []
    with torch.inference_mode():
        for x, f, j1, j2 in loader:
            n, v = x.shape[:2]
            e = model.encode_image(x.view(n * v, *x.shape[2:]).to(device), proj_contrast=True, normalize=True)
            embs.append(e.float().cpu().numpy().reshape(n, v, -1))
            fitted += f
            prev_tn += j1
            prev_crop += j2
            print(f"{len(fitted)}/{len(rows)}", flush=True)
    E = np.concatenate(embs)
    np.save(out / "subset_emb.npy", E)
    sub["tile_fit_ok"] = fitted
    for j, v in enumerate(VARIANTS):
        for k, val in predict(models, E[:, j]).items():
            if not k.startswith("t8_") or k in ("t8_label", "t8_conf"):
                sub[f"{k}_{v}"] = val
    sub.round(4).to_csv(out / "subset_results.csv", index=False)
    np.savez(out / "subset_previews.npz", tilenorm=np.array(prev_tn, dtype=object),
             raw_crop=np.array(prev_crop, dtype=object))
    print("done", flush=True)


if __name__ == "__main__":
    main()
