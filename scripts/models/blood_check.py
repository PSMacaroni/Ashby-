"""
Blood color check (Delta job: runs on CPU, or on a GPU if the job has one).

Our tissue map calls blood "necrosis". The models that know blood (structures_split_v1, conch_lr_starc9_lr) were
trained on STARC-9 tiles, which are Macenko color-normalized (blood looks purple there). On our raw TCGA slides
blood is bright red, and the blood detector barely fires. Is color the reason, and does normalizing our tiles to
STARC-9's colors fix it?

For each TCGA slide: tile it like scripts/process_slide.py (slide-pipeline branch: 224 px at 0.5 um/px, tissue mask
saturation > 12, grid cells >= 50% tissue) and embed every tile twice with CONCH:
  raw   as scanned
  norm  Macenko-normalized to STARC-9's stain colors. The stain colors are estimated once per slide, not per tile:
        a tile of pure blood has only one stain, so a per-tile estimate breaks on exactly the tiles we care about.
Then run on both: tissue8, structures_split_v1 (p_blood), conch_lr_starc9_lr (8 classes + blood) and zero-shot
CONCH with a blood prompt added to our 8 classes.

    ASHBY_REPO=~/Ashby- python blood_check.py --out /work/hdd/bfqi/$USER/blood_check/run1

Writes to --out: ref.json (reference stains), starc9_check.json (models on STARC-9's own blood tiles),
starc9_colors.csv + starc9_previews.npz, summary.json, and per slide <slide_id>/: tiles.csv (colors + predictions,
raw and norm), emb_raw.npy, emb_norm.npy, previews.npz (128 px JPEGs of the tiles worth looking at), thumbnail.jpg
"""
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("ASHBY_REPO", Path(__file__).resolve().parents[2])).expanduser()
sys.path.insert(0, str(REPO))
from src import config  # noqa: E402,F401  first: applies HF_TOKEN / HF_HOME from the repo's .env

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import openslide  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.prompts import CLASSES, TEMPLATES  # noqa: E402

IO_LIGHT, BETA, ALPHA = 240.0, 0.15, 1.0   # Macenko settings (same as torchstain and the original paper code)
TILE_PX, MPP, DOWNSAMPLE, PREVIEW_PX = 224, 0.5, 64, 128
MASK_SAT = 12                               # cutoffs.TISSUE_MASK_SAT on the slide-pipeline branch
STRUCT = ["other", "nerve", "vessel", "blood"]
NAMES9 = CLASSES + ["blood"]
NECROSIS = CLASSES.index("necrosis")
BLOOD_NAMES = ["red blood cells", "blood", "hemorrhage", "blood clot", "erythrocytes", "fresh blood"]
STAT_COLS = ["colorful_raw", "r_raw", "g_raw", "b_raw", "colorful_norm", "r_norm", "g_norm", "b_norm",
             "h_amount", "e_amount", "stained_frac"]


# ---------- Macenko stain normalization ----------

def to_od(rgb):
    """RGB (0-255) -> optical density [n, 3]: how much light the stains absorbed, per pixel and channel."""
    return -np.log((np.asarray(rgb, dtype=np.float64).reshape(-1, 3) + 1.0) / IO_LIGHT)


def stained(od):
    return (od >= BETA).all(axis=1)


def fit_stains(pixels):
    """Macenko: H and E stain colors (rows, unit length in OD) + 99th-percentile amounts, from pooled RGB pixels."""
    od = to_od(pixels)
    od = od[stained(od)]
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, [1, 2]].copy()            # the two main directions of the stain colors
    if plane[:, 1].sum() < 0:                 # main direction must point into positive OD, else the angles wrap
        plane[:, 1] *= -1
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [ALPHA, 100 - ALPHA])
    v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])
    he = np.array([v_lo, v_hi]) if v_lo[0] > v_hi[0] else np.array([v_hi, v_lo])   # H absorbs more red light
    he = he / np.linalg.norm(he, axis=1, keepdims=True)
    amounts = np.linalg.pinv(he.T) @ od.T
    return he, np.percentile(amounts, 99, axis=1)


class StainNormalizer:
    """Split each pixel into H and E amounts with the slide's own stain colors, rescale the amounts to the
    reference, and repaint them with the reference stain colors."""

    def __init__(self, he_src, max_src, he_ref, max_ref):
        self.to_amounts = np.linalg.pinv(np.asarray(he_src).T)          # [2, 3]
        self.scale = (np.asarray(max_ref) / np.asarray(max_src))[:, None]
        self.he_ref = np.asarray(he_ref)

    def amounts(self, rgb):
        od = to_od(rgb)
        return self.to_amounts @ od.T, stained(od)                     # [2, n], [n]

    def __call__(self, rgb):
        amounts, _ = self.amounts(rgb)
        out = IO_LIGHT * np.exp(-(self.he_ref.T @ (amounts * self.scale)))
        return np.clip(out.T, 0, 255).reshape(np.shape(rgb)).astype(np.uint8)


# ---------- tiles ----------

def color_stats(rgb):
    """Share of colorful (stained) pixels, and the mean R, G, B of those pixels."""
    sat = np.asarray(Image.fromarray(rgb).convert("HSV"))[..., 1].reshape(-1)
    px = rgb.reshape(-1, 3)
    colorful = sat > 20
    mean = px[colorful].mean(0) if colorful.any() else px.mean(0)
    return [float(colorful.mean()), *map(float, mean)]


def jpeg(img):
    buf = io.BytesIO()
    img.resize((PREVIEW_PX, PREVIEW_PX), Image.BILINEAR).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def tile_grid(slide):
    """The same tile positions as scripts/process_slide.py on the slide-pipeline branch."""
    mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X) or 0.25)
    read_px = int(round(TILE_PX * MPP / mpp))
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((W // DOWNSAMPLE, H // DOWNSAMPLE)).convert("RGB")
    sx, sy = W / thumb.width, H / thumb.height
    mask = np.asarray(thumb.convert("HSV"))[..., 1] > MASK_SAT
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    R, C = H // read_px, W // read_px
    rows = []
    for r in range(R):
        y0 = int(r * read_px / sy)
        y1 = max(int((r + 1) * read_px / sy), y0 + 1)
        for c in range(C):
            x0 = int(c * read_px / sx)
            x1 = max(int((c + 1) * read_px / sx), x0 + 1)
            cov = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / ((y1 - y0) * (x1 - x0))
            if cov >= 0.5:
                rows.append((r, c, c * read_px, r * read_px, cov))
    tiles = pd.DataFrame(rows, columns=["row", "col", "x", "y", "mask_coverage"])
    return tiles, thumb, read_px, mpp, (R, C), (sx, sy)


def read_tile(slide, x, y, read_px):
    tile = slide.read_region((int(x), int(y)), 0, (read_px, read_px)).convert("RGB")
    return tile.resize((TILE_PX, TILE_PX), Image.BILINEAR)


class SlideTiles(Dataset):
    def __init__(self, path, xy, read_px, preprocess, normalizer):
        self.path, self.xy, self.read_px = str(path), xy, read_px
        self.preprocess, self.normalizer, self.slide = preprocess, normalizer, None

    def __len__(self):
        return len(self.xy)

    def __getitem__(self, i):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.path)                 # one handle per DataLoader worker
        raw_img = read_tile(self.slide, *self.xy[i], self.read_px)
        raw = np.asarray(raw_img)
        norm_img = Image.fromarray(self.normalizer(raw))
        amounts, st = self.normalizer.amounts(raw)
        he = amounts[:, st].mean(1) if st.any() else np.zeros(2)
        stats = color_stats(raw) + color_stats(np.asarray(norm_img)) + [float(he[0]), float(he[1]), float(st.mean())]
        return (self.preprocess(raw_img), self.preprocess(norm_img), np.array(stats, np.float32),
                jpeg(raw_img), jpeg(norm_img))


def collate(batch):
    raw, norm, stats, jpg_raw, jpg_norm = zip(*batch)
    return torch.stack(raw), torch.stack(norm), np.stack(stats), list(jpg_raw), list(jpg_norm)


def one_thread(_):
    """DataLoader workers: keep numpy's math libraries to one thread each (the main process uses the rest)."""
    from threadpoolctl import threadpool_limits
    global _LIMITS
    _LIMITS = threadpool_limits(1)


# ---------- models ----------

def load_lr(path):
    obj = joblib.load(path)
    return obj["clf"] if isinstance(obj, dict) else obj


def proba(clf, X, n):
    p = np.zeros((len(X), n))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


@torch.inference_mode()
def zero_shot_text(model):
    """Text embeddings for our 8 classes (prompts v1) plus blood, rows in NAMES9 order."""
    from conch.open_clip_custom import get_tokenizer
    from src.models import build_text_classifier, tokenize

    device = next(model.parameters()).device
    rows = build_text_classifier(model)
    tokens = tokenize(get_tokenizer(), [t.format(n) for n in BLOOD_NAMES for t in TEMPLATES]).to(device)
    blood = model.encode_text(tokens).mean(0)
    return torch.cat([rows, (blood / blood.norm())[None]]).float().cpu().numpy()


def predict(models, E):
    t8 = proba(models["tissue8"], E, 8)
    st = proba(models["structures"], E, 4)
    s9 = proba(models["starc9"], E, 9)
    logits = models["zs_scale"] * E @ models["zs_text"].T
    zs = np.exp(logits - logits.max(1, keepdims=True))
    zs /= zs.sum(1, keepdims=True)
    out = {"t8_label": np.array(CLASSES)[t8.argmax(1)], "t8_conf": t8.max(1)}
    out.update({f"t8_{c}": t8[:, i] for i, c in enumerate(CLASSES)})
    out.update({"struct_label": np.array(STRUCT)[st.argmax(1)], "struct_p_blood": st[:, 3],
                "s9_label": np.array(NAMES9)[s9.argmax(1)], "s9_p_blood": s9[:, 8],
                "zs_label": np.array(NAMES9)[zs.argmax(1)], "zs_p_blood": zs[:, 8]})
    return out


def shares(values):
    s = pd.Series(values).value_counts(normalize=True)
    return {k: round(float(v), 4) for k, v in s.items()}


# ---------- the steps ----------

def reference_stains(starc9, n_per_class, rng):
    """STARC-9's stain colors, from training tiles of the 8 tissue classes (no blood)."""
    px = []
    for c in CLASSES:
        files = sorted((starc9 / "train" / c).glob("*.png"))
        for i in rng.choice(len(files), min(n_per_class, len(files)), replace=False):
            a = np.asarray(Image.open(files[i]).convert("RGB")).reshape(-1, 3)
            px.append(a[rng.choice(len(a), min(4000, len(a)), replace=False)])
    he, max_c = fit_stains(np.concatenate(px))
    return {"he": he.tolist(), "max_c": max_c.tolist(), "from": f"STARC-9 train, {n_per_class} tiles per tissue class"}


def starc9_colors(starc9, out, rng, n=300, n_previews=48):
    """Colors of STARC-9's TCGA test tiles (already normalized) for a few classes, + previews to look at."""
    rows, previews = [], {}
    for cls in ["blood", "necrosis", "muscle", "stroma", "tumor"]:
        files = sorted((starc9 / "val_tcga" / cls).glob("*.png"))
        pick = rng.choice(len(files), min(n, len(files)), replace=False)
        previews[cls] = np.array([jpeg(Image.open(files[i]).convert("RGB")) for i in pick[:n_previews]], dtype=object)
        for i in pick:
            rows.append([cls, files[i].name, *color_stats(np.asarray(Image.open(files[i]).convert("RGB")))])
    pd.DataFrame(rows, columns=["label", "file", "colorful", "r", "g", "b"]).round(4).to_csv(
        out / "starc9_colors.csv", index=False)
    np.savez(out / "starc9_previews.npz", **previews)


def starc9_check(models, emb_dir):
    """What each model says about STARC-9's own blood and necrosis tiles (normalized colors, new patients)."""
    out = {}
    for name in ["starc9_val_tcga", "starc9_val_large"]:
        d = np.load(emb_dir / f"{name}.npz", allow_pickle=True)
        E, y = d["embeddings"].astype(np.float32), d["labels"]
        out[name] = {}
        for label, k in (("blood", 8), ("necrosis", NECROSIS)):
            p = predict(models, E[y == k])
            out[name][label] = {
                "n": int((y == k).sum()),
                "tissue8_says": shares(p["t8_label"]),
                "structures_says_blood": round(float((p["struct_label"] == "blood").mean()), 4),
                "structures_p_blood_ge_0.9": round(float((p["struct_p_blood"] >= 0.9).mean()), 4),
                "starc9_9class_says_blood": round(float((p["s9_label"] == "blood").mean()), 4),
                "zero_shot_says": shares(p["zs_label"]),
            }
    return out


def slide_pixels(path, xy, read_px, n_tiles, rng):
    slide = openslide.OpenSlide(str(path))
    px = []
    for i in rng.choice(len(xy), min(n_tiles, len(xy)), replace=False):
        a = np.asarray(read_tile(slide, *xy[i], read_px)).reshape(-1, 3)
        px.append(a[rng.choice(len(a), 4000, replace=False)])
    return np.concatenate(px)


def run_slide(path, args, models, model, preprocess, ref, rng):
    slide_id = path.name.split(".")[0]
    out = args.out / slide_id
    out.mkdir(parents=True, exist_ok=True)
    slide = openslide.OpenSlide(str(path))
    tiles, thumb, read_px, slide_mpp, grid, (sx, sy) = tile_grid(slide)
    if args.max_tiles:
        tiles = tiles.sample(n=min(args.max_tiles, len(tiles)), random_state=0).sort_index()
    tiles = tiles.reset_index(drop=True)
    xy = tiles[["x", "y"]].to_numpy()
    thumb.save(out / "thumbnail.jpg", quality=90)
    print(f"{slide_id}: {len(tiles)} tiles, {read_px} px at level 0 ({slide_mpp} um/px)", flush=True)

    he_src, max_src = fit_stains(slide_pixels(path, xy, read_px, args.src_tiles, rng))
    normalizer = StainNormalizer(he_src, max_src, ref["he"], ref["max_c"])
    loader = DataLoader(SlideTiles(path, xy, read_px, preprocess, normalizer), batch_size=args.batch_size,
                        num_workers=args.workers, collate_fn=collate, worker_init_fn=one_thread)
    device = next(model.parameters()).device
    E_raw, E_norm, stats, jpg_raw, jpg_norm = [], [], [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for x_raw, x_norm, st, jr, jn in tqdm(loader, desc=slide_id, mininterval=30):
            x = torch.cat([x_raw, x_norm]).to(device)
            e = model.encode_image(x, proj_contrast=True, normalize=True).float().cpu().numpy()
            E_raw.append(e[:len(x_raw)])
            E_norm.append(e[len(x_raw):])
            stats.append(st)
            jpg_raw += jr
            jpg_norm += jn
    seconds = time.time() - t0
    E_raw, E_norm = np.concatenate(E_raw), np.concatenate(E_norm)
    np.save(out / "emb_raw.npy", E_raw)
    np.save(out / "emb_norm.npy", E_norm)

    stats = np.concatenate(stats)
    for j, col in enumerate(STAT_COLS):
        tiles[col] = stats[:, j]
    for tag, E in (("raw", E_raw), ("norm", E_norm)):
        for k, v in predict(models, E).items():
            tiles[f"{k}_{tag}"] = v
    tiles.round(4).to_csv(out / "tiles.csv", index=False)

    # previews: every tile some model calls necrosis or blood, plus random controls
    blood_any = tiles[[f"{m}_p_blood_{t}" for m in ("struct", "s9") for t in ("raw", "norm")]].max(axis=1) >= 0.5
    keep = ((tiles.t8_label_raw == "necrosis") | (tiles.t8_label_norm == "necrosis") | blood_any
            | (tiles.zs_label_raw == "blood") | (tiles.zs_label_norm == "blood")).to_numpy()
    sel = np.flatnonzero(keep)
    if len(sel) > args.max_previews:
        sel = np.sort(rng.choice(sel, args.max_previews, replace=False))
    rest = np.setdiff1d(np.arange(len(tiles)), sel)
    sel = np.union1d(sel, rng.choice(rest, min(150, len(rest)), replace=False))
    np.savez(out / "previews.npz", idx=sel, raw=np.array([jpg_raw[i] for i in sel], dtype=object),
             norm=np.array([jpg_norm[i] for i in sel], dtype=object))

    def n(mask):
        return int(np.asarray(mask).sum())

    res = {"slide_id": slide_id, "n_tiles": len(tiles), "read_px": read_px, "slide_mpp": slide_mpp,
           "grid_shape": list(grid), "thumbnail_px_per_level0_px": [1 / sx, 1 / sy],
           "stains": {"he": he_src.round(4).tolist(), "max_c": max_src.round(4).tolist()},
           "seconds": round(seconds, 1), "images_per_second": round(2 * len(tiles) / seconds, 1),
           "tissue8_same_label_raw_vs_norm": round(float((tiles.t8_label_raw == tiles.t8_label_norm).mean()), 4)}
    for tag in ("raw", "norm"):
        res[tag] = {
            "tissue8": tiles[f"t8_label_{tag}"].value_counts().to_dict(),
            "structures_says_blood": n(tiles[f"struct_label_{tag}"] == "blood"),
            "structures_p_blood_ge_0.9_not_fat": n((tiles[f"struct_p_blood_{tag}"] >= 0.9)
                                                    & (tiles[f"t8_label_{tag}"] != "fat")),
            "starc9_9class_says_blood": n(tiles[f"s9_label_{tag}"] == "blood"),
            "zero_shot_says_blood": n(tiles[f"zs_label_{tag}"] == "blood"),
        }
    res["raw_necrosis_becomes"] = tiles.loc[tiles.t8_label_raw == "necrosis", "t8_label_norm"].value_counts().to_dict()
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slides-dir", default="/work/hdd/bfqi/natmakur/tcga_slides")
    ap.add_argument("--slides", nargs="*", help="slide id prefixes (default: every .svs in --slides-dir)")
    ap.add_argument("--starc9", default="/work/hdd/bfqi/natmakur/datasets/starc9")
    ap.add_argument("--emb-dir", default="/work/hdd/bfqi/natmakur/embeddings/conch")
    ap.add_argument("--models", default=str(REPO / "models"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tiles", type=int, default=0, help="quick test: only this many random tiles per slide")
    ap.add_argument("--src-tiles", type=int, default=300, help="tiles per slide used to estimate its stain colors")
    ap.add_argument("--ref-tiles", type=int, default=60, help="STARC-9 tiles per class for the reference stains")
    ap.add_argument("--max-previews", type=int, default=3000)
    args = ap.parse_args()
    args.out, args.starc9, args.emb_dir, models_dir = (Path(args.out), Path(args.starc9), Path(args.emb_dir),
                                                       Path(args.models))
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"torch threads: {torch.get_num_threads()}, loader workers: {args.workers}", flush=True)

    ref = reference_stains(args.starc9, args.ref_tiles, rng)
    (args.out / "ref.json").write_text(json.dumps(ref, indent=2))
    starc9_colors(args.starc9, args.out, rng)

    from src.models import load_conch

    model, preprocess = load_conch(device=config.get_device(), image_size=TILE_PX)   # GPU if the job has one
    models = {"tissue8": load_lr(models_dir / "tissue8_split_v1.joblib"),
              "structures": load_lr(models_dir / "structures_split_v1.joblib"),
              "starc9": load_lr(models_dir / "conch_lr_starc9_lr.joblib"),
              "zs_text": zero_shot_text(model), "zs_scale": float(model.logit_scale.exp())}
    check = starc9_check(models, args.emb_dir)
    (args.out / "starc9_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check), flush=True)

    slides = sorted(Path(args.slides_dir).glob("*.svs"))
    if args.slides:
        slides = [s for s in slides if any(s.name.startswith(p) for p in args.slides)]
    summary = {"reference": ref, "starc9_check": check, "slides": []}
    for path in slides:
        summary["slides"].append(run_slide(path, args, models, model, preprocess, ref, rng))
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
