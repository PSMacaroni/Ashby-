"""
Makes the DEMO slides in app/demo_data/ from the team's fake slides (src/fake_slide.py), then runs
the team's analysis on them (src/analysis.py). So a demo slide has exactly the files a real slide
will have: tiles.csv, phrase_scores.csv, summary.json, tile_analysis.csv and thumbnail.png.

The one difference: a demo slide has no real scan, so its "thumbnail" is a plain, neutral outline of
the tissue (no fake H&E colors). Real slides show their real scan under the map. Every number in a
demo slide is made up; the app marks them DEMO.

    python app/make_demo_data.py

The app runs this by itself the first time (when app/demo_data/ is empty).
"""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import CLASSES, DEMO_DIR

DEMOS = {   # variant: (name on the card, date entered)
    "infiltrative": ("Demo · infiltrative tumor", "2026-10-17"),
    "pushing": ("Demo · pushing tumor", "2026-10-17"),
    "normal": ("Demo · normal colon wall", "2026-10-16"),
}
PX_PER_TILE = 10          # thumbnail pixels per tile
DEMO_VERSION = 3          # bump to rebuild existing demo slides when this file changes
GLASS = -1
K = {c: i for i, c in enumerate(CLASSES)}

# Rough H&E colors (RGB) per class, for the painted thumbnail and the fake tile pictures
HE_COLORS = {
    "tumor": (118, 58, 140), "normal_mucosa": (165, 95, 165), "muscle": (206, 96, 138),
    "fat": (240, 228, 234), "lymphocytes": (78, 48, 128), "mucus": (192, 192, 218),
    "necrosis": (202, 150, 172), "stroma": (228, 166, 190),
}
GLASS_RGB = (244, 243, 246)
NUCLEI = {"tumor": 0.30, "normal_mucosa": 0.18, "muscle": 0.08, "fat": 0.01, "lymphocytes": 0.55,
          "mucus": 0.02, "necrosis": 0.15, "stroma": 0.05}   # share of pixels that are dark nuclei


def phrase_grids(scores, shape):
    """phrase_scores table -> {phrase key: [R, C] grid}, like scripts/analyze_slide.py does."""
    rows, cols = scores["row"].to_numpy(int), scores["col"].to_numpy(int)
    out = {}
    for key in scores.columns.drop(["row", "col"]):
        grid = np.zeros(shape)
        grid[rows, cols] = scores[key].to_numpy(float)
        out[key] = grid
    return out


def make_demo(variant, out_dir):
    from src.analysis import analyze
    from src.fake_slide import LEVEL0_PER_TILE, make_fake_slide
    from src.tilemap import TileMap

    tiles, scores, meta, truth = make_fake_slide(variant)
    name, entered = DEMOS[variant]
    meta.update(slide_id=f"demo_{variant}", display_name=name, case_id="DEMO", processed_at=entered,
                is_demo=True, demo_version=DEMO_VERSION, thumbnail_downsample=LEVEL0_PER_TILE / PX_PER_TILE)
    shape = tuple(meta["grid_shape"])
    tm = TileMap.from_dataframe(tiles, mpp=meta["mpp"], shape=shape)
    summary, tile_table = analyze(tm, phrase_grids(scores, shape), meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles.to_csv(out_dir / "tiles.csv", index=False)
    scores.to_csv(out_dir / "phrase_scores.csv", index=False)
    tile_table.to_csv(out_dir / "tile_analysis.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outline_thumbnail(truth["label"], np.random.default_rng(1)).save(out_dir / "thumbnail.png")
    overview_figure(tm, summary, tile_table, out_dir / "analysis_overview.png")
    return summary


def overview_figure(tm, summary, tile_table, path):
    """The team's four-panel check picture (src/viz.py): tissue map, zones + front + growth arrows,
    risk points, flags. scripts/analyze_slide.py makes the same file for real slides."""
    try:
        from src.front import find_front
        from src.viz import overview_figure as draw
        points = np.zeros(tm.shape, int)
        points[tile_table["row"], tile_table["col"]] = tile_table["risk_points"]
        draw(tm, find_front(tm), summary, points, path)
    except Exception as e:      # needs matplotlib; the app works without the picture
        print(f"[demo] no analysis_overview.png: {type(e).__name__}: {e}")


def outline_thumbnail(labels, rng):
    """A plain stand-in for the scan: the tissue's outline in neutral gray on white glass, with soft
    edges. No fake H&E colors, so nothing on a demo slide pretends to be a real scan."""
    from scipy import ndimage

    tissue = np.repeat(np.repeat(labels != GLASS, PX_PER_TILE, 0), PX_PER_TILE, 1).astype(float)
    soft = ndimage.gaussian_filter(tissue, 3)
    grain = ndimage.gaussian_filter(rng.normal(size=tissue.shape), 1.2) * 6
    gray = 250 - soft * (26 + grain)
    rgb = np.stack([gray, gray - 2 * soft, gray + 1 * soft], -1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def paint_thumbnail(labels, rng):
    """(Not used any more) a painted, H&E-looking picture; kept for fake tile previews' colors."""
    from scipy import ndimage

    lab = np.repeat(np.repeat(labels, PX_PER_TILE, 0), PX_PER_TILE, 1)
    rgb = np.empty(lab.shape + (3,))
    rgb[:] = GLASS_RGB
    dens = np.zeros(lab.shape)
    for c, color in HE_COLORS.items():
        rgb[lab == K[c]] = color
        dens[lab == K[c]] = NUCLEI[c]
    for ch in range(3):                                  # soft edges, not squares
        rgb[..., ch] = ndimage.gaussian_filter(rgb[..., ch], 3.5)
    dots = rng.random(lab.shape) < ndimage.gaussian_filter(dens, 3) * 0.6
    rgb[dots] = rgb[dots] * 0.45 + np.array([60, 30, 100]) * 0.55     # nuclei
    for ch in range(3):
        rgb[..., ch] = ndimage.gaussian_filter(rgb[..., ch], 0.6)
    rgb += rng.normal(0, 3, rgb.shape)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def fake_preview(label, seed, size=160):
    """A fake 'microscope' picture of one tile, for DEMO slides only (real ones come from Delta)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), HE_COLORS.get(label, GLASS_RGB))
    d = ImageDraw.Draw(img)
    nucleus = (70, 35, 110)

    def dot(x, y, r, color=nucleus):
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    if label in ("tumor", "normal_mucosa"):
        normal = label == "normal_mucosa"
        for _ in range(3 if normal else 5):
            cx, cy = rng.uniform(10, size - 10, 2)
            rx, ry = rng.uniform(14, 22, 2) if normal else rng.uniform(8, 24, 2)
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(236, 220, 232))   # gland opening
            for t in np.linspace(0, 2 * np.pi, 20 if normal else 30, endpoint=False):
                wobble = 0 if normal else rng.normal(0, 2.5)
                dot(cx + (rx + 3 + wobble) * np.cos(t), cy + (ry + 3 + wobble) * np.sin(t),
                    2.2 if normal else rng.uniform(2, 3.8))
    elif label == "lymphocytes":
        for _ in range(400):
            dot(*rng.uniform(0, size, 2), rng.uniform(1.6, 2.4), (60, 35, 115))
    elif label == "fat":
        for _ in range(12):
            cx, cy = rng.uniform(0, size, 2)
            r = rng.uniform(16, 30)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(248, 244, 247), outline=(214, 140, 170), width=2)
    elif label in ("muscle", "stroma"):
        base = rng.uniform(0, np.pi)
        spread = 0.1 if label == "muscle" else 0.6
        for _ in range(60):                                   # fibers
            x, y = rng.uniform(0, size, 2)
            a = base + rng.normal(0, spread)
            d.line([x - 30 * np.cos(a), y - 30 * np.sin(a), x + 30 * np.cos(a), y + 30 * np.sin(a)],
                   fill=(235, 175, 198) if label == "stroma" else (190, 80, 125), width=2)
        for _ in range(40 if label == "muscle" else 18):      # nuclei
            x, y = rng.uniform(0, size, 2)
            a = base + rng.normal(0, spread)
            d.line([x - 5 * np.cos(a), y - 5 * np.sin(a), x + 5 * np.cos(a), y + 5 * np.sin(a)],
                   fill=nucleus, width=3 if label == "muscle" else 2)
    elif label == "mucus":
        for _ in range(16):
            x, y = rng.uniform(0, size, 2)
            d.arc([x - 20, y - 10, x + 20, y + 10], rng.uniform(0, 180), rng.uniform(180, 360),
                  fill=(170, 175, 210), width=2)
        for _ in range(5):
            dot(*rng.uniform(0, size, 2), 2)
    elif label == "necrosis":
        for _ in range(30):
            dot(*rng.uniform(0, size, 2), rng.uniform(4, 9), (215, 160, 180))
        for _ in range(120):
            dot(*rng.uniform(0, size, 2), rng.uniform(0.6, 1.6))
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    arr = np.asarray(img, float) + rng.normal(0, 5, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_all():
    for variant in DEMOS:
        s = make_demo(variant, DEMO_DIR / f"demo_{variant}")
        print(f"{s['slide_id']}: {s['n_tissue_tiles']} tiles, tumor found: {s['tumor']['found']}, "
              f"{len(s['high_risk_flags'])} high-risk flags, {len(s.get('risk', {}).get('cards', []))} risk cards")


def ensure_demo_data():
    """Make the demo slides if they aren't there yet (first launch, or after an update)."""
    def current(v):
        folder = DEMO_DIR / f"demo_{v}"
        try:
            version = json.loads((folder / "summary.json").read_text(encoding="utf-8")).get("demo_version")
        except (OSError, ValueError):
            return False
        return version == DEMO_VERSION and all((folder / n).exists() for n in ("tile_analysis.csv", "analysis_overview.png"))

    if not all(current(v) for v in DEMOS):
        make_all()


if __name__ == "__main__":
    make_all()
    print("saved to", DEMO_DIR)
