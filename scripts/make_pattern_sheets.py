"""
Blind sheets for labeling the growth pattern (pushing vs infiltrative) by eye: for each slide, the thumbnail cropped
around the main tumor (plus 1 mm), with the analysis's invasive-front tiles marked by small yellow dots so you know
where to look. The tool's own pattern is NOT shown. Two slides per image; index.csv maps panel -> slide. CPU only.

    python scripts/make_pattern_sheets.py <results root> <file with one result folder per line, relative to root> <out>
"""
import json, sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

root, listing, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
panels, index = [], []
for k, line in enumerate(listing.read_text().split(), 1):
    folder = root / line
    s = json.loads((folder / "summary.json").read_text())
    ds, mpp = s["thumbnail_downsample"], s.get("slide_mpp") or 0.25
    tile0 = s["tile_size_px"] * s["mpp"] / mpp                       # level-0 px per tile
    tiles = pd.read_csv(folder / "tiles.csv", usecols=["row", "col", "x", "y"])
    ta = pd.read_csv(folder / "tile_analysis.csv", usecols=["row", "col", "border", "in_main_tumor"]).merge(tiles, on=["row", "col"])
    main = ta[ta.in_main_tumor.astype(bool)]
    margin = 1000 / mpp
    x0, y0 = (main.x.min() - margin) / ds, (main.y.min() - margin) / ds
    x1, y1 = (main.x.max() + tile0 + margin) / ds, (main.y.max() + tile0 + margin) / ds
    thumb = Image.open(folder / "thumbnail.png").convert("RGB")
    d = ImageDraw.Draw(thumb)
    for r in ta[ta.border == "front"].itertuples():
        cx, cy = (r.x + tile0 / 2) / ds, (r.y + tile0 / 2) / ds
        d.ellipse([cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5], fill=(255, 220, 0))
    crop = thumb.crop((max(0, int(x0)), max(0, int(y0)), min(thumb.width, int(x1)), min(thumb.height, int(y1))))
    crop.thumbnail((640, 480))
    panel = Image.new("RGB", (640, 500), "white")
    panel.paste(crop, (0, 20))
    ImageDraw.Draw(panel).text((4, 3), f"#{k}", fill="black")
    panels.append(panel)
    index.append({"id": k, "folder": line, "slide": s["slide_id"], "tool_pattern": s["border"]["pattern"],
                  "solidity": s["border"]["solidity"], "n_fingers": s["border"]["n_fingers"],
                  "n_satellites": s["border"]["n_satellites_within_1mm"]})
for i in range(0, len(panels), 2):
    sheet = Image.new("RGB", (1290, 500), "white")
    for j, p in enumerate(panels[i:i + 2]):
        sheet.paste(p, (j * 650, 0))
    sheet.save(out / f"pattern_{i // 2 + 1:02d}.jpg", quality=85)
pd.DataFrame(index).to_csv(out / "index.csv", index=False)
print(len(panels), "slides on", (len(panels) + 1) // 2, "sheets")
