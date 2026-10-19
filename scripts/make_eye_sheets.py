"""
By-eye check sheets for the tumor-front features (reports/front_check_v1): one image per slide with the thumbnail
(reported satellites S1.. and lymph nodes N1.. circled), the tissue map and zones from analysis_overview.png, the
summary's pattern / direction text, and 3 x 3-tile close-ups of each satellite and node. CPU only.

    python scripts/make_eye_sheets.py <file with one result-folder path per line, relative to R below>
"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
R = Path("/work/hdd/bfqi/natmakur"); OUT = R / "eye_check"; OUT.mkdir(exist_ok=True)
def montage(folder, r, c, k=1, px=64):
    m = Image.new("RGB", ((2 * k + 1) * px, (2 * k + 1) * px), "white")
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            p = folder / "tiles" / f"{r + dr}_{c + dc}.jpg"
            if p.exists():
                m.paste(Image.open(p).convert("RGB").resize((px, px)), ((dc + k) * px, (dr + k) * px))
    ImageDraw.Draw(m).rectangle([k * px, k * px, (k + 1) * px - 1, (k + 1) * px - 1], outline="yellow")
    return m
for i, line in enumerate(open(sys.argv[1]).read().split(), 1):
    folder = R / line; s = json.loads((folder / "summary.json").read_text())
    thumb = Image.open(folder / "thumbnail.png").convert("RGB"); ds = s["thumbnail_downsample"]
    sats = [x for x in s.get("satellites", []) if x.get("satellite")][:4]; nodes = s.get("lymph_nodes", [])[:2]
    d = ImageDraw.Draw(thumb)
    for tag, items, col in (("S", sats, (0, 90, 255)), ("N", nodes, (0, 160, 0))):
        for j, x in enumerate(items, 1):
            cx, cy = (x["x"] + 112) / ds, (x["y"] + 112) / ds; rad = max(8, thumb.width / 60)
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, width=3); d.text((cx + rad, cy - rad), f"{tag}{j}", fill=col)
    thumb.thumbnail((760, 600))
    ov = Image.open(folder / "analysis_overview.png").convert("RGB"); ov = ov.crop((0, 0, ov.width, int(ov.height * 0.49)))
    ov.thumbnail((1300, 600))
    closeups = [montage(folder, x["row"], x["col"]) for x in sats + nodes]
    W = thumb.width + ov.width + 10; H = max(thumb.height, ov.height) + 70 + (200 if closeups else 0)
    sheet = Image.new("RGB", (W, H), "white"); sheet.paste(thumb, (0, 0)); sheet.paste(ov, (thumb.width + 10, 0))
    t = ImageDraw.Draw(sheet); y = max(thumb.height, ov.height) + 5
    inv = s["invasion"]; dirn = (inv.get("direction") or {}).get("text", "")[:150]
    t.text((5, y), f"#{i} {s['slide_id']}  pT {s.get('pT')}  |  {s['border']['pattern_text'][:140]}", fill="black")
    t.text((5, y + 18), f"direction: {dirn}", fill="black")
    t.text((5, y + 36), f"satellites {len([x for x in s.get('satellites', []) if x.get('satellite')])}, lymph nodes {len(s.get('lymph_nodes', []))}, depth: {inv.get('deepest_layer')}", fill="black")
    for j, m in enumerate(closeups):
        tag = f"S{j + 1}" if j < len(sats) else f"N{j - len(sats) + 1}"
        sheet.paste(m, (5 + j * 205, y + 62)); t.text((5 + j * 205, y + 62 + 193), tag, fill="black")
    sheet.save(OUT / f"eye_{i:02d}.jpg", quality=85)
    print(i, s["slide_id"], len(sats), len(nodes))
