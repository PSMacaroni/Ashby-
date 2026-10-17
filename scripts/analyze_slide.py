"""
Run the slide analysis (sections 3 and 4 of the plan) on one slide's result folder. CPU only: no GPU,
no model. Works on a laptop once the folder has been copied down from Delta.

    python scripts/analyze_slide.py data/results/<slide_id>

Reads   tiles.csv          (required: the tile map, see CLAUDE.md data contract)
        summary.json       (optional: slide info such as mpp; analysis fields get added to it)
        phrase_scores.csv  (optional: text-phrase scores for the flags, from scripts/score_phrases.py)
Writes  summary.json       (slide info + everything the summary panel shows)
        tile_analysis.csv  (per tile: zone, border type, distance to front, risk points, unsure)
        analysis_overview.png  (four panels to check the result by eye)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: F401  (reads .env)

import numpy as np
import pandas as pd

from src.analysis import analyze
from src.front import find_front
from src.tilemap import TileMap


def load_scores(path, tm):
    """phrase_scores.csv -> {phrase key: [R, C] grid} (0 where there's no tile)."""
    df = pd.read_csv(path)
    rows, cols = df["row"].to_numpy(int), df["col"].to_numpy(int)
    scores = {}
    for key in df.columns.drop(["row", "col"]):
        grid = np.zeros(tm.shape)
        grid[rows, cols] = df[key].to_numpy(float)
        scores[key] = grid
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="data/results/<slide_id>")
    ap.add_argument("--no-figure", action="store_true", help="skip analysis_overview.png")
    args = ap.parse_args()

    folder = Path(args.folder)
    meta_path = folder / "summary.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"slide_id": folder.name}
    shape = tuple(meta["grid_shape"]) if meta.get("grid_shape") else None
    tm = TileMap.from_csv(folder / "tiles.csv", mpp=meta.get("mpp", 0.5), shape=shape)

    scores = None
    if (folder / "phrase_scores.csv").exists():
        scores = load_scores(folder / "phrase_scores.csv", tm)
    else:
        print("no phrase_scores.csv: text-based flags will be skipped")

    summary, tile_table = analyze(tm, scores, meta)
    meta_path.write_text(json.dumps(summary, indent=2))
    tile_table.to_csv(folder / "tile_analysis.csv", index=False)

    if not args.no_figure:
        from src.viz import overview_figure

        points = np.zeros(tm.shape, int)
        points[tile_table["row"], tile_table["col"]] = tile_table["risk_points"]
        try:
            overview_figure(tm, find_front(tm), summary, points, folder / "analysis_overview.png")
        except Exception as e:  # a drawing problem must not lose the results
            print(f"[warning] could not draw analysis_overview.png: {type(e).__name__}: {e}")

    print_summary(summary)
    print(f"\nwrote {meta_path}, {folder / 'tile_analysis.csv'}")


def print_summary(s):
    print(f"\n== {s.get('slide_id', '')}: {s['n_tissue_tiles']} tissue tiles")
    t = s["tumor"]
    if not t["found"]:
        print("No tumor found.")
        return
    print(f"Tumor: {t['fraction_of_tissue']:.0%} of the section, {t['area_mm2']} mm2")
    print("Growth:", s["border"]["pattern_text"])
    print("Depth: ", s["invasion"]["t_stage_hint"])
    print("Direction:", s["invasion"]["direction"]["text"])
    print("Ahead: ", s["invasion"]["heading_into"]["text"])
    print("Fat:   ", s["fat"]["text"])
    print("Stroma:", s["stroma"]["text"])
    print("Necrosis:", s["necrosis"]["text"])
    print("Mucin: ", s["mucin"]["label"], *[p["text"] for p in s["mucin"]["pools"]])
    print("Immune:", s["immune"]["text"])
    for node in s["lymph_nodes"]:
        print("Node:  ", node["text"])
    print("Edge:  ", s["outer_edge"]["text"])
    print("Unsure:", s["uncertainty"]["text"])
    for f in s["flags"]:
        status = (f"{len(f['candidates'])} spot(s), {f['n_strong']} strong" if f["status"] == "ok"
                  else f"skipped ({f['why_skipped']})")
        print(f"Flag:   {f['title']}: {status} [{f['source']}]")
    for card in s["risk"]["cards"][:5]:
        print("Risk:  ", card["text"])
    print("High-risk list:")
    for line in s["high_risk_flags"]:
        print("  -", line)


if __name__ == "__main__":
    main()
