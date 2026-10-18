"""
Run the whole slide analysis on a TileMap: step 0 (src/front.py), the measurements (src/features.py),
the flags (src/flags.py) and the risk map (src/risk.py). CPU only, no model needed.

    summary, tile_table = analyze(tm, scores=None, meta=None)

summary     dict for summary.json (see CLAUDE.md data contract and docs/summary_example.json)
tile_table  pandas DataFrame for tile_analysis.csv: one row per tissue tile with its zone, border type,
            distance to the front, risk points and "unsure" flag, for the app's overlays
"""
import numpy as np
import pandas as pd

from src import features as F
from src.flags import all_flags
from src.front import BORDER_FRONT, BORDER_NAMES, ZONE_NAMES, find_front
from src.prompts import CLASSES
from src.risk import RISK_POINTS, risk_map
from src.tilemap import NO_TISSUE

ALWAYS_CAVEATS = [
    "Decision support only: not a diagnosis and not a treatment recommendation.",
    "One slide is a 2D snapshot: 'direction' means where the front faces in this section, not where it will grow.",
    "T stage needs more than one slide: the depth line is a 'consistent with' hint, never a stage.",
    "H&E shows where lymphocytes sit, not their type (CD8 vs CD4) or what they are doing.",
    "The map works at 112 µm tiles: single budding cells and tumor pieces smaller than ~0.3 mm are invisible to it.",
    "Flags are 'look here' hints from text matching and shape rules; expect false alarms.",
]
# keys this analysis writes; anything else already in summary.json (slide info from the Delta
# pipeline: slide_id, mpp, tile_size_px, thumbnail_downsample, model, ...) is kept as it is
ANALYSIS_KEYS = ("grid_shape", "n_tissue_tiles", "tile_um", "tissue_fractions", "tumor", "border", "satellites",
                 "invasion", "fat", "spread", "stroma", "necrosis", "mucin", "immune", "lymph_nodes",
                 "outer_edge", "uncertainty", "flags", "risk", "high_risk_flags", "caveats")


def analyze(tm, scores=None, meta=None):
    summary = {k: v for k, v in (meta or {}).items() if k not in ANALYSIS_KEYS}
    summary.update({"grid_shape": list(tm.shape), "n_tissue_tiles": int(tm.tissue.sum()),
                    "tile_um": tm.tile_um, "tissue_fractions": F.tissue_fractions(tm)})
    unsure_info, unsure = F.uncertainty(tm)
    fr = find_front(tm)
    nodes, node_mask = F.lymph_nodes(tm, fr)

    if fr is None:
        summary.update({
            "tumor": {"found": False, "fraction_of_tissue": 0.0, "area_mm2": 0.0, "centroid_rc": None},
            "lymph_nodes": nodes, "uncertainty": unsure_info, "flags": [], "risk": {"max_level": 0, "cards": []},
            "high_risk_flags": [],
            "caveats": ALWAYS_CAVEATS + ["No tumor found on this slide (expected for normal-tissue slides)."],
        })
        return plain(summary), _tile_table(tm, None, unsure, np.zeros(tm.shape, int))

    pieces = F.tumor_pieces(tm, fr, node_mask)
    border, growth_grids = F.growth_pattern(tm, fr, pieces)
    arcs = border["arcs"]
    invasion, fat, deep_grid = F.depth_and_fat(tm, fr)
    invasion["direction"] = F.direction(arcs, pieces)
    invasion["heading_into"], ray = F.heading_into(tm, fr, arcs)
    stroma, stroma_field = F.tumor_stroma_ratio(tm, fr)
    necrosis, rim = F.necrosis_and_rim(tm, fr)
    mucin, pool_grid = F.mucin(tm, fr)
    edge, serosal_grid = F.outer_edge(tm, fr)
    flags, flag_grids = all_flags(tm, fr, pieces, scores)

    # where each risk-map feature is
    infiltrative_arcs = np.zeros(tm.shape, bool)
    for a in arcs:
        if a["infiltrative"]:
            infiltrative_arcs |= growth_grids["arc_masks"][a["id"]]
    where = dict(flag_grids)
    where.update({
        "serosal_involvement": serosal_grid if edge["touches"] else None,
        "tumor_in_fat_over_5mm": deep_grid,
        "mucin_pool_past_muscle": pool_grid,
        "necrosis_rim": rim,
        "infiltrative_arc": infiltrative_arcs,
        "satellite": np.isin(fr.pieces, [p["id"] for p in pieces if p["satellite"]]),
        "faces_fat": fr.contact["fat"],
        "stroma_high_field": stroma_field if stroma.get("stroma_high") else None,
    })
    points, cards = risk_map(tm, fr, where)

    summary.update({
        "tumor": F.tumor_summary(tm, fr),
        "border": border,
        "satellites": pieces,
        "invasion": invasion,
        "fat": fat,
        "spread": F.sideways_spread(tm, fr),
        "stroma": stroma,
        "necrosis": necrosis,
        "mucin": mucin,
        "immune": F.immune(tm, fr, node_mask),
        "lymph_nodes": nodes,
        "outer_edge": edge,
        "uncertainty": unsure_info,
        "flags": flags,
        "risk": {"max_level": int(points.max()), "cards": cards,
                 "points_table": [{k: row[k] for k in ("key", "points", "bin", "source")} for row in RISK_POINTS]},
    })
    summary["high_risk_flags"] = _high_risk_flags(summary)
    summary["caveats"] = _caveats(summary)
    return plain(summary), _tile_table(tm, fr, unsure, points)


def plain(obj):
    """Turn numpy values (np.bool_, np.int64, arrays...) into plain Python, so json.dumps works."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and obj != obj:      # NaN -> null
        return None
    return obj


def _high_risk_flags(s):
    """Short "check this" lines for the summary panel (stage II high-risk features and friends)."""
    out = []
    if s["invasion"]["deepest_layer"] == "fat":
        out.append("Tumor touches fat: consistent with at least pT3 (check).")
    if s["invasion"].get("fat_contact_without_wall"):
        out.append("Tumor touches fat, but no muscle wall in this section: depth unknown (check the fat).")
    if s["outer_edge"]["touches"]:
        out.append("Possible serosal involvement: check (pT4a).")
    if s["border"]["pattern"] == "infiltrative":
        out.append(f"Infiltrative front ({s['border']['pattern_text']})")
    if s["fat"]["over_5mm"]:
        out.append("Tumor more than 5 mm past the muscle (check).")
    for f in s["flags"]:
        if f["status"] == "ok" and f["n_strong"] and f["key"] in ("perineural", "venous", "lymphatic"):
            out.append(f"{f['title']}: check {f['n_strong']} marked spot(s) ({f['action']})")
    deposits = [p for p in s["satellites"] if p["possible_deposit"]]
    if deposits:
        out.append(f"Possible tumor deposit: check {len(deposits)} island(s) in fat for a lymph node remnant or vessel wall (N1c).")
    if any(n["possible_metastasis"] for n in s["lymph_nodes"]):
        out.append("Possible lymph node metastasis: check.")
    if s["stroma"].get("stroma_high"):
        out.append(s["stroma"]["text"])
    budding = next((f for f in s["flags"] if f["key"] == "budding" and f["n_strong"]), None)
    if budding:
        out.append("Budding hotspot marked: count buds there.")
    return out


def _caveats(s):
    out = list(ALWAYS_CAVEATS)
    if not s["border"]["pattern_cutoff_calibrated"]:
        out.append("Pushing vs infiltrative uses a placeholder solidity cutoff: not yet calibrated with a pathologist.")
    if any(row["source"] == "PLACEHOLDER" for row in RISK_POINTS):
        out.append("Some risk-map points are placeholders until the features doc's points table is copied in.")
    skipped = [f["title"] for f in s["flags"] if f["status"] == "skipped"]
    if skipped:
        out.append("Not computed (missing inputs): " + "; ".join(skipped) + ".")
    if s["tumor"].get("n_tissue_sections", 1) > 1:
        out.append(f"This slide has {s['tumor']['n_tissue_sections']} tissue sections; the analysis covers the one "
                   "with the largest tumor.")
    out.append(s["outer_edge"]["caveat"])
    return out


def _tile_table(tm, fr, unsure, points):
    rows, cols = np.nonzero(tm.tissue)
    label = tm.label[rows, cols]
    table = pd.DataFrame({
        "row": rows, "col": cols,
        "label": [CLASSES[k] if k != NO_TISSUE else "" for k in label],
        "unsure": unsure[rows, cols],
        "risk_points": points[rows, cols],
    })
    if fr is not None:
        table["zone"] = [ZONE_NAMES[z] for z in fr.zone[rows, cols]]
        table["border"] = [BORDER_NAMES[b] for b in fr.border_type[rows, cols]]
        table["dist_to_front_um"] = np.round(fr.signed_um[rows, cols]).astype("float")
        table["in_main_tumor"] = fr.main_filled[rows, cols]
        table["on_front"] = fr.border_type[rows, cols] == BORDER_FRONT
    return table
