"""
The slide report: every number the analysis wrote to summary.json, arranged in the order of a
pathology report, plus the per-square explanation.

Rule: restate what the analysis says; never add a diagnosis or a treatment suggestion. The
sentences come from summary.json (every section has a ready "text"); this file only picks,
orders and labels them. Lumen can reword the key findings (lumen.py); this is the fallback.

A report is a list of Section(title, rows, items, note): rows are (label, value) pairs; items are
things with a place on the map [{"title", "sub", "loc": (row, col), "color"}].
"""
import re
from dataclasses import dataclass, field

import numpy as np

from common import (CLASSES, KNOWN_MIXUPS, LOOKS_LIKE, LOW_CONFIDENCE, NAMES, POINTER_COLORS, ZONE_SHORT, get, mm,
                    pct, risk_color, site_name, stage_text)


@dataclass
class Section:
    key: str
    title: str
    rows: list = field(default_factory=list)      # (label, value) or (label, value, strong)
    items: list = field(default_factory=list)     # {"title", "sub", "loc", "color"}
    note: str = ""
    open: bool = False                            # expanded by default in the app


def yes_no(v):
    return "n/a" if v is None else ("yes" if v else "no")


def num(v, digits=2):
    return "n/a" if v is None else f"{float(v):.{digits}f}"


# ---------------------------------------------------------------- key findings (the short version)

KEY_FINDINGS = [
    ("Growth pattern", "border.pattern_text"),
    ("Depth", "invasion.t_stage_hint"),
    ("Direction", "invasion.direction.text"),
    ("Ahead of the front", "invasion.heading_into.text"),
    ("Fat", "fat.text"),
    ("Immune", "immune.text"),
    ("Necrosis", "necrosis.text"),
    ("Mucin", "mucin.label"),
    ("Stroma", "stroma.text"),
    ("Outer edge", "outer_edge.text"),
    ("Unsure", "uncertainty.text"),
]


def tumor_line(s):
    if not get(s, "tumor.found", True) or not get(s, "tumor.fraction_of_tissue"):
        return "No tumor found by the map in this section."
    area = get(s, "tumor.area_mm2")
    return f"Tumor covers {pct(get(s, 'tumor.fraction_of_tissue'))} of the tissue" + (f" ({area} mm²)." if area else ".")


def key_findings(s):
    """[(label, sentence)]: one line per topic, the ones the analysis wrote."""
    rows = [("Tumor", tumor_line(s))]
    if not get(s, "tumor.found", True):
        return rows + ([("Unsure", get(s, "uncertainty.text"))] if get(s, "uncertainty.text") else [])
    for label, path in KEY_FINDINGS:
        text = get(s, path)
        if text:
            rows.append((label, text))
    return rows


def short_findings(s):
    """[(label, short value, full sentence)]: the Report tab's compact lines. Short values fit on one
    line; the full sentence from the analysis is the tooltip and is also in Details."""
    if not get(s, "tumor.found", True):
        return [("Tumor", "none found by the map", tumor_line(s)),
                ("Unsure", pct(get(s, "uncertainty.fraction_unsure")) + " of tissue", get(s, "uncertainty.text") or "")]
    b, inv, f, n, m, im, st_, o, u = (s.get(k) or {} for k in
                                      ("border", "invasion", "fat", "necrosis", "mucin", "immune", "stroma", "outer_edge", "uncertainty"))
    d, h = inv.get("direction") or {}, inv.get("heading_into") or {}
    depth = {"fat": "≥ pT3 · touches fat", "muscle": "≥ pT2 · reaches muscle"}.get(inv.get("deepest_layer"),
                                                                               inv.get("deepest_layer") or "not judged")
    pattern = (b.get("pattern") or "not judged").capitalize()
    growth = pattern if b.get("solidity") is None else (
        f"{pattern} · solidity {num(b.get('solidity'))} · {b.get('n_fingers', 0)} fingers · "
        f"{b.get('n_satellites_within_1mm', 0)} satellite{'s' if b.get('n_satellites_within_1mm', 0) != 1 else ''}")
    mix = ", ".join(f"{NAMES.get(k, k).lower()} {pct(v)}" for k, v in list((h.get("mix") or {}).items())[:3])
    nearest_fat = (h.get("nearest_um") or {}).get("fat")
    fat = f"{f.get('contact_mm', 0)} mm of front touches fat"
    if f.get("depth_past_muscle_um"):
        fat += f" · {mm(f.get('depth_past_muscle_um'))} past the muscle" + (" (> 5 mm)" if f.get("over_5mm") else "")
    necro = pct(n.get("fraction_in_tumor")) + " of tumor"
    if n.get("fraction_in_tumor"):
        necro += " · central (dead core)" if n.get("central") else " · not central"
    if n.get("rim_fraction_of_tumor"):
        necro += f" · low-oxygen rim {pct(n.get('rim_fraction_of_tumor'))}"
    mucin = f"{pct(m.get('fraction_in_tumor'), 1)} of tumor" + (" · check for mucinous features" if m.get("check_mucinous") else "")
    if m.get("pools"):
        mucin += f" · {len(m['pools'])} pools beyond the muscle"
    immune = f"{short_immune(im.get('pattern'))} · core {num(im.get('core_fraction'))} · front {num(im.get('margin_fraction'))}"
    stroma = f"{'stroma-high' if st_.get('stroma_high') else 'not stroma-high'} · {pct(st_.get('worst_field_stroma_fraction'))} in the worst field"
    edge = f"tumor {mm(o.get('distance_um'))} from the outer edge" if o.get("found") else "outer edge not found"
    return [
        ("Tumor", f"{pct(get(s, 'tumor.fraction_of_tissue'))} of tissue · {get(s, 'tumor.area_mm2', 'n/a')} mm²", tumor_line(s)),
        ("Growth pattern", growth, b.get("pattern_text") or ""),
        ("Depth hint", depth, inv.get("t_stage_hint") or ""),
        ("Direction", f"{d.get('side', 'n/a')} · {d.get('n_infiltrative_arcs', 0)} infiltrative arcs", d.get("text") or ""),
        ("Ahead of the front", mix + (f" · nearest fat {mm(nearest_fat)}" if nearest_fat else ""), h.get("text") or ""),
        ("Fat", fat, f.get("text") or ""),
        ("Immune", immune, im.get("text") or ""),
        ("Necrosis", necro, n.get("text") or ""),
        ("Mucin", mucin, m.get("label") or ""),
        ("Stroma", stroma, st_.get("text") or ""),
        ("Outer edge", edge, o.get("text") or ""),
        ("Unsure here", f"{pct(u.get('fraction_unsure'))} of tissue", u.get("text") or ""),
    ]


def plain_summary(s, name):
    """One paragraph: the template version of the slide summary (Lumen rewords this)."""
    parts = [f"{name}."] + [text for _, text in key_findings(s)]
    flags = s.get("high_risk_flags") or []
    if flags:
        parts.append("To check: " + " ".join(f if f.endswith(".") else f + "." for f in flags))
    return " ".join(parts)


def short_immune(pattern):
    """'few lymphocytes (desert-like)' -> 'desert-like'."""
    if not pattern:
        return "n/a"
    m = re.search(r"\(([^)]+)\)", pattern)
    return m.group(1) if m else pattern


def strip_cells(s, n_to_check):
    """The cells of the summary strip at the top of a slide: (label, value, tone)."""
    if not get(s, "tumor.found", True):
        return [("Tumor", "none found", "green"), ("High-risk features", str(n_to_check), "red" if n_to_check else ""),
                ("Unsure", pct(get(s, "uncertainty.fraction_unsure")), "amber")]
    pattern = get(s, "border.pattern") or "not judged"
    depth = {"fat": "≥ pT3 · touches fat", "muscle": "≥ pT2 · reaches muscle"}.get(
        get(s, "invasion.deepest_layer"), get(s, "invasion.deepest_layer") or "not judged")
    immune = short_immune(get(s, "immune.pattern"))
    return [
        ("Tumor", f"{pct(get(s, 'tumor.fraction_of_tissue'))} of tissue", ""),
        ("Growth pattern", pattern, "red" if pattern == "infiltrative" else ("green" if pattern == "pushing" else "")),
        ("Depth hint", depth, ""),
        ("Immune", immune, ""),
        ("High-risk features", str(n_to_check), "red" if n_to_check else "green"),
        ("Unsure here", pct(get(s, "uncertainty.fraction_unsure")), "amber"),
    ]


# ---------------------------------------------------------------- the full report

def sections(s):
    """Every number in summary.json, in 14 sections, in pathology-report order."""
    out = []
    if not get(s, "tumor.found", True):
        sec = Section("tumor", "Tumor")
        sec.rows = [("Found", "no"), ("Tissue sections", str(get(s, "tumor.n_tissue_sections", "n/a")))]
        return [sec, uncertainty_section(s), method_section(s), caveats_section(s)]

    # 1 tumor
    sec = Section("tumor", "Tumor")
    sec.rows = [("Share of tissue", pct(get(s, "tumor.fraction_of_tissue")), True),
                ("Area", f"{get(s, 'tumor.area_mm2', 'n/a')} mm²"),
                ("Tumor pieces", str(get(s, "tumor.n_tumor_pieces", "n/a"))),
                ("Tissue sections", f"{get(s, 'tumor.n_tissue_sections', 'n/a')} on the slide; the analysis covers "
                                    f"the one with the largest tumor")]
    cen = get(s, "tumor.centroid_rc")
    if cen:
        sec.items.append(dict(title="Tumor center", sub="", loc=(int(round(cen[0])), int(round(cen[1]))),
                              color=POINTER_COLORS["other"]))
    out.append(sec)

    # 2 growth pattern
    b = s.get("border") or {}
    arcs = b.get("arcs") or []
    sec = Section("growth", "Growth pattern at the invasive front")
    sec.rows = [("Pattern", b.get("pattern_text") or "not judged", True),
                ("Solidity", num(b.get("solidity")) + "  (1 = smooth outline)"),
                ("Front roughness", f"{num(b.get('roughness_um', b.get('irregularity')), 0)} µm"),
                ("Fractal dimension", num(b.get("fractal_dimension"), 3)),
                ("Finger protrusions", str(b.get("n_fingers", "n/a"))),
                ("Satellites within 1 mm", str(b.get("n_satellites_within_1mm", "n/a"))),
                ("Front", f"{get(s, 'spread.front_mm', 'n/a')} mm long; {sum(1 for a in arcs if a.get('infiltrative'))} "
                          f"of {len(arcs)} arcs (about 1 mm each) infiltrative")]
    if b.get("pattern_cutoff_calibrated") is False:
        sec.note = "Pushing / infiltrative cut-offs were set from by-eye calls on slide thumbnails, not by a pathologist."
    worst = sorted([a for a in arcs if a.get("infiltrative")], key=lambda a: -a.get("score", 0))[:3]
    for n, a in enumerate(worst, 1):
        r, c = a["center_rc"]
        sec.items.append(dict(title=f"Most infiltrative arc {n}", loc=(int(round(r)), int(round(c))),
                              sub=f"toward {a.get('clock', '?')} o'clock · roughness {a.get('roughness_um', '?')} µm · "
                                  f"{a.get('n_fingers_near', 0)} fingers · {a.get('n_satellites_near', 0)} satellites near",
                              color=POINTER_COLORS["other"]))
    out.append(sec)

    # 3 depth (with fat, spread along the lining, outer edge)
    inv = s.get("invasion") or {}
    touch = inv.get("border_touch_fraction") or {}
    f = s.get("fat") or {}
    sp = s.get("spread") or {}
    o = s.get("outer_edge") or {}
    sec = Section("depth", "Depth of invasion")
    sec.rows = [("Hint", inv.get("t_stage_hint") or "not judged", True),
                ("Deepest layer reached", str(inv.get("deepest_layer") or "none")),
                ("In this section", f"muscle wall {yes_no(inv.get('wall_in_section'))}, fat {yes_no(inv.get('fat_in_section'))}"),
                ("Tumor squares touching", f"muscle {inv.get('wall_contact_tiles', 'n/a')}, fat {inv.get('fat_contact_tiles', 'n/a')}, "
                                           f"beyond the wall {inv.get('beyond_wall_tiles', 'n/a')}"),
                ("Front touching", ", ".join(f"{NAMES.get(k, k).lower()} {pct(v)}" for k, v in touch.items()) or "n/a"),
                ("Fat", f.get("text") or "n/a", True),
                ("Front in contact with fat", f"{f.get('contact_mm', 'n/a')} mm ({pct(f.get('share_of_front'))} of the front)"),
                ("Depth past the muscle", mm(f.get("depth_past_muscle_um")) + (" (more than 5 mm)" if f.get("over_5mm") else "")),
                ("Spread along the lining", f"border against normal mucosa {sp.get('mucosa_border_mm', 'n/a')} mm vs "
                                            f"invasive front {sp.get('front_mm', 'n/a')} mm"),
                ("Outer edge", o.get("text") or "n/a", True),
                ("Tumor to the outer edge", f"{mm(o.get('distance_um'))}; touches it: {yes_no(o.get('touches'))}; "
                                            f"runs along it for {mm(o.get('run_on_edge_um'))}")]
    sec.note = ("A 'consistent with' hint from one section, never a stage. " + (o.get("caveat") or "")).strip()
    out.append(sec)

    # 4 direction
    d = inv.get("direction") or {}
    h = inv.get("heading_into") or {}
    sec = Section("direction", "Direction and what lies ahead")
    sec.rows = [("Direction", d.get("text") or "n/a", True),
                ("Side", f"{d.get('side', 'n/a')} (bearing {d.get('bearing_deg', 'n/a')}°), {d.get('n_infiltrative_arcs', 'n/a')} infiltrative arcs"),
                ("Ahead of the front", h.get("text") or "n/a", True),
                ("Tissue mix ahead (2 mm)", ", ".join(f"{NAMES.get(k, k).lower()} {pct(v)}" for k, v in (h.get("mix") or {}).items()) or "n/a"),
                ("Nearest ahead", ", ".join(f"{NAMES.get(k, k).lower()} {mm(v)}" for k, v in (h.get("nearest_um") or {}).items()) or "n/a")]
    sec.note = "A slide is a 2D snapshot: where the front faces in this section, not where it will grow. The Zones layer draws the arrows."
    out.append(sec)

    # 5 stroma
    st_ = s.get("stroma") or {}
    sec = Section("stroma", "Stroma around the tumor")
    sec.rows = [("Tumor-stroma ratio", st_.get("text") or "n/a", True),
                ("Worst field", f"{pct(st_.get('worst_field_stroma_fraction'))} stroma in a "
                                f"{st_.get('window_diameter_um', 'n/a')} µm window; stroma-high (> 50%): {yes_no(st_.get('stroma_high'))}")]
    mat = next((fl for fl in s.get("flags") or [] if fl.get("key") == "stroma_maturity"), None)
    if mat and mat.get("measure"):
        shares = mat["measure"].get("shares") or {}
        sec.rows.append(("Stroma maturity (experimental)", mat["measure"].get("text") or ", ".join(
            f"{k.replace('_', ' ')} {pct(v)}" for k, v in shares.items())))
    if st_.get("row") is not None:
        sec.items.append(dict(title="Worst stroma field", sub=f"{pct(st_.get('worst_field_stroma_fraction'))} stroma",
                              loc=(int(st_["row"]), int(st_["col"])), color=POINTER_COLORS["other"]))
    sec.note = "H&E shows the stroma reaction, not which cells are cancer-associated fibroblasts (that needs staining)."
    out.append(sec)

    # 6 necrosis
    n = s.get("necrosis") or {}
    sec = Section("necrosis", "Necrosis and likely low oxygen")
    sec.rows = [("Summary", n.get("text") or "n/a", True),
                ("Necrosis inside the tumor", f"{pct(n.get('fraction_in_tumor'))}; central (a dead core): {yes_no(n.get('central'))}"),
                ("Rim next to necrosis", f"{pct(n.get('rim_fraction_of_tumor'))} of tumor squares (within ~200 µm; likely low oxygen)"),
                ("Lymphocytes in the rim vs rest", f"{num(n.get('lymphocytes_in_rim'))} vs {num(n.get('lymphocytes_rest_of_tumor'))}")]
    out.append(sec)

    # 7 mucin
    m = s.get("mucin") or {}
    sec = Section("mucin", "Mucin")
    sec.rows = [("Summary", m.get("label") or "n/a", True),
                ("Mucin in / attached to the tumor", f"{pct(m.get('fraction_in_tumor'))}; check for mucinous features: {yes_no(m.get('check_mucinous'))}"),
                ("Mucin pools beyond the muscle", str(len(m.get("pools") or [])))]
    for pool in sorted(m.get("pools") or [], key=lambda p: -(p.get("area_mm2") or 0))[:10]:
        sec.items.append(dict(title=pool.get("text") or "Mucin pool", sub=f"{pool.get('area_mm2', '?')} mm²",
                              loc=(int(pool["row"]), int(pool["col"])), color=POINTER_COLORS["mucin"]))
    if len(m.get("pools") or []) > 10:
        sec.note = f"Showing the 10 largest of {len(m['pools'])} pools."
    out.append(sec)

    # 8 immune
    im = s.get("immune") or {}
    side = im.get("margin_by_side") or {}
    sec = Section("immune", "Immune response (lymphocytes)")
    sec.rows = [("Summary", im.get("text") or "n/a", True),
                ("Lymphocyte score", f"core {num(im.get('core_fraction'))} · front {num(im.get('margin_fraction'))} · "
                                     f"outside {num(im.get('peritumoral_fraction'))} (mean lymphocyte-detector score per square, 0-1)"),
                ("Front, by side", ", ".join(f"{k} {num(v)}" for k, v in side.items()) or "n/a")]
    if im.get("pattern_cutoffs_calibrated") is False:
        sec.note = "The pattern cut-offs (desert / excluded / inflamed) are placeholders until a pathologist sets them."
    out.append(sec)

    # 9 islands and nodes
    sats = s.get("satellites") or []
    counted = [p for p in sats if p.get("satellite")]
    deposits = [p for p in sats if p.get("possible_deposit")]
    nodes = s.get("lymph_nodes") or []
    sec = Section("islands", "Tumor islands and lymph nodes")
    sec.rows = [("Tumor islands", f"{len(sats)} separate tumor pieces; {len(counted)} count as satellites (separated from the "
                                  f"main tumor by tissue, within 1 mm); {len(deposits)} possible deposits"),
                ("Lymph nodes", str(len(nodes)) if nodes else "none on this slide (nodes usually come on separate slides)")]
    for p in sorted(sats, key=lambda p: (not p.get("satellite"), p.get("distance_to_main_um") or 0)):
        what = "satellite" if p.get("satellite") else "island (not counted)"
        if p.get("possible_deposit"):
            what += " · possible tumor deposit: check for a node remnant or vessel wall"
        if p.get("in_lymph_node"):
            what += " · inside a lymphocyte-rich nodule"
        sec.items.append(dict(title=what.capitalize(), loc=(int(p["row"]), int(p["col"])),
                              sub=f"{p.get('diameter_um', '?')} µm across · {p.get('area_mm2', '?')} mm² · "
                                  f"{mm(p.get('distance_to_main_um'))} from the main tumor · sits in {p.get('sits_in', '?')}",
                              color=POINTER_COLORS["islands"]))
    for k, node in enumerate(nodes, 1):
        sec.items.append(dict(title=node.get("text") or f"Lymph node {k}", sub="", color=POINTER_COLORS["nodes"],
                              loc=(int(node["row"]), int(node["col"])) if node.get("row") is not None else None))
    out.append(sec)

    # 10 flags
    sec = Section("flags", "Places to check (look-here flags)")
    for fl in s.get("flags") or []:
        if fl.get("status") == "skipped":
            sec.rows.append((fl.get("title", fl.get("key")), f"not computed: {fl.get('why_skipped')}"))
            continue
        cands = fl.get("candidates") or []
        strong = [c for c in cands if c.get("strong")]
        text = f"{len(strong)} strong of {len(cands)} spots · {fl.get('source', '')} · trust: {fl.get('trust', '')}"
        if fl.get("action"):
            text += f" · {fl['action']}"
        if fl.get("note"):
            text += f" · {fl['note']}"
        sec.rows.append((fl.get("title", fl.get("key")), text, bool(strong)))
        for c in strong[:5]:
            sec.items.append(dict(title=f"{fl.get('title')}: {c.get('why', '')}", color=POINTER_COLORS["flags"],
                                  sub=f"score {c.get('score', '?')} · z {c.get('z', '?')}", loc=(int(c["row"]), int(c["col"]))))
    sec.note = "'Look here' hints from text matching, detectors and shape rules; expect false alarms."
    out.append(sec)

    # 11 risk
    r = s.get("risk") or {}
    sec = Section("risk", "Risk map")
    sec.rows = [("Regions", f"{len(r.get('cards') or [])}, highest level {r.get('max_level', 'n/a')}")]
    for c in (r.get("cards") or [])[:12]:
        sec.items.append(dict(title=f"Level {c['level']} · {c['n_tiles']} squares", sub="; ".join(c.get("features") or []),
                              loc=(int(c["row"]), int(c["col"])), color=risk_color(c["level"], max(4, r.get("max_level") or 4))))
    table = r.get("points_table") or []
    if table:
        sec.rows.append(("Points per feature (within 1 mm)", "; ".join(
            f"{row['key'].replace('_', ' ')} +{row['points']}" + (" (placeholder)" if row.get("source") == "PLACEHOLDER" else "")
            for row in table)))
    out.append(sec)

    out += [uncertainty_section(s), method_section(s), caveats_section(s)]
    return out


def uncertainty_section(s):
    u = s.get("uncertainty") or {}
    sec = Section("unsure", "Where the map is unsure")
    sec.rows = [("Summary", u.get("text") or "n/a", True),
                ("Unsure squares", f"{pct(u.get('fraction_unsure'))} of tissue (top score below {LOW_CONFIDENCE}; "
                                   f"the second model's opinion is not used)")]
    for n, reg in enumerate(u.get("regions") or [], 1):
        sec.items.append(dict(title=f"Unsure area {n} · {reg['n_tiles']} squares", sub=reg.get("why", ""),
                              loc=(int(reg["row"]), int(reg["col"])), color=POINTER_COLORS["unsure"]))
    return sec


def method_section(s):
    qc = s.get("qc") or {}
    mask = s.get("tissue_mask") or {}
    fr = s.get("tissue_fractions") or {}
    sec = Section("method", "Tissue mix and how the slide was processed")
    sec.rows = [("Tissue mix", ", ".join(f"{NAMES[c]} {pct(fr.get(c, 0))}" for c in sorted(CLASSES, key=lambda c: -(fr.get(c) or 0))))]
    sec.rows += [
        ("Slide", f"{s.get('slide_id', 'n/a')} · case {s.get('case_id', 'n/a')} · {site_name(s) or 'site n/a'} · "
                  f"TCGA report {stage_text(s) or 'not recorded'} · processed {s.get('processed_at', 'n/a')}"),
        ("Squares", f"{s.get('n_tiles_read', 'n/a'):,} read, {s.get('n_tissue_tiles', 'n/a'):,} tissue; grid "
                    f"{' x '.join(str(v) for v in s.get('grid_shape') or [])}; each square {s.get('tile_um', 'n/a')} µm "
                    f"({s.get('tile_size_px', 'n/a')} px at {s.get('mpp', 'n/a')} µm/px; slide scanned at {s.get('slide_mpp', 'n/a')} µm/px)"),
        ("Skipped by quality check", f"background / glass {qc.get('background', 'n/a')} · blood {qc.get('blood', 'n/a')} · "
                                     f"blur {qc.get('blur', 'n/a')} ({qc.get('blur_rule', 'n/a')})"),
        ("Tissue mask", f"{mask.get('rule', 'n/a')}; background guard: {mask.get('background_guard', 'n/a')}"),
        ("Blood check", f"{get(s, 'blood_check.rule', 'n/a')} ({get(s, 'blood_check.file', '')})"),
        ("Tissue model", f"{get(s, 'model.name', 'n/a')} · prompts {get(s, 'model.prompts_version', 'n/a')} · "
                         f"flag phrases v{s.get('phrases_version', 'n/a')}"),
        ("Second opinion", f"{get(s, 'alt_model.name', 'none')} (kept for reference only)"),
        ("Embeddings", f"{s.get('embeddings', 'n/a')}" + (f"; {s['embeddings_musk']}" if s.get("embeddings_musk") else "")),
        ("Detectors", ", ".join(f"{k}: {v}" for k, v in (s.get("detectors") or {}).items()) or "n/a"),
    ]
    if qc.get("blood_cutoff_is_placeholder"):
        sec.note = "The blood cut-off is a placeholder (no clean cut-off was found)."
    return sec


def caveats_section(s):
    sec = Section("caveats", "Caveats (read before using any number)")
    sec.rows = [("", c) for c in s.get("caveats") or []]
    return sec


# ---------------------------------------------------------------- one square

def tile_explanation(data, i):
    """Why the model picked this class, how sure it is and why, and what to check by eye."""
    out = {"risk_points": int(data.risk_points[i])}
    if not data.ok[i]:
        reason = data.tiles["qc_reason"].iloc[i] or "failed check"
        out.update(headline="Not analyzed", conf=None, sure="Not analyzed", why=[f"Skipped by the quality check ({reason})."],
                   how=[], advice=None, mixup=None, second=None)
        return out
    probs = data.probs[i]
    order = np.argsort(probs)[::-1]
    top, second = CLASSES[order[0]], CLASSES[order[1]]
    p1, p2 = probs[order[0]], probs[order[1]]
    pair = (top, second)
    why = [f"Closest match: {NAMES[top]} ({p1:.0%}); next: {NAMES[second]} ({p2:.0%})."]
    how = []
    if p1 >= 0.9:
        how.append(f"high top score ({p1:.0%})")
    elif p1 < 0.5:
        how.append(f"low top score ({p1:.0%})")
    else:
        how.append(f"moderate top score ({p1:.0%})")
    alt_text = None
    if data.has_alt and data.alt[i] >= 0:
        alt = CLASSES[data.alt[i]]
        alt_text = "agrees" if alt == top else f"says {NAMES[alt].lower()}"
        if alt != top:
            pair = (top, alt)
    share = float(data.share[i])
    why.append(f"{share:.0%} of the squares around it are also {NAMES[top].lower()}.")
    how.append("neighbors match" if share >= 0.5 else "neighbors differ (it may sit on a border)")
    why.append(f"{NAMES[top]} typically looks like {LOOKS_LIKE[top]}.")
    sure = str(data.certainty[i])
    advice = None
    if sure != "Sure":
        a, b = pair
        advice = f"Check by eye: {NAMES[a].lower()} ({LOOKS_LIKE[a]}) or {NAMES[b].lower()} ({LOOKS_LIKE[b]})?"
    out.update(headline=NAMES[top], conf=float(p1), sure=sure, why=why, how=how, advice=advice,
               mixup=KNOWN_MIXUPS.get(frozenset(pair)), second=alt_text)
    return out


def zone_text(data, i):
    zone = data.zone[i]
    if not zone:
        return ""
    d = data.dist_um[i]
    where = "" if np.isnan(d) else f", {abs(d) / 1000:.1f} mm {'inside' if d < 0 else 'outside'} the front"
    text = ("Invasive front" if data.on_front[i] else ZONE_SHORT.get(zone, zone)) + where
    if data.in_main_tumor[i]:
        text += " · in the main tumor"
    return text
