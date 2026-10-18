"""
Words for the panels. The slide-level sentences come straight from summary.json (the team's
analysis writes a "text" for every section); this file only picks and orders them, and writes the
per-tile explanation. Lumen can reword the slide summary (lumen.py); this is the fallback.
Rule: restate what the analysis says; never add a diagnosis or a treatment suggestion.
"""
import numpy as np

from common import CLASSES, KNOWN_MIXUPS, LOOKS_LIKE, NAMES, get, pct

# (label, path to the sentence in summary.json), in the order a pathologist reads them
KEY_FINDINGS = [
    ("Growth", "border.pattern_text"),
    ("Depth", "invasion.t_stage_hint"),
    ("Direction", "invasion.direction.text"),
    ("Ahead of front", "invasion.heading_into.text"),
    ("Fat", "fat.text"),
    ("Immune", "immune.text"),
    ("Necrosis", "necrosis.text"),
]
MORE_FINDINGS = [
    ("Stroma", "stroma.text"),
    ("Mucin", "mucin.label"),
    ("Spread", "spread.text"),
    ("Outer edge", "outer_edge.text"),
    ("Unsure", "uncertainty.text"),
]


def tumor_line(s):
    if not get(s, "tumor.found", True) or not get(s, "tumor.fraction_of_tissue"):
        return "No tumor found by the map in this section."
    area = get(s, "tumor.area_mm2")
    return f"Tumor covers {pct(get(s, 'tumor.fraction_of_tissue'))} of the tissue" + (f" ({area} mm²)." if area else ".")


def findings(s, more=False):
    """[(label, sentence)] for the summary panel; only the ones the analysis wrote."""
    rows = [("Tumor", tumor_line(s))]
    if not get(s, "tumor.found", True):
        return rows + ([("Unsure", get(s, "uncertainty.text"))] if get(s, "uncertainty.text") else [])
    for label, path in KEY_FINDINGS + (MORE_FINDINGS if more else []):
        text = get(s, path)
        if text:
            rows.append((label, text))
    if more:
        for pool in get(s, "mucin.pools", []) or []:
            rows.append(("Mucin pool", pool["text"]))
        for node in s.get("lymph_nodes") or []:
            rows.append(("Lymph node", node["text"]))
        deposits = [p for p in s.get("satellites") or [] if p.get("possible_deposit")]
        if deposits:
            rows.append(("Islands", f"{len(deposits)} tumor island(s) in fat: possible deposits, check for a node "
                                    "remnant or vessel wall."))
    return rows


def plain_summary(s, name):
    """One paragraph: the template version of the slide summary (Lumen rewords this)."""
    parts = [f"{name}."] + [text for _, text in findings(s)]
    flags = s.get("high_risk_flags") or []
    if flags:
        parts.append("To check: " + " ".join(f if f.endswith(".") else f + "." for f in flags))
    return " ".join(parts)


def tile_explanation(data, i):
    """Why the AI picked this class, how sure it is and why, and what to check by eye."""
    out = {"risk_points": int(data.risk_points[i])}
    if not data.ok[i]:
        reason = data.tiles["qc_reason"].iloc[i] or "failed check"
        out.update(headline="Not analyzed", conf=None, sure="Not analyzed", why=[f"Skipped by the quality check ({reason})."],
                   how=[], advice=None, mixup=None)
        return out
    probs = data.probs[i]
    order = np.argsort(probs)[::-1]
    top, second = CLASSES[order[0]], CLASSES[order[1]]
    p1, p2 = probs[order[0]], probs[order[1]]
    pair = (top, second)
    why = [f"Closest match: {NAMES[top]} ({p1:.0%}); next: {NAMES[second]} ({p2:.0%})."]
    how = []
    if p1 >= 0.8:
        how.append(f"high top score ({p1:.0%})")
    elif p1 < 0.5:
        how.append(f"low top score ({p1:.0%})")
    else:
        how.append(f"moderate top score ({p1:.0%})")
    if data.has_alt and data.alt[i] >= 0:
        alt = CLASSES[data.alt[i]]
        if alt == top:
            why.append("The second model agrees.")
            how.append("second model agrees")
        else:
            why.append(f"The second model says {NAMES[alt]}.")
            how.append("the two models disagree")
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
               mixup=KNOWN_MIXUPS.get(frozenset(pair)))
    return out
