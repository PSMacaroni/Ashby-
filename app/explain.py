"""
Plain-language text built from the numbers: fill-in-the-blank templates, no AI involved.
Lumen can reword this text later; when Lumen is off, the app shows it as is.
Rule: only restate numbers we have; never add a diagnosis or a treatment suggestion.
"""
import numpy as np

from common import CLASSES, HIGH_CONF, KNOWN_MIXUPS, LOOKS_LIKE, LOW_CONF, NAMES, SHORT_NAMES, pct

BORDER = {
    "pushing": "pushing (a smooth, broad edge)",
    "infiltrative": "infiltrative (a ragged edge with fingers or islands)",
}
IMMUNE = {
    "inflamed": "Lymphocytes sit inside the tumor as well as at its edge ('inflamed' pattern).",
    "excluded": "Lymphocytes gather at the tumor edge but few get inside ('excluded' pattern).",
    "desert": "Few lymphocytes inside the tumor or at its edge ('desert' pattern).",
}
IMMUNE_NOTE = ("This shows where lymphocytes sit, not what they are doing; "
               "H&E can't tell T-cell types (CD4, CD8) apart.")
SURE_MEANING = {
    "Sure": "one class clearly wins, the models agree, and most neighbors match",
    "Somewhat sure": "one class wins, but not by much or its neighbors look different",
    "Unsure": "the score is low or CONCH and MUSK disagree",
}


def get(d, path, default=None):
    """get(summary, 'border.pattern') -> value, or default if any part is missing."""
    for part in path.split("."):
        if not isinstance(d, dict) or d.get(part) is None:
            return default
        d = d[part]
    return d


# ---------------------------------------------------------------- one tile

def tile_explanation(data, idx):
    """Why the AI picked this class, how sure it is and why, and what to check by eye."""
    t = data["tiles"].iloc[idx]
    out = {"risk_level": int(t["risk_level"]), "risk_flags": t["risk_flags"]}
    if not t["qc_ok"]:
        out.update(headline="Not analyzed", sure="Not analyzed", conf=None, mixup=None, advice=None,
                   why=[f"The quality check skipped this square ({t['qc_reason'] or 'failed check'}), "
                        "so the AI made no call here."], how_sure=[])
        return out

    probs = data["probs"][idx]
    order = np.argsort(probs)[::-1]
    top, second = CLASSES[order[0]], CLASSES[order[1]]
    p1, p2 = probs[order[0]], probs[order[1]]
    share = float(data["share"][idx])
    sure = str(data["sure"][idx])
    pair = (top, second)

    why = [f"The models matched this square most closely to {NAMES[top]} ({p1:.0%}); "
           f"next closest was {NAMES[second]} ({p2:.0%})."]
    how = []
    if p1 >= HIGH_CONF:
        how.append(f"the top score is high ({p1:.0%})")
    elif p1 < LOW_CONF:
        how.append(f"the top score is low ({p1:.0%})")
    else:
        how.append(f"the top score is only moderate ({p1:.0%})")

    if data["has_models"]:
        m = data["model"]
        cl, ml = CLASSES[m["conch"]["label"][idx]], CLASSES[m["musk"]["label"][idx]]
        cp, mp = m["conch"]["probs"][idx].max(), m["musk"]["probs"][idx].max()
        if cl == ml:
            why.append(f"CONCH and MUSK agree: both say {SHORT_NAMES[cl]} (CONCH {cp:.0%}, MUSK {mp:.0%}).")
            how.append("both models agree")
        else:
            why.append(f"CONCH and MUSK disagree: CONCH says {SHORT_NAMES[cl]} ({cp:.0%}), "
                       f"MUSK says {SHORT_NAMES[ml]} ({mp:.0%}).")
            how.append("CONCH and MUSK disagree")
            pair = (cl, ml)
        if cp >= 0.98 and sure != "Sure":
            how.append("CONCH is often over-confident (in our test it averaged 88% even when wrong), "
                       "so its high score alone isn't proof")

    why.append(f"{share:.0%} of the squares around it are also labeled {SHORT_NAMES[top]}.")
    how.append("most neighbors match" if share >= 0.5
               else "most neighbors look different, so it may sit on a border between tissues")
    why.append(f"{SHORT_NAMES[top]} typically looks like {LOOKS_LIKE[top]}.")

    advice = None
    if sure != "Sure":
        a, b = pair
        advice = (f"Check this spot by eye: is it {SHORT_NAMES[a].lower()} ({LOOKS_LIKE[a]}) "
                  f"or {SHORT_NAMES[b].lower()} ({LOOKS_LIKE[b]})?")
    out.update(headline=NAMES[top], conf=float(p1), sure=sure, why=why, how_sure=how,
               mixup=KNOWN_MIXUPS.get(frozenset(pair)), advice=advice)
    return out


def sure_sentence(expl):
    if expl["sure"] == "Not analyzed":
        return "Not analyzed."
    return f"{expl['sure']}: " + "; ".join(expl["how_sure"]) + "."


# ---------------------------------------------------------------- whole slide

def most_common_tissue(summary):
    fr = get(summary, "tissue_fractions", {}) or {}
    if not fr:
        return None, None
    c = max(fr, key=lambda k: fr[k] or 0)
    return c, fr[c]


def key_findings(summary):
    """The 4-5 one-line findings for the short summary view."""
    lines = []
    tumor = get(summary, "tumor.fraction_of_tissue")
    if tumor is None:
        lines.append("Tumor: not computed yet.")
    elif tumor == 0:
        lines.append("No tumor found by the map in this section.")
        return lines
    else:
        area = get(summary, "tumor.area_mm2")
        lines.append(f"Tumor covers {pct(tumor)} of the tissue" + (f" ({area:.1f} mm²)." if area else "."))
    pattern = get(summary, "border.pattern")
    lines.append(f"Border: {BORDER.get(pattern, pattern)}." if pattern else "Border: not computed yet.")
    hint = get(summary, "invasion.t_stage_hint")
    lines.append(f"Depth: {hint}" if hint else "Depth: not computed yet.")
    immune = get(summary, "immune.pattern")
    if immune:
        lines.append(f"Immune: {immune} (tumor core {pct(get(summary, 'immune.core_fraction'))} "
                     f"lymphocytes, edge {pct(get(summary, 'immune.margin_fraction'))}).")
    nec = get(summary, "necrosis.fraction_in_tumor")
    if nec:
        central = ", forming a central core" if get(summary, "necrosis.central") else ""
        lines.append(f"Necrosis: {pct(nec)} of the tumor area{central}.")
    return lines


def full_report(summary, tile_um=112):
    """Sections for the expanded summary: {title: [lines]}."""
    sec = {}
    tumor = get(summary, "tumor.fraction_of_tissue")
    if tumor == 0:
        sec["Tumor"] = ["No tumor found by the map in this section, so border, depth, immune and "
                        "necrosis measurements don't apply."]
        return sec

    sec["Tumor"] = [f"Share of tissue: {pct(tumor, 1)}",
                    f"Area: {get(summary, 'tumor.area_mm2', 'n/a')} mm²"]

    sol, irr = get(summary, "border.solidity"), get(summary, "border.irregularity")
    pattern = get(summary, "border.pattern")
    sec["Border shape"] = [
        f"Pattern: {BORDER.get(pattern, pattern or 'not computed yet')}",
        f"Solidity: {sol} (how fully the tumor fills its outline; 1 = perfectly smooth)" if sol else
        "Solidity: not computed yet",
        f"Irregularity: {irr} (border length vs. a circle of the same area; higher = more ragged)" if irr else
        "Irregularity: not computed yet",
        f"Border tiles: {get(summary, 'border.n_border_tiles', 'n/a')}",
        "Cutoffs between pushing and infiltrative are placeholders until a pathologist calibrates them.",
    ]

    touch = get(summary, "invasion.border_touch_fraction", {}) or {}
    side = get(summary, "invasion.direction.side")
    clock = get(summary, "invasion.direction.clock")
    lines = [f"Deepest layer reached: {get(summary, 'invasion.deepest_layer', 'not computed yet')}",
             f"T-stage hint: {get(summary, 'invasion.t_stage_hint', 'not computed yet')}"]
    if touch:
        lines.append("Border touching: " + ", ".join(f"{SHORT_NAMES.get(k, k).lower()} {pct(v)}"
                                                    for k, v in touch.items() if v is not None))
    if side:
        lines.append(f"The front pushes deepest toward the {side}"
                     + (f" ({clock} o'clock)" if clock else "") + " in this slice.")
    lines.append("One slide: this is where the tumor is invading in this section, not a prediction.")
    sec["Depth and direction"] = lines

    immune = get(summary, "immune.pattern")
    by_side = get(summary, "immune.margin_by_side", {}) or {}
    lines = [IMMUNE.get(immune, f"Pattern: {immune or 'not computed yet'}"),
             f"Lymphocytes in the tumor core: {pct(get(summary, 'immune.core_fraction'))} of squares",
             f"Lymphocytes at the edge (within 0.5 mm): {pct(get(summary, 'immune.margin_fraction'))} of squares"]
    if by_side:
        lines.append("Edge by side: " + ", ".join(f"{k} {pct(v)}" for k, v in by_side.items() if v is not None))
    lines.append(IMMUNE_NOTE)
    sec["Immune response"] = lines

    nec = get(summary, "necrosis.fraction_in_tumor")
    sec["Necrosis"] = [f"Share of tumor area: {pct(nec)}" if nec is not None else "Not computed yet",
                       "Forms a central core" if get(summary, "necrosis.central") else "No central core"]

    tsr = get(summary, "stroma.worst_field_stroma_fraction")
    if tsr is not None:
        high = get(summary, "stroma.stroma_high")
        sec["Stroma around the tumor"] = [
            f"Most stroma-rich field at the front: {pct(tsr)} stroma ({'stroma-high' if high else 'stroma-low'}; "
            "the cutoff is 50%)",
            "H&E shows the stroma reaction, not which cells are cancer-associated fibroblasts "
            "(that needs α-SMA or FAP staining)."]

    sats = summary.get("satellites") or []
    if sats:
        sec["Tumor islands near the main mass"] = [
            f"Island {i}: {s['n_tiles']} squares, {s['distance_mm']} mm from the main tumor, "
            f"surrounded by {SHORT_NAMES.get(s['surrounded_by'], s['surrounded_by']).lower()}"
            for i, s in enumerate(sats, 1)]
    return sec


def plain_summary(summary, name):
    """One paragraph for the PDF and for Lumen to reword."""
    flags = summary.get("high_risk_flags") or []
    parts = [f"Slide {name}."] + key_findings(summary)
    if flags:
        parts.append("Flags to check: " + "; ".join(flags) + ".")
    parts.append("Decision support only, not a diagnosis.")
    return " ".join(parts)
