"""
The worklist: everything on a slide a pathologist should look at, in one ordered list with a
place on the map where there is one. Items are grouped:

  flags    the high-risk features the analysis listed (summary.high_risk_flags) and the strong
           "look here" spots of every flag (perineural, venous, lymphatic, budding, adenoma)
  mucin    mucin pools beyond the muscle
  islands  tumor islands that count as satellites, and possible deposits
  nodes    lymph nodes on the slide (with or without tumor inside)
  risk     risk regions (level 3 and up)
  unsure   the biggest unsure areas

Each item has a stable id, so "reviewed" ticks survive a restart (slide_data.set_reviewed).
"""
from common import POINTER_COLORS, get, mm, risk_color

GROUPS = [("flags", "Flags"), ("risk", "Risk regions"), ("unsure", "Unsure areas"), ("mucin", "Mucin pools"),
          ("islands", "Tumor islands"), ("nodes", "Lymph nodes")]
GROUP_NAMES = dict(GROUPS)
SHORT_NAMES = {"flags": "Flags", "risk": "Risk", "unsure": "Unsure", "mucin": "Mucin", "islands": "Islands", "nodes": "Nodes"}
LETTERS = {"flags": "F", "risk": "R", "unsure": "U", "mucin": "M", "islands": "I", "nodes": "N"}
MIN_RISK_LEVEL = 3


def _item(group, id_, title, sub="", loc=None, color=None, action=""):
    return {"id": f"{group}:{id_}", "group": group, "title": title, "sub": sub, "loc": loc,
            "color": color or POINTER_COLORS.get(group, POINTER_COLORS["other"]), "action": action}


def build(s):
    """[item] for a summary. Items with a loc get a numbered pointer on the map (F1, R2, ...)."""
    items = []
    if not get(s, "tumor.found", True):
        for n, reg in enumerate(get(s, "uncertainty.regions", []) or [], 1):
            items.append(_item("unsure", f"region{n}", f"Unsure area · {reg['n_tiles']} squares", reg.get("why", ""),
                               (int(reg["row"]), int(reg["col"]))))
        return number(items)

    # high-risk features: the analysis's own list, each pointed at a place where one exists
    flags = {f.get("key"): f for f in s.get("flags") or []}
    stroma = s.get("stroma") or {}
    arcs = sorted([a for a in get(s, "border.arcs", []) or [] if a.get("infiltrative")], key=lambda a: -a.get("score", 0))
    for n, text in enumerate(s.get("high_risk_flags") or [], 1):
        low = text.lower()
        loc, action = None, "Check by eye."
        if "stroma-high" in low and stroma.get("row") is not None:
            loc = (int(stroma["row"]), int(stroma["col"]))
            action = "The marked field has the most stroma next to tumor."
        elif "infiltrative" in low and arcs:
            r, c = arcs[0]["center_rc"]
            loc, action = (int(round(r)), int(round(c))), "The most infiltrative arc of the front is marked."
        elif "venous" in low and flags.get("venous"):
            strong = [c for c in flags["venous"].get("candidates") or [] if c.get("strong")]
            if strong:
                loc, action = (int(strong[0]["row"]), int(strong[0]["col"])), flags["venous"].get("action") or action
        elif "perineural" in low and flags.get("perineural"):
            strong = [c for c in flags["perineural"].get("candidates") or [] if c.get("strong")]
            if strong:
                loc, action = (int(strong[0]["row"]), int(strong[0]["col"])), flags["perineural"].get("action") or action
        elif "mucin" in low:
            pools = sorted(get(s, "mucin.pools", []) or [], key=lambda p: -(p.get("area_mm2") or 0))
            if pools:
                loc = (int(pools[0]["row"]), int(pools[0]["col"]))
        elif "fat" in low and "touch" in low:
            action = "Confirm that this fat lies outside the muscle wall (pT3), not in the submucosa."
        elif "serosa" in low or "outer" in low:
            action = get(s, "outer_edge.caveat") or action
        items.append(_item("flags", f"high{n}", text, "High-risk feature from the analysis", loc, POINTER_COLORS["flags"], action))

    # strong "look here" spots per flag
    for key, f in flags.items():
        if f.get("status") != "ok":
            continue
        strong = [c for c in f.get("candidates") or [] if c.get("strong")]
        for n, c in enumerate(strong[:5], 1):
            items.append(_item("flags", f"{key}{n}", f"{f.get('title')}", f"{c.get('why', '')} · {f.get('source', '')} · trust: {f.get('trust', '')}",
                               (int(c["row"]), int(c["col"])), POINTER_COLORS["flags"], f.get("action") or ""))

    for n, c in enumerate(get(s, "risk.cards", []) or [], 1):
        if c.get("level", 0) >= MIN_RISK_LEVEL:
            items.append(_item("risk", f"card{n}", f"Risk level {c['level']} · {c['n_tiles']} squares",
                               "; ".join(c.get("features") or []), (int(c["row"]), int(c["col"])),
                               risk_color(c["level"], max(4, get(s, "risk.max_level") or 4))))

    for n, reg in enumerate(get(s, "uncertainty.regions", []) or [], 1):
        items.append(_item("unsure", f"region{n}", f"Unsure area · {reg['n_tiles']} squares", reg.get("why", ""),
                           (int(reg["row"]), int(reg["col"])), action="Decide by eye what this tissue is."))

    for n, p in enumerate(sorted(get(s, "mucin.pools", []) or [], key=lambda p: -(p.get("area_mm2") or 0))[:10], 1):
        items.append(_item("mucin", f"pool{n}", p.get("text") or "Mucin pool beyond the muscle",
                           f"{p.get('area_mm2', '?')} mm² · {mm(p.get('distance_past_muscle_um'))} past the muscle",
                           (int(p["row"]), int(p["col"])), action="Look for floating tumor cells in the pool."))

    for n, p in enumerate(s.get("satellites") or [], 1):
        if not (p.get("satellite") or p.get("possible_deposit")):
            continue
        title = "Possible tumor deposit" if p.get("possible_deposit") else "Satellite tumor island"
        items.append(_item("islands", f"sat{p.get('id', n)}", title,
                           f"{p.get('diameter_um', '?')} µm across · {mm(p.get('distance_to_main_um'))} from the main tumor · in {p.get('sits_in', '?')}",
                           (int(p["row"]), int(p["col"])),
                           action="Check for a lymph node remnant or a vessel wall." if p.get("possible_deposit") else ""))

    for n, node in enumerate(s.get("lymph_nodes") or [], 1):
        loc = (int(node["row"]), int(node["col"])) if node.get("row") is not None else None
        items.append(_item("nodes", f"node{n}", node.get("text") or f"Lymph node {n}", "", loc))
    return number(items)


def number(items):
    """Give every item with a place a short pointer label: F1, F2, R1, ..."""
    counts = {}
    for it in items:
        if it["loc"] is None:
            it["label"] = ""
            continue
        g = it["group"]
        counts[g] = counts.get(g, 0) + 1
        it["label"] = f"{LETTERS[g]}{counts[g]}"
    return items


def markers(items):
    """Map pointers for a list of items."""
    return [{"r": it["loc"][0], "c": it["loc"][1], "label": it["label"], "color": it["color"],
             "hover": f"{it['label']} · {it['title']}"} for it in items if it["loc"] is not None]
