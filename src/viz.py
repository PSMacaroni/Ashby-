"""
Colors and pictures shared by the scripts and the app, so every tissue class has one color everywhere.
"""
import numpy as np

from src.prompts import CLASSES

CLASS_COLORS = {
    "tumor": "#d62728",          # red
    "normal_mucosa": "#2ca02c",  # green
    "muscle": "#e377c2",         # pink
    "fat": "#f2d16b",            # yellow
    "lymphocytes": "#1f77b4",    # blue
    "mucus": "#17becf",          # cyan
    "necrosis": "#555555",       # dark gray
    "stroma": "#c49c94",         # tan
}
BACKGROUND = "#ffffff"
FLAG_MARKERS = {"perineural": ("^", "#000000"), "venous": ("s", "#6a3d9a"), "lymphatic": ("D", "#ff7f00"),
                "budding": ("*", "#b15928"), "stroma_maturity": ("x", "#666666"), "adenoma": ("P", "#33a02c")}


def hex_to_rgb(color):
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def label_image(label):
    """[R, C] class indices (-1 = no tissue) -> [R, C, 3] RGB image in the class colors."""
    rgb = np.ones((*label.shape, 3))
    for i, name in enumerate(CLASSES):
        rgb[label == i] = hex_to_rgb(CLASS_COLORS[name])
    return rgb


def overview_figure(tm, fr, summary, points, path):
    """Four panels to check the analysis by eye: tissue map, zones + front, risk map, flags."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from src.front import BORDER_FRONT, ZONE_CORE, ZONE_FRONT, ZONE_PERI

    fig, axes = plt.subplots(2, 2, figsize=(16, 16 * tm.shape[0] / tm.shape[1] + 1.5))
    ax = axes[0, 0]
    ax.imshow(label_image(tm.label), interpolation="nearest")
    ax.legend(handles=[Patch(color=CLASS_COLORS[n], label=n) for n in CLASSES], loc="lower right", fontsize=8)
    ax.set_title("Tissue map (smoothed)")

    ax = axes[0, 1]
    zones = np.ones((*tm.shape, 3))
    zones[tm.tissue] = (0.85, 0.85, 0.85)
    if fr is not None:
        for z, color in ((ZONE_CORE, (0.55, 0.0, 0.0)), (ZONE_FRONT, (1.0, 0.45, 0.3)), (ZONE_PERI, (1.0, 0.85, 0.7))):
            zones[fr.zone == z] = color
        zones[fr.border_type == BORDER_FRONT] = (0, 0, 0)
        for a in summary["border"]["arcs"]:
            if a["infiltrative"]:
                (r, c), (dr, dc) = a["center_rc"], a["normal_rc"]
                ax.arrow(c, r, dc * 6, dr * 6, color="blue", width=0.3, head_width=1.5)
        for p in summary["satellites"]:
            ax.plot(p["col"], p["row"], "o", mfc="none", mec="blue", ms=12)
    ax.imshow(zones, interpolation="nearest")
    ax.set_title("Zones: core / front band / peritumoral; front in black; arrows = infiltrative arcs")

    ax = axes[1, 0]
    shown = np.ma.masked_where(~tm.tissue, points.astype(float))
    im = ax.imshow(shown, cmap="Reds", vmin=0, vmax=max(4, points.max()), interpolation="nearest")
    for card in summary.get("risk", {}).get("cards", [])[:8]:
        ax.text(card["col"], card["row"], f"L{card['level']}", color="black", fontsize=10, ha="center",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 1})
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Risk points (1 mm neighborhood)")

    ax = axes[1, 1]
    ax.imshow(0.5 + 0.5 * label_image(tm.label), interpolation="nearest")
    for f in summary.get("flags", []):
        marker, color = FLAG_MARKERS.get(f["key"], ("o", "black"))
        for i, cand in enumerate(f["candidates"]):
            ax.plot(cand["col"], cand["row"], marker, color=color, ms=9, label=f["title"] if i == 0 else None)
    for region in summary.get("uncertainty", {}).get("regions", []):
        ax.plot(region["col"], region["row"], "o", mfc="none", mec="gray", ms=16, mew=2,
                label="unsure here" if region is summary["uncertainty"]["regions"][0] else None)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Flags: look here (hints, not findings)")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(summary.get("slide_id", ""), fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
