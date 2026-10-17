"""
One slide's tile map as 2-D grids, so the analysis can use image tools (scipy, scikit-image).

tiles.csv has one row per tissue tile with its grid position (row, col). TileMap puts the 8 class
probabilities into a [rows, cols, 8] array. Grid cells with no tile (background glass) or a failed
quality check count as "no tissue".

    tm = TileMap.from_csv("data/results/<slide_id>/tiles.csv")
    tm.label        # [R, C] smoothed top class index (CLASSES order), NO_TISSUE off tissue
    tm.class_mask("tumor")
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from src import cutoffs
from src.prompts import CLASSES

NO_TISSUE = -1
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
# Optional per-tile columns (see CLAUDE.md data contract): detector outputs and a second model's label
EXTRA_PROB_COLUMNS = ("p_nerve", "p_artery")
ALT_LABEL_COLUMN = "alt_label"


@dataclass
class TileMap:
    probs: np.ndarray        # [R, C, 8] raw class probabilities (0 where there is no tissue)
    tissue: np.ndarray       # [R, C] True where there is a usable tissue tile
    tile_um: float           # width of one tile in microns (224 px x 0.5 µm/px = 112 µm)
    x: np.ndarray            # [R, C] level-0 pixel x of each tile's top-left corner (NaN = no tile)
    y: np.ndarray            # [R, C] level-0 pixel y
    extra: dict = field(default_factory=dict)   # optional [R, C] layers, e.g. "p_nerve", "alt_label"

    def __post_init__(self):
        self.smooth = smooth_probs(self.probs, self.tissue)
        self.label = top_class(self.smooth, self.tissue)       # smoothed: use for the analysis
        self.raw_label = top_class(self.probs, self.tissue)    # raw: what the model said per tile
        self.confidence = np.where(self.tissue, self.probs.max(axis=2), 0.0)

    @classmethod
    def from_csv(cls, path, mpp=0.5, shape=None):
        return cls.from_dataframe(pd.read_csv(path), mpp=mpp, shape=shape)

    @classmethod
    def from_dataframe(cls, df, mpp=0.5, shape=None):
        """mpp = microns per pixel of the tiles (0.5 for our 224 px tiles), from summary.json."""
        rows, cols = df["row"].to_numpy(int), df["col"].to_numpy(int)
        R, C = shape if shape is not None else (rows.max() + 1, cols.max() + 1)
        ok = df["qc_ok"].astype(bool).to_numpy() if "qc_ok" in df else np.ones(len(df), bool)

        probs = np.zeros((R, C, len(CLASSES)))
        probs[rows, cols] = df[[f"p_{c}" for c in CLASSES]].to_numpy(float)
        tissue = np.zeros((R, C), bool)
        tissue[rows[ok], cols[ok]] = True
        probs[~tissue] = 0

        x = np.full((R, C), np.nan)
        y = np.full((R, C), np.nan)
        if "x" in df and "y" in df:
            x[rows, cols], y[rows, cols] = df["x"].to_numpy(float), df["y"].to_numpy(float)

        extra = {}
        for name in EXTRA_PROB_COLUMNS:
            if name in df:
                grid = np.zeros((R, C))
                grid[rows, cols] = df[name].fillna(0).to_numpy(float)
                extra[name] = np.where(tissue, grid, 0.0)
        if ALT_LABEL_COLUMN in df:
            grid = np.full((R, C), NO_TISSUE)
            grid[rows, cols] = [CLASS_INDEX.get(v, NO_TISSUE) for v in df[ALT_LABEL_COLUMN]]
            extra[ALT_LABEL_COLUMN] = np.where(tissue, grid, NO_TISSUE)

        tile_um = float(df["tile_size_px"].iloc[0]) * mpp
        return cls(probs=probs, tissue=tissue, tile_um=tile_um, x=x, y=y, extra=extra)

    @property
    def shape(self):
        return self.tissue.shape

    @property
    def tile_mm2(self):
        return (self.tile_um / 1000.0) ** 2

    def tiles(self, um):
        """Convert a distance in microns to a (fractional) number of tiles."""
        return um / self.tile_um

    def class_mask(self, name):
        """Tiles whose smoothed top class is `name`."""
        return self.label == CLASS_INDEX[name]

    def tile_ref(self, r, c):
        """Where a tile is: grid position + level-0 pixel position (for the app and the pathologist)."""
        r, c = int(r), int(c)
        x, y = self.x[r, c], self.y[r, c]
        return {"row": r, "col": c,
                "x": None if np.isnan(x) else int(x), "y": None if np.isnan(y) else int(y)}


def smooth_probs(probs, tissue, size=cutoffs.SMOOTH_SIZE):
    """Average each class probability over a size x size neighborhood, counting tissue tiles only."""
    weight = tissue.astype(float)
    count = ndimage.uniform_filter(weight, size=size, mode="constant")
    out = np.empty_like(probs)
    for k in range(probs.shape[2]):
        out[..., k] = ndimage.uniform_filter(probs[..., k] * weight, size=size, mode="constant")
    out /= np.maximum(count, 1e-9)[..., None]
    out[~tissue] = 0
    return out


def top_class(probs, tissue):
    label = probs.argmax(axis=2)
    label[~tissue] = NO_TISSUE
    return label
