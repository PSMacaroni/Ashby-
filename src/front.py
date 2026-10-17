"""
Step 0 of the analysis: the tumor, its invasive front, and distance zones around it.

Definitions (from our features doc):
  - Tumor = smoothed P(tumor) >= 0.5, cleaned with an opening (radius 1) and small holes filled.
    The largest piece is the main tumor; the other pieces are candidate satellites.
    (Smoothing + opening means pieces smaller than about 3x3 tiles, ~0.3 mm, are not detected.)
  - Invasive front = where the main tumor touches stroma, muscle or fat. We also count lymphocytes,
    because an immune band at the edge sits in the stroma there. Where the tumor touches background,
    mucus or the gut lumen is its SURFACE, not its front (otherwise every ragged surface looks
    infiltrative). Where it touches normal mucosa is spread within the lining, labelled "mucosa".
  - Zones, by distance to the front: core = more than 500 µm inside, front band = within 500 µm on
    either side, peritumoral = 0.5 to 2 mm outside.
  - A slide can hold several tissue sections (TCGA slides often have 2-3 serial sections). We analyze
    the section that holds the main tumor.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from src import cutoffs
from src.prompts import CLASSES
from src.tilemap import CLASS_INDEX, NO_TISSUE

FRONT_CLASSES = ("stroma", "muscle", "fat", "lymphocytes")
MUCOSA_CLASSES = ("normal_mucosa",)

BORDER_NONE, BORDER_FRONT, BORDER_MUCOSA, BORDER_SURFACE = 0, 1, 2, 3
BORDER_NAMES = {BORDER_NONE: "", BORDER_FRONT: "front", BORDER_MUCOSA: "mucosa", BORDER_SURFACE: "surface"}

ZONE_NONE, ZONE_CORE, ZONE_FRONT, ZONE_PERI, ZONE_FAR = 0, 1, 2, 3, 4
ZONE_NAMES = {ZONE_NONE: "", ZONE_CORE: "core", ZONE_FRONT: "front_band",
              ZONE_PERI: "peritumoral", ZONE_FAR: "outside"}

EIGHT = np.ones((3, 3), bool)   # a tile plus its 8 neighbors
OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


@dataclass
class Front:
    sections: np.ndarray      # [R, C] int: tissue section id per tile (0 = no tissue)
    n_sections: int
    section_id: int           # the section that holds the main tumor
    in_section: np.ndarray    # [R, C] bool: tissue tiles of that section
    tumor: np.ndarray         # [R, C] bool: all tumor tiles (cleaned)
    pieces: np.ndarray        # [R, C] int: tumor piece labels (0 = not tumor)
    main_id: int              # label of the main (largest) tumor piece
    main: np.ndarray          # [R, C] bool: main tumor
    main_filled: np.ndarray   # main tumor with interior holes filled (necrosis / mucus inside it)
    border_type: np.ndarray   # [R, C] int: BORDER_* for the main tumor's border tiles
    contact: dict             # name -> [R, C] bool: border tiles touching that class ("background" too)
    signed_um: np.ndarray     # [R, C] distance to the invasive front (negative inside the tumor)
    zone: np.ndarray          # [R, C] int: ZONE_*


# --- small grid helpers (also used by features.py, flags.py, risk.py) -----------------------

def disk(radius_tiles):
    """A round structuring element of the given radius (in tiles)."""
    return morphology.disk(max(0, int(round(radius_tiles)))).astype(bool)


def dilate(mask, radius_tiles=1):
    """Grow a mask: radius 1 adds the 8 neighbors; larger radii use a disk."""
    structure = EIGHT if radius_tiles == 1 else disk(radius_tiles)
    return ndimage.binary_dilation(mask, structure=structure)


def neighbors_of(mask, off_grid=False):
    """Tiles that are in `mask` or have one of their 8 neighbors in it (off the grid counts as `off_grid`)."""
    padded = np.pad(mask, 1, constant_values=off_grid)
    return ndimage.binary_dilation(padded, structure=EIGHT)[1:-1, 1:-1]


def shifted(grid, fill):
    """The 8 neighbor grids: shifted[i][r, c] = grid[r + dr, c + dc] for OFFSETS[i] (fill off the grid)."""
    R, C = grid.shape
    padded = np.pad(grid, 1, constant_values=fill)
    return [padded[1 + dr:1 + dr + R, 1 + dc:1 + dc + C] for dr, dc in OFFSETS]


def fill_small_holes(mask, max_tiles):
    holes = ndimage.binary_fill_holes(mask) & ~mask
    lab, n = ndimage.label(holes)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    small = np.isin(lab, np.flatnonzero(sizes <= max_tiles)) & (lab > 0)
    return mask | small


# --- the steps ------------------------------------------------------------------------------

def tissue_sections(tm):
    """Label separate tissue sections (pieces closer than SECTION_GAP_UM are joined)."""
    grow = max(1, int(round(tm.tiles(cutoffs.SECTION_GAP_UM) / 2)))
    lab = measure.label(ndimage.binary_dilation(tm.tissue, structure=disk(grow)), connectivity=2)
    lab[~tm.tissue] = 0
    return lab, len(np.unique(lab[lab > 0]))


def tumor_mask(tm):
    tumor = (tm.smooth[..., CLASS_INDEX["tumor"]] >= cutoffs.TUMOR_PROB) & tm.tissue
    tumor = ndimage.binary_opening(tumor, structure=disk(1))
    return fill_small_holes(tumor, cutoffs.SMALL_HOLE_TILES)


def classify_border(tm, main_filled):
    """Which border tiles of the main tumor are front, mucosa, or surface, and what each one touches."""
    border = main_filled & neighbors_of(~main_filled, off_grid=True)
    contact = {name: np.zeros(tm.shape, bool) for name in [*CLASSES, "background"]}
    for nb_label, nb_inside in zip(shifted(tm.label, NO_TISSUE), shifted(main_filled, False)):
        outside = border & ~nb_inside
        contact["background"] |= outside & (nb_label == NO_TISSUE)
        for name in CLASSES:
            contact[name] |= outside & (nb_label == CLASS_INDEX[name])
    is_front = np.logical_or.reduce([contact[n] for n in FRONT_CLASSES])
    is_mucosa = np.logical_or.reduce([contact[n] for n in MUCOSA_CLASSES])

    border_type = np.full(tm.shape, BORDER_NONE)
    border_type[border] = BORDER_SURFACE
    border_type[border & is_mucosa] = BORDER_MUCOSA
    border_type[border & is_front] = BORDER_FRONT
    return border_type, contact


def distance_zones(tm, main_filled, border_type, in_section):
    """Signed distance (µm) from every tile to the invasive front, and the zone it falls in."""
    front = border_type == BORDER_FRONT
    ref = front if front.any() else border_type > BORDER_NONE   # no front at all: use the whole border
    dist = ndimage.distance_transform_edt(~ref) * tm.tile_um
    region = in_section | main_filled

    zone = np.full(tm.shape, ZONE_NONE)
    zone[region & ~main_filled & (dist > cutoffs.PERITUMORAL_UM)] = ZONE_FAR
    zone[region & ~main_filled & (dist > cutoffs.FRONT_BAND_UM) & (dist <= cutoffs.PERITUMORAL_UM)] = ZONE_PERI
    zone[region & main_filled & (dist > cutoffs.CORE_UM)] = ZONE_CORE
    zone[region & (dist <= cutoffs.FRONT_BAND_UM)] = ZONE_FRONT

    signed = np.where(main_filled, -dist, dist).astype(float)
    signed[~region] = np.nan
    return signed, zone


def find_front(tm):
    """Run step 0 on a TileMap. Returns a Front, or None if there is no tumor on the slide."""
    sections, n_sections = tissue_sections(tm)
    tumor = tumor_mask(tm)
    pieces = measure.label(tumor, connectivity=2)
    if pieces.max() == 0:
        return None

    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0
    main_id = int(sizes.argmax())
    main = pieces == main_id
    main_filled = ndimage.binary_fill_holes(main)

    under_main = sections[main & (sections > 0)]
    section_id = int(np.bincount(under_main).argmax()) if under_main.size else 0
    in_section = (sections == section_id) & tm.tissue

    border_type, contact = classify_border(tm, main_filled)
    signed_um, zone = distance_zones(tm, main_filled, border_type, in_section)
    return Front(sections=sections, n_sections=n_sections, section_id=section_id, in_section=in_section,
                 tumor=tumor, pieces=pieces, main_id=main_id, main=main, main_filled=main_filled,
                 border_type=border_type, contact=contact, signed_um=signed_um, zone=zone)
