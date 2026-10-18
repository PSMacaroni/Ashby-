"""The texture-based glass / pen-ink check (src/detectors.texture_measures, texture_glass, quality_check)."""
import numpy as np
import pandas as pd
from PIL import Image

from src.detectors import TEXTURE_COLUMNS, quality_check, texture_glass, texture_measures

GLASS = np.array([236, 233, 226])          # slightly tinted glass, as on many TCGA slides
rng = np.random.default_rng(0)


def image(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def glass_tile():
    return image(np.broadcast_to(GLASS, (224, 224, 3)) + rng.normal(0, 1.5, (224, 224, 3)))


def tissue_tile():                          # stained tissue: purple/pink with lots of structure
    base = np.broadcast_to([170, 90, 160], (224, 224, 3)).astype(float)
    return image(base + rng.normal(0, 35, (224, 224, 3)))


def fat_tile():                             # white cells separated by thin pink membranes across the whole tile
    a = np.broadcast_to(GLASS, (224, 224, 3)).astype(float).copy()
    for k in range(0, 224, 45):
        a[k:k + 2, :] = [200, 120, 160]
        a[:, k:k + 2] = [200, 120, 160]
    return image(a)


def edge_tile(tissue_share):                # tissue on the left, glass on the right
    a = np.asarray(glass_tile()).astype(float).copy()
    n = int(224 * tissue_share)
    a[:, :n] = np.asarray(tissue_tile())[:, :n]
    return image(a)


def ink_tile():
    return image(np.broadcast_to([40, 80, 200], (224, 224, 3)) + rng.normal(0, 3, (224, 224, 3)))


def is_glass(tile):
    return bool(texture_glass(dict(zip(TEXTURE_COLUMNS, [[v] for v in texture_measures(tile, GLASS)])))[0])


def test_glass_and_ink_are_caught():
    assert is_glass(glass_tile())
    assert is_glass(ink_tile())
    assert is_glass(edge_tile(0.15))        # a sliver of tissue on glass


def test_tissue_fat_and_half_tissue_edges_are_kept():
    assert not is_glass(tissue_tile())
    assert not is_glass(fat_tile())         # pale, but membranes run across the whole tile
    assert not is_glass(edge_tile(0.55))


def test_quality_check_is_unchanged_without_texture_columns():
    tiles = pd.DataFrame({"colorful": [0.05, 0.8, 0.2], "blur": [1e-4, 5e-3, 2e-3], "label": ["necrosis", "tumor", "fat"]})
    before = quality_check(tiles.colorful, tiles.blur, tiles.label)
    after = quality_check(tiles.colorful, tiles.blur, tiles.label, texture=tiles)     # no texture columns
    assert (before[0] == after[0]).all() and (before[1] == after[1]).all()
    with_nan = tiles.assign(**{c: np.nan for c in TEXTURE_COLUMNS})                   # e.g. old tiles after --extend
    assert (quality_check(tiles.colorful, tiles.blur, tiles.label, texture=with_nan)[0] == before[0]).all()


def test_quality_check_uses_texture_even_for_fat_labels():
    m = [texture_measures(t, GLASS) for t in (glass_tile(), fat_tile())]
    tiles = pd.DataFrame(m, columns=TEXTURE_COLUMNS).assign(colorful=[0.25, 0.2], blur=[2e-3, 2e-3], label=["fat", "fat"])
    background = quality_check(tiles.colorful, tiles.blur, tiles.label, texture=tiles)[0]
    assert background.tolist() == [True, False]
