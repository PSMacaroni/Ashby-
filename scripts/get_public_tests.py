"""
Download the two public Heidelberg/Mannheim test sets from Zenodo (CC-BY 4.0) and unzip them:

  CRC-VAL-HE-7K  (Zenodo 1214456, 800 MB): 7,180 tiles from 50 patients not in NCT-CRC-HE-100K,
                 224 px at 0.5 µm/px, Macenko color-normalized, 9 NCT classes  -> <out>/CRC-VAL-HE-7K/<ADI|...>/
  Kather 2016    (Zenodo 53169, 258 MB): 5,000 tiles of 150 px at 0.495 µm/px from 10 patients, NOT color-
                 normalized, 8 coarser classes -> <out>/kather2016/Kather_texture_2016_image_tiles_5000/<01_TUMOR|...>/

Safe to re-run: finished downloads and unzipped folders are skipped. CPU only (fine on a Delta login node).

    python scripts/get_public_tests.py --out data/datasets
"""
import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

FILES = {
    "CRC-VAL-HE-7K": ("https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content", ".", "CRC-VAL-HE-7K"),
    "kather2016": ("https://zenodo.org/api/records/53169/files/Kather_texture_2016_image_tiles_5000.zip/content",
                   "kather2016", "kather2016/Kather_texture_2016_image_tiles_5000"),
}


def download(url, path):
    if path.exists():
        return
    tmp = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--sets", nargs="+", choices=list(FILES), default=list(FILES))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        url, unzip_to, check = FILES[name]
        if (out / check).is_dir():
            print(f"{name}: already there ({out / check})")
            continue
        zip_path = out / f"{name}.zip"
        download(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out / unzip_to)
        if not (out / check).is_dir():
            sys.exit(f"{name}: unzipped, but {out / check} is missing: check the zip's folder layout")
        zip_path.unlink()
        n = sum(1 for p in (out / check).rglob("*") if p.is_file())
        print(f"{name}: {n} files in {out / check}")


if __name__ == "__main__":
    main()
