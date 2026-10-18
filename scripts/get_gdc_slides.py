"""
Download open-access TCGA slides from the NCI GDC (https://api.gdc.cancer.gov/data/<file_id>), several at a time,
and check each file's md5. Safe to re-run: verified files are skipped.

The manifest is a CSV with at least file_id, file_name, md5sum (e.g. calibration/manifest_v1.csv). It is also copied
next to the slides as slides.csv, which scripts/process_slide.py reads for project / pT / pN / stage.

    python scripts/get_gdc_slides.py --manifest calibration/manifest_v1.csv --out /projects/bfqi/$USER/tcga_calib
"""
import argparse
import hashlib
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(row, out):
    dest = out / row.file_name
    if dest.exists() and md5(dest) == row.md5sum:
        return row.file_name, "already there"
    part = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://api.gdc.cancer.gov/data/{row.file_id}", timeout=300) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            if md5(part) == row.md5sum:
                part.replace(dest)
                return row.file_name, "ok"
            raise IOError("md5 mismatch")
        except Exception as e:
            if attempt == 3:
                return row.file_name, f"FAILED: {e!r}"
            time.sleep(10 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    df.to_csv(out / "slides.csv", index=False)
    t0, failed = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(fetch, r, out) for r in df.itertuples()]), 1):
            name, status = f.result()
            failed += status.startswith("FAILED")
            print(f"[{i}/{len(df)}] {time.time() - t0:5.0f} s  {status:14} {name}", flush=True)
    print(f"done: {len(df) - failed}/{len(df)} slides verified in {out}" + (f" ({failed} failed: re-run)" if failed else ""))


if __name__ == "__main__":
    main()
