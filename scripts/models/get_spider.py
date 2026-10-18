"""
Pull labeled nerve / vessel tiles (and "other" tissue as negatives) out of the SPIDER datasets
(HistAI, CC-BY-NC 4.0, gated on Hugging Face: accept the terms with the account in HF_TOKEN).

SPIDER tiles are 224 x 224 px at 20x (~0.5 microns/px, like our pipeline), labeled by pathologists,
with a train/test split by slide. Each dataset is one huge tar archive split into ~5 GB parts
(colorectal 89 GB, skin 255 GB): ~1-2 million image files first, then metadata.json (the labels) at
the very end. We never store the archive:
  1. read only metadata.json from the end of the last part (HTTP range requests),
  2. pick the tiles we want,
  3. stream all parts in parallel and save only the picked central tiles.

Labels we save (folder names):
  nerve   SPIDER "Nerves" (skin only: no colon dataset has nerves)
  vessel  SPIDER "Vessels" (colorectal and skin; arteries, veins and lymphatics are not separated)
  other   a random sample of every other class (negatives), source class kept in manifest.csv

Output: <out>/<split>/<label>/<image>.png and <out>/manifest.csv
        (image, split, label, source_class, slide_id, organ)

Run on Delta as a CPU job (scripts/models/get_spider_cpu.sbatch); needs internet + HF_TOKEN in .env.
    python scripts/models/get_spider.py --organ colorectal --out /projects/bfqi/$USER/datasets/spider_colorectal
    python scripts/models/get_spider.py --organ skin --out /projects/bfqi/$USER/datasets/spider_skin --vessel-cap 3000
"""
import argparse
import csv
import io
import json
import random
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config

import requests
from huggingface_hub import HfApi, hf_hub_url

TARGETS = {"Nerves": "nerve", "Vessels": "vessel"}
HEADERS = {"Authorization": f"Bearer {config.HF_TOKEN}"} if config.HF_TOKEN else {}


def list_parts(repo):
    info = HfApi(token=config.HF_TOKEN).dataset_info(repo, files_metadata=True)
    parts = sorted((s.rfilename, s.size) for s in info.siblings if ".tar." in s.rfilename)
    urls = [hf_hub_url(repo, fn, repo_type="dataset") for fn, _ in parts]
    sizes = [size for _, size in parts]
    starts = [sum(sizes[:k]) for k in range(len(sizes))]
    return urls, sizes, starts


def read_range(urls, sizes, starts, a, b):
    """Bytes a..b-1 of the whole (concatenated) archive."""
    out = b""
    for url, size, s in zip(urls, sizes, starts):
        lo, hi = max(a, s), min(b, s + size)
        if lo < hi:
            for attempt in range(5):
                try:
                    r = requests.get(url, headers={**HEADERS, "Range": f"bytes={lo - s}-{hi - s - 1}"}, timeout=300)
                    r.raise_for_status()
                    out += r.content
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
    return out


def read_metadata(urls, sizes, starts, organ):
    """metadata.json is the last file in the archive: search backwards for its tar header."""
    total = starts[-1] + sizes[-1]
    name = f"SPIDER-{organ}/metadata.json".encode() + b"\0"
    end, chunk = total, 128 * 2**20
    while end > 0:
        start = max(0, end - chunk)
        buf = read_range(urls, sizes, starts, start, min(total, end + 1024))
        i = buf.find(name)
        if i >= 0 and (start + i) % 512 == 0:
            size = int(buf[i + 124:i + 136].strip(b"\0 ") or b"0", 8)
            data_start = start + i + 512
            print(f"metadata.json: {size / 1e6:.0f} MB", flush=True)
            return json.loads(read_range(urls, sizes, starts, data_start, data_start + size))
        end = start
    raise RuntimeError("metadata.json not found")


def pick(records, organ, other_per_class, vessel_cap, nerve_cap, seed, targets=TARGETS):
    """{image_name: row} for the tiles we keep."""
    rng = random.Random(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r["class"], []).append(r)
    wanted = {}
    for cls, recs in sorted(by_class.items()):
        recs.sort(key=lambda r: r["image_name"])
        label = targets.get(cls, "other")
        cap = other_per_class if label == "other" else {"vessel": vessel_cap, "nerve": nerve_cap}.get(label)
        if cap is not None and cap < len(recs):  # cap 0 = skip this class
            recs = rng.sample(recs, cap)
        for r in recs:
            wanted[r["image_name"]] = {"image": r["image_name"], "split": r.get("split", "train"), "label": label,
                                       "source_class": cls, "slide_id": r.get("slide_id", ""), "organ": organ}
    return wanted


def first_header(urls, sizes, starts, k):
    """Global offset of the first tar header that starts inside part k (blocks are 512-byte aligned)."""
    if k == 0:
        return 0
    buf = read_range(urls, sizes, starts, starts[k], starts[k] + 4 * 2**20)
    for j in range(0, len(buf) - 512, 512):
        block = buf[j:j + 512]
        if block[257:262] != b"ustar":
            continue
        try:
            stored = int(block[148:156].strip(b"\0 ") or b"-1", 8)
        except ValueError:
            continue
        if stored == sum(block[:148]) + 8 * 32 + sum(block[156:]):
            return starts[k] + j
    raise RuntimeError(f"no tar header found at the start of part {k}")


class PartsStream(io.RawIOBase):
    """Read the concatenated archive from a global offset onwards, one streaming GET per part."""

    def __init__(self, urls, sizes, starts, pos):
        self.urls, self.sizes, self.starts, self.pos = urls, sizes, starts, pos
        self.total = starts[-1] + sizes[-1]
        self.resp = None

    def readable(self):
        return True

    def _open(self):
        k = max(i for i, s in enumerate(self.starts) if s <= self.pos)
        for attempt in range(6):
            try:
                self.resp = requests.get(self.urls[k], headers={**HEADERS, "Range": f"bytes={self.pos - self.starts[k]}-"},
                                         stream=True, timeout=300)
                self.resp.raise_for_status()
                self.part_end = self.starts[k] + self.sizes[k]
                return
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def readinto(self, b):
        while self.pos < self.total:
            if self.resp is None:
                self._open()
            try:
                data = self.resp.raw.read(min(len(b), self.part_end - self.pos))
            except Exception:
                data = None  # dropped connection: reopen at the same position
            if data:
                b[:len(data)] = data
                self.pos += len(data)
                if self.pos >= self.part_end:
                    self.close_resp()
                return len(data)
            self.close_resp()
        return 0

    def close_resp(self):
        if self.resp is not None:
            self.resp.close()
            self.resp = None


def scan_part(k, urls, sizes, starts, wanted, out):
    """Save every wanted tile whose tar header starts inside part k. Returns saved image names."""
    start = first_header(urls, sizes, starts, k)
    part_end = starts[k] + sizes[k]
    stream = PartsStream(urls, sizes, starts, start)
    saved = []
    with tarfile.open(fileobj=io.BufferedReader(stream, 8 * 2**20), mode="r|") as tf:
        for m in tf:
            if start + m.offset >= part_end:
                break
            row = wanted.get(Path(m.name).name)
            if row is None or not m.isfile():
                continue
            path = out / row["split"] / row["label"] / row["image"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".part")
                tmp.write_bytes(tf.extractfile(m).read())
                tmp.replace(path)
            saved.append(row["image"])
    stream.close_resp()
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organ", required=True, choices=["colorectal", "skin", "thorax", "breast"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--other-per-class", type=int, default=500, help="negatives per non-target class")
    ap.add_argument("--vessel-cap", type=int, default=None, help="max vessel tiles (default: all)")
    ap.add_argument("--nerve-cap", type=int, default=None, help="max nerve tiles (default: all)")
    ap.add_argument("--workers", type=int, default=12, help="archive parts streamed in parallel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="+", default=None,
                    help='SPIDER class=our label pairs (default "Nerves=nerve Vessels=vessel"), '
                         'e.g. --targets Inflammation=inflammation --other-per-class 0')
    args = ap.parse_args()
    targets = dict(t.split("=", 1) for t in args.targets) if args.targets else TARGETS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = f"histai/SPIDER-{args.organ}"
    urls, sizes, starts = list_parts(repo)
    print(f"{repo}: {len(urls)} parts, {sum(sizes) / 1e9:.0f} GB", flush=True)

    records = read_metadata(urls, sizes, starts, args.organ)
    print("classes:", dict(sorted(Counter(r["class"] for r in records).items())), flush=True)
    wanted = pick(records, args.organ, args.other_per_class, args.vessel_cap, args.nerve_cap, args.seed, targets)
    del records
    print("picked:", dict(Counter((r["split"], r["label"]) for r in wanted.values())), flush=True)

    t0, saved = time.time(), set()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(scan_part, k, urls, sizes, starts, wanted, out): k for k in range(len(urls))}
        for f in as_completed(futures):
            k = futures[f]
            if f.exception() is not None:
                print(f"part {k}: FAILED {f.exception()!r} (re-run to retry; saved tiles are kept)", flush=True)
                continue
            saved.update(f.result())
            print(f"part {k} done: {len(saved)}/{len(wanted)} tiles, {time.time() - t0:.0f} s", flush=True)

    rows = [r for name, r in sorted(wanted.items()) if (out / r["split"] / r["label"] / name).exists()]
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "split", "label", "source_class", "slide_id", "organ"])
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)}/{len(wanted)} tiles on disk -> {out}", flush=True)
    print("on disk:", dict(Counter((r["split"], r["label"]) for r in rows)), flush=True)


if __name__ == "__main__":
    main()
