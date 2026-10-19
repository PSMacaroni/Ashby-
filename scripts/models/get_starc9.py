"""
Download a subset of STARC-9 (a second labeled colon tile dataset) to train / test the small classifier.

STARC-9 (Subramanian et al., NeurIPS 2025 Datasets track, CC-BY 4.0, huggingface.co/datasets/Path2AI/STARC-9):
630,000 H&E colon tiles from ~200 Stanford patients, 9 classes (our 8 + blood), 256 x 256 px at
0.25 microns/px, Macenko color-normalized. Plus three test sets from *other* patients.
The full set is ~90 GB, so we don't download the zips: we read each zip's file list over the
internet and fetch only the tiles we pick (HTTP range requests).

Each tile is shrunk 256 -> 128 px, i.e. to 0.5 microns/px like our slide tiles (src/slides.py), so
tissue looks the same size as in our pipeline. CONCH's preprocess then resizes to 224 as usual.

Output (git-ignored), one folder per set, one subfolder per class (our class names):
    data/datasets/starc9/
        train/<class>/<SRC>_<n>.png     --train-per-class random tiles per class (no patient IDs given)
        val_stanford_small/...          18,000 tiles, 20 slides: use this to tune the classifier (C)
        val_stanford_large/...          54,000 tiles, 50 slides: final test
        val_tcga/...                    20,000 tiles, 30 TCGA slides: final test, closest to our slides
        <set>/manifest.csv              path, set, label, source_class, source_member
        dataset_info.json               settings used
Classes: tumor, normal_mucosa, muscle, fat, lymphocytes, mucus, necrosis, stroma, blood (STARC_TO_CLASS).

Safe to stop and re-run: tiles already on disk are skipped.

    python scripts/models/get_starc9.py --train-per-class 5 --val-per-class 5 --out data/datasets/starc9_test   # quick test
    python scripts/models/get_starc9.py                                                                          # full run
"""
import argparse
import csv
import io
import json
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets import STARC_TO_CLASS

# Pinned to one dataset version (the authors say an update is coming), so re-runs give the same tiles.
REVISION = "06e80900204d099b2f0409772f94baf1687f754f"
HF = f"https://huggingface.co/datasets/Path2AI/STARC-9/resolve/{REVISION}"
SETS = {
    "train": {code: f"{HF}/Training_data_normalized/{code}.zip" for code in STARC_TO_CLASS},
    "val_stanford_small": {"all": f"{HF}/Validation_data/STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip"},
    "val_stanford_large": {"all": f"{HF}/Validation_data/STANFORD-CRC-HE-VAL-LARGE-NORMALIZED.zip"},
    "val_tcga": {"all": f"{HF}/Validation_data/CURATED-TCGA-CRC-HE-VAL-20K-NORMALIZED.zip"},
}
SOURCE_MPP = 0.25


class RemoteZip:
    """A zip file on a web server. Reads its file list, then single files, without downloading it all."""

    def __init__(self, url):
        self.hf_url = url
        self._resolve()
        self.members = [m for m in zipfile.ZipFile(_RangeFile(self)).infolist()
                        if not m.is_dir() and m.filename.lower().endswith(".png")]

    def _resolve(self):
        # Hugging Face redirects to a signed download link; look it up once instead of on every request.
        with urllib.request.urlopen(urllib.request.Request(self.hf_url, headers={"Range": "bytes=0-0"}),
                                    timeout=60) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def read_range(self, start, end):
        """Bytes start..end (inclusive), with retries."""
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
                return data
            except (urllib.error.URLError, OSError) as e:
                if attempt == 5:
                    raise
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404, 410):
                    self._resolve()  # signed link expired: get a fresh one
                time.sleep(2 ** attempt)

    def read_member(self, m):
        """The file's bytes. A zip stores each file as: 30-byte header, name, extra field, data."""
        start = m.header_offset
        guess = 30 + len(m.filename.encode()) + len(m.extra) + 64 + m.compress_size
        buf = self.read_range(start, min(start + guess, self.size) - 1)
        name_len, extra_len = struct.unpack("<HH", buf[26:30])
        data_start = 30 + name_len + extra_len
        if data_start + m.compress_size > len(buf):
            buf += self.read_range(start + len(buf), start + data_start + m.compress_size - 1)
        data = buf[data_start:data_start + m.compress_size]
        if m.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(data)  # raw deflate
        elif m.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"unsupported compression {m.compress_type} for {m.filename}")
        if zlib.crc32(data) != m.CRC:
            raise OSError(f"checksum mismatch for {m.filename}")
        return data


class _RangeFile(io.RawIOBase):
    """Lets Python's zipfile read the remote zip's file list (it only reads the end of the file)."""

    def __init__(self, rz):
        self.rz, self.pos = rz, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.rz.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.rz.size - self.pos)
        if n <= 0:
            return 0
        data = self.rz.read_range(self.pos, self.pos + n - 1)
        b[:n] = data
        self.pos += n
        return n


def plan_set(rz, set_name, n_per_class, seed):
    """Pick tiles: n random per class (None = all). Returns [(member, source_class, label, rel_path)]."""
    by_class = defaultdict(list)
    for m in rz.members:
        src = Path(m.filename).parent.name
        if src not in STARC_TO_CLASS:
            raise ValueError(f"unknown STARC-9 class folder {src!r} in {m.filename}")
        by_class[src].append(m)
    picked = []
    for src, members in sorted(by_class.items()):
        members.sort(key=lambda m: m.filename)
        if n_per_class is not None and n_per_class < len(members):
            members = sorted(random.Random(f"{seed}-{set_name}-{src}").sample(members, n_per_class),
                             key=lambda m: m.filename)
        label = STARC_TO_CLASS[src]
        picked += [(m, src, label, f"{set_name}/{label}/{Path(m.filename).name}") for m in members]
    return picked


def fetch_and_save(rz, member, out_path, size):
    im = Image.open(io.BytesIO(rz.read_member(member))).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)  # same resize as src/slides.py read_patch
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    im.save(tmp, format="PNG")
    tmp.replace(out_path)  # only finished tiles get the real name, so re-runs never keep half files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/datasets/starc9")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--train-per-class", type=int, default=10000, help="random training tiles per class")
    ap.add_argument("--val-per-class", type=int, default=None, help="test tiles per class (default: all)")
    ap.add_argument("--size", type=int, default=128, help="output px; 128 = 0.5 microns/px, 256 = original")
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info_path = out / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update({"source": "STARC-9 (Path2AI/STARC-9 on Hugging Face), CC-BY 4.0", "revision": REVISION,
                 "source_mpp": SOURCE_MPP, "tile_px": args.size, "mpp": SOURCE_MPP * 256 / args.size,
                 "color_normalized": "Macenko (by the dataset authors)", "seed": args.seed,
                 "label_map": STARC_TO_CLASS})
    info.setdefault("sets", {})

    for set_name in args.sets:
        n = args.train_per_class if set_name == "train" else args.val_per_class
        rows, failed = [], []
        for src, url in SETS[set_name].items():
            print(f"[{set_name}] reading file list of {url.rsplit('/', 1)[-1]} ...", flush=True)
            rz = RemoteZip(url)
            todo = plan_set(rz, set_name, n, args.seed)
            rows += todo
            missing = [t for t in todo if not (out / t[3]).exists()]
            print(f"[{set_name}] {len(todo)} tiles picked, {len(todo) - len(missing)} already on disk", flush=True)
            with ThreadPoolExecutor(args.workers) as pool:
                futures = {pool.submit(fetch_and_save, rz, m, out / rel, args.size): rel
                           for m, _, _, rel in missing}
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"{set_name} {src}", unit="tile"):
                    if f.exception() is not None:
                        failed.append((futures[f], repr(f.exception())))

        done = [r for r in rows if (out / r[3]).exists()]
        with open(out / set_name / "manifest.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "set", "label", "source_class", "source_member"])
            w.writerows([rel, set_name, label, src, m.filename] for m, src, label, rel in done)
        counts = defaultdict(int)
        for _, _, label, _ in done:
            counts[label] += 1
        info["sets"][set_name] = {"n_tiles": len(done), "per_class": dict(sorted(counts.items())),
                                  "per_class_requested": n, "updated": str(date.today())}
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"[{set_name}] done: {len(done)} tiles, {len(failed)} failed", flush=True)
        for rel, err in failed[:10]:
            print(f"   failed: {rel}: {err}")
        if failed:
            print("   (re-run the same command to retry the failed tiles)")


if __name__ == "__main__":
    main()
