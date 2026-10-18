"""
Embed every labeled tile set used by the backbone comparison with CONCH and/or MUSK, loading each
model only ONCE (MUSK takes ~30 s to load; doing it per set wastes GPU time).

Writes data/embeddings/<model>/<set>.npz (same format as scripts/models/embed_tiles.py). Sets whose folder
isn't there yet (e.g. SPIDER still downloading) are skipped; sets already embedded are skipped, so the
job can simply be re-run.

    python scripts/models/embed_many.py --models conch musk                 # everything that's downloaded
    python scripts/models/embed_many.py --models musk --sets kather2016 --limit 5    # quick test
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config  # first, so HF_HOME from .env is applied

import numpy as np
import torch

from embed_tiles import MODELS, embed_items, list_tiles, load_model, save_npz
from src.classifier import LABEL_NAMES
from src.models import build_text_classifier

STRUCTURE_LABELS = ["other", "nerve", "vessel"]
KATHER_LABELS = ["01_TUMOR", "02_STROMA", "03_COMPLEX", "04_LYMPHO", "05_DEBRIS", "06_MUCOSA",
                 "07_ADIPOSE", "08_EMPTY"]
NCT_DEFAULT = "/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples"

# set name: (folder, labels). {d} = --datasets-dir, {nct} = --nct. labels None = our tissue classes + blood
SETS = {
    "nct_100k": ("{nct}", None),
    "starc9_train": ("{d}/starc9/train", None),
    "starc9_val_small": ("{d}/starc9/val_stanford_small", None),
    "starc9_val_large": ("{d}/starc9/val_stanford_large", None),
    "starc9_val_tcga": ("{d}/starc9/val_tcga", None),
    "crc_val_7k": ("{d}/CRC-VAL-HE-7K", None),
    "kather2016": ("{d}/kather2016/Kather_texture_2016_image_tiles_5000", KATHER_LABELS),
    "spider_colon_train": ("{d}/spider_colorectal/train", STRUCTURE_LABELS),
    "spider_colon_test": ("{d}/spider_colorectal/test", STRUCTURE_LABELS),
    "spider_skin_train": ("{d}/spider_skin/train", STRUCTURE_LABELS),
    "spider_skin_test": ("{d}/spider_skin/test", STRUCTURE_LABELS),
}
BATCH = {"conch": 256, "musk": 128}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    ap.add_argument("--datasets-dir", default="data/datasets")
    ap.add_argument("--nct", default=NCT_DEFAULT)
    ap.add_argument("--out", default="data/embeddings")
    ap.add_argument("--limit", type=int, default=None, help="max tiles per class (quick tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conch", default=None, help="local CONCH checkpoint (default: Hugging Face cache)")
    ap.add_argument("--force", action="store_true", help="re-embed sets that already exist")
    args = ap.parse_args()

    device = config.get_device()
    timing = {}
    for model_name in args.models:
        todo = []
        for name in args.sets:
            folder, labels = SETS[name]
            root = Path(folder.format(d=args.datasets_dir, nct=args.nct))
            out = Path(args.out) / model_name / f"{name}.npz"
            if out.exists() and not args.force:
                print(f"[{model_name}] {name}: already done ({out})")
            elif not root.is_dir():
                print(f"[{model_name}] {name}: skipped, folder not found ({root})")
            else:
                todo.append((name, root, labels or LABEL_NAMES, out))
        if not todo:
            continue

        t0 = time.time()
        model, preprocess, size = load_model(model_name, device, args.conch)
        text_emb = build_text_classifier(model)
        print(f"[{model_name}] loaded in {time.time() - t0:.0f} s")
        for name, root, label_names, out in todo:
            items = list_tiles(root, label_names, args.limit)
            counts = np.bincount([y for _, y in items], minlength=len(label_names))
            print(f"[{model_name}] {name}: {len(items)} tiles: "
                  + ", ".join(f"{n} {c}" for c, n in zip(label_names, counts) if n), flush=True)
            t1 = time.time()
            emb = embed_items(model, preprocess, items, device, BATCH[model_name], args.workers, desc=name)
            seconds = time.time() - t1
            save_npz(out, items, root, label_names, emb, text_emb, model, model_name, size)
            timing[f"{model_name}/{name}"] = {"tiles": len(items), "seconds": round(seconds, 1),
                                              "tiles_per_s": round(len(items) / max(seconds, 1e-9), 1)}
            print(f"[{model_name}] saved {out}: {seconds:.0f} s ({timing[f'{model_name}/{name}']['tiles_per_s']} tiles/s)",
                  flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    log = Path(args.out) / "timing.json"
    old = json.loads(log.read_text()) if log.exists() else {}
    log.write_text(json.dumps({**old, **timing}, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":  # needed on Windows: DataLoader workers re-import this file
    main()
