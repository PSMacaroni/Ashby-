"""
Compute CONCH or MUSK image embeddings for a folder of labeled tiles, once, so classifiers can be trained
and tested later on CPU in seconds.

Input: a folder with one subfolder per class, e.g.
    NCT-CRC-HE-100K/<TUM|NORM|...>/*.tif     (NCT codes, BACK is skipped)
    data/datasets/starc9/train/<tumor|...|blood>/*.png
Folder names can be NCT codes, STARC-9 codes, or our class names. "blood" is kept as an extra
label (index 8, after the 8 CLASSES): we don't map it, we only use it to recognize and ignore blood.

Output: <out>/<name>.npz with
    embeddings  float32 [N, dim]   image embeddings (normalized, same as classify_images; CONCH 512, MUSK 1024)
    labels      int     [N]        index into label_names
    label_names str     [9]        CLASSES + ["blood"]
    paths       str     [N]        tile paths relative to --root
    text_emb    float32 [8, dim]   zero-shot text classifier (prompts in src/prompts.py)
    logit_scale float              so zero-shot probs = softmax(logit_scale * embeddings @ text_emb.T)
    model       str                "conch" or "musk"

    python scripts/embed_tiles.py --root data/datasets/starc9/train --name starc9_train --limit 5   # quick test
    python scripts/embed_tiles.py --root data/datasets/starc9/train --name starc9_train
    python scripts/embed_tiles.py --model musk --out data/embeddings/musk --root ... --name ...

To embed many folders with one model load (saves GPU time), use scripts/embed_many.py.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # first, so HF_HOME from .env is applied

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.classifier import LABEL_NAMES
from src.datasets import NCT_TO_CLASS, STARC_TO_CLASS, load_rgb
from src.models import build_text_classifier, load_conch, load_musk
from src.prompts import PROMPTS_VERSION

FOLDER_TO_LABEL = {**NCT_TO_CLASS, **STARC_TO_CLASS, **{c: c for c in LABEL_NAMES}}
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
MODELS = ("conch", "musk")


def list_tiles(root, label_names, limit=None):
    """[(path, label_index)] for every image in root/<class>/; skips folders we don't use (e.g. BACK)."""
    items = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        label = folder.name if label_names is not LABEL_NAMES else FOLDER_TO_LABEL.get(folder.name)
        if label not in label_names:
            print(f"  skipping folder {folder.name!r} (not one of the labels)")
            continue
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        items += [(f, label_names.index(label)) for f in files[:limit]]
    return items


class Tiles(Dataset):
    def __init__(self, items, preprocess):
        self.items, self.preprocess = items, preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.preprocess(load_rgb(path)), label


def load_model(name, device, conch_checkpoint=None, image_size=224):
    """(model, preprocess, input size) for "conch" (at image_size) or "musk" (always 384)."""
    if name == "musk":
        model, preprocess = load_musk(device=device)
        return model, preprocess, 384
    model, preprocess = load_conch(conch_checkpoint, device=device, image_size=image_size)
    return model, preprocess, image_size


def embed_items(model, preprocess, items, device, batch_size=256, workers=8, desc="embedding"):
    """Image embeddings [N, dim] (float32) for [(path, label)] items, in order."""
    loader = DataLoader(Tiles(items, preprocess), batch_size=batch_size, num_workers=workers,
                        pin_memory=device == "cuda")
    embs = []
    with torch.inference_mode():
        for x, _ in tqdm(loader, desc=f"{desc} on {device}", unit="batch"):
            emb = model.encode_image(x.to(device, non_blocking=True), proj_contrast=True, normalize=True)
            embs.append(emb.float().cpu().numpy())
    return np.concatenate(embs).astype(np.float32)


def save_npz(path, items, root, label_names, embeddings, text_emb, model, model_name, image_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             embeddings=embeddings,
             labels=np.array([y for _, y in items]),
             label_names=np.array(label_names),
             paths=np.array([str(Path(p).relative_to(root)) for p, _ in items]),
             text_emb=text_emb.float().cpu().numpy(),
             logit_scale=model.logit_scale.exp().item(),
             image_size=image_size, prompts_version=PROMPTS_VERSION, model=model_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with one subfolder per class")
    ap.add_argument("--name", required=True, help="output file name (without .npz)")
    ap.add_argument("--model", choices=MODELS, default="conch")
    ap.add_argument("--out", default="data/embeddings")
    ap.add_argument("--limit", type=int, default=None, help="max tiles per class (for quick tests)")
    ap.add_argument("--image-size", type=int, default=224, help="CONCH input size (224 matches the pipeline)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8, help="parallel image readers")
    ap.add_argument("--conch", default=None, help="local CONCH checkpoint (default: download from Hugging Face)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="use these folder names as labels as-is (e.g. other nerve vessel) instead of tissue classes")
    args = ap.parse_args()

    root = Path(args.root)
    label_names = args.labels or LABEL_NAMES
    items = list_tiles(root, label_names, args.limit)
    counts = np.bincount([y for _, y in items], minlength=len(label_names))
    print(f"{len(items)} tiles in {root}: " + ", ".join(f"{n} {c}" for c, n in zip(label_names, counts) if n))

    device = config.get_device()
    model, preprocess, size = load_model(args.model, device, args.conch, args.image_size)
    text_emb = build_text_classifier(model)

    t0 = time.time()
    embeddings = embed_items(model, preprocess, items, device, args.batch_size, args.workers)
    seconds = time.time() - t0

    path = Path(args.out) / f"{args.name}.npz"
    save_npz(path, items, root, label_names, embeddings, text_emb, model, args.model, size)
    print(f"saved {path}: {len(items)} tiles in {seconds:.0f} s ({len(items) / max(seconds, 1e-9):.0f} tiles/s)")


if __name__ == "__main__":  # needed on Windows: DataLoader workers re-import this file
    main()
