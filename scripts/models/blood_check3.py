"""
Blood check, part 3: MUSK vs CONCH vs both on our own tiles (Delta job, GPU if available).

For the same 2,237 tiles as part 2 (subset.csv), this embeds each tile as scanned with MUSK (the same way
scripts/embed_tiles.py on musk-experiments does: MUSK's own preprocessing, 384 px), reuses the CONCH embeddings from
part 1, and runs the models from the MUSK experiment (data/results/compare_backbones/musk_v1/):
tissue8_{conch,musk,both} and structures_{conch,musk,both} ("both" = CONCH and MUSK side by side), plus zero-shot
MUSK with a blood prompt added to our 8 classes.

    ASHBY_REPO=~/Ashby- python blood_check3.py --run <part 1 dir> --subset subset.csv --out <dir>
Writes <out>/musk_results.csv and <out>/subset_musk_emb.npy
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blood_check import BLOOD_NAMES, NAMES9, REPO, config, proba, read_tile  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import openslide  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.prompts import CLASSES, TEMPLATES  # noqa: E402

FEATURES = ("conch", "musk", "both")


class Tiles(Dataset):
    def __init__(self, rows, slides_dir, preprocess):
        self.rows, self.slides_dir, self.preprocess, self.open = rows, slides_dir, preprocess, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        if r["slide"] not in self.open:
            self.open[r["slide"]] = openslide.OpenSlide(str(next(self.slides_dir.glob(r["slide"] + "*.svs"))))
        return self.preprocess(read_tile(self.open[r["slide"]], int(r["x"]), int(r["y"]), int(r["read_px"])))


@torch.inference_mode()
def musk_text(model):
    """MUSK text embeddings for our 8 classes (prompts v1) plus blood, rows in NAMES9 order."""
    from src.models import build_text_classifier

    rows = build_text_classifier(model)
    blood = model.encode_texts([t.format(n) for n in BLOOD_NAMES for t in TEMPLATES]).mean(0)
    return torch.cat([rows, (blood / blood.norm())[None]]).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="part 1 output folder (per-slide emb_raw.npy)")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slides-dir", default="/work/hdd/bfqi/natmakur/tcga_slides")
    ap.add_argument("--exp", default=str(REPO / "data/results/compare_backbones/musk_v1"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    run, out, exp = Path(args.run), Path(args.out), Path(args.exp)
    out.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(args.subset)
    conch = np.zeros((len(sub), 512), np.float32)
    for slide, idx in sub.groupby("slide").groups.items():
        E = np.load(next(run.glob(slide + "*")) / "emb_raw.npy", mmap_mode="r")
        conch[np.asarray(idx)] = E[sub.loc[idx, "i"].to_numpy()]

    from src.models import load_musk

    device = config.get_device()
    model, preprocess = load_musk(device=device)
    loader = DataLoader(Tiles(sub.to_dict("records"), Path(args.slides_dir), preprocess),
                        batch_size=args.batch_size, num_workers=args.workers)
    musk = []
    with torch.inference_mode():
        for x in loader:
            musk.append(model.encode_image(x.to(device), proj_contrast=True, normalize=True).float().cpu().numpy())
            print(f"{sum(len(m) for m in musk)}/{len(sub)}", flush=True)
    musk = np.concatenate(musk)
    np.save(out / "subset_musk_emb.npy", musk)

    feats = {"conch": conch, "musk": musk, "both": np.hstack([conch, musk])}
    for f in FEATURES:
        t8 = joblib.load(exp / f"tissue8_{f}.joblib")["clf"]
        p = proba(t8, feats[f], 8)
        sub[f"t8_label_{f}"] = np.array(CLASSES)[p.argmax(1)]
        sub[f"t8_conf_{f}"] = p.max(1)
        st = joblib.load(exp / f"structures_{f}.joblib")
        labels = list(st["labels"])
        ps = proba(st["clf"], feats[f], len(labels))
        sub[f"struct_label_{f}"] = np.array(labels)[ps.argmax(1)]
        sub[f"struct_p_blood_{f}"] = ps[:, labels.index("blood")]

    text = musk_text(model)
    logits = float(model.logit_scale.exp()) * musk @ text.T
    zs = np.exp(logits - logits.max(1, keepdims=True))
    zs /= zs.sum(1, keepdims=True)
    sub["zs_musk_label"] = np.array(NAMES9)[zs.argmax(1)]
    sub["zs_musk_p_blood"] = zs[:, 8]
    sub.round(4).to_csv(out / "musk_results.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
