"""
Score every tile of a slide against the flag phrases (src/phrases.py) and save phrase_scores.csv,
which scripts/analyze_slide.py uses for the "look here" flags.

Needs the slide's saved tile embeddings (embeddings.npy, rows in the same order as tiles.csv) and the
text side of the SAME model that made them. Text encoding is light: a laptop CPU is fine.

    python scripts/score_phrases.py data/results/<slide_id> --model conch
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: F401  (reads .env, applies HF_HOME)

import numpy as np
import pandas as pd

from src.phrases import PHRASES_VERSION, score_phrases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="data/results/<slide_id>")
    ap.add_argument("--model", choices=["conch", "musk"], default="conch",
                    help="must be the model that made embeddings.npy")
    ap.add_argument("--conch", help="optional: path to a CONCH pytorch_model.bin")
    args = ap.parse_args()

    folder = Path(args.folder)
    tiles = pd.read_csv(folder / "tiles.csv")
    emb = np.load(folder / "embeddings.npy").astype(np.float32)
    if len(emb) != len(tiles):
        sys.exit(f"embeddings.npy has {len(emb)} rows but tiles.csv has {len(tiles)}: they must match row for row")

    from src.models import load_conch, load_musk

    model = load_musk(device="cpu")[0] if args.model == "musk" else load_conch(args.conch, device="cpu")[0]
    scores = score_phrases(emb, model)
    out = pd.DataFrame({"row": tiles["row"], "col": tiles["col"], **{k: np.round(v, 5) for k, v in scores.items()}})
    out.to_csv(folder / "phrase_scores.csv", index=False)

    meta_path = folder / "summary.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.update({"phrases_version": PHRASES_VERSION, "embeddings_model": args.model.upper()})
        meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {folder / 'phrase_scores.csv'} ({len(out)} tiles x {len(scores)} phrases)")


if __name__ == "__main__":
    main()
