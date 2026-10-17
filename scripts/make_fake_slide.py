"""
Write fake slide result folders, so the analysis and the app can be built and tested before real
slides are processed on Delta. No GPU, no model.

    python scripts/make_fake_slide.py                 # all variants
    python scripts/make_fake_slide.py --variant pushing

Writes data/results/fake_<variant>/: tiles.csv, phrase_scores.csv, summary.json (slide info only),
thumbnail.png. Then run:  python scripts/analyze_slide.py data/results/fake_infiltrative
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.fake_slide import VARIANTS, make_fake_slide, thumbnail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS, help="default: all of them")
    ap.add_argument("--out", default=str(config.RESULTS_DIR))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for variant in [args.variant] if args.variant else VARIANTS:
        tiles, scores, meta, truth = make_fake_slide(variant)
        out = Path(args.out) / meta["slide_id"]
        out.mkdir(parents=True, exist_ok=True)
        tiles.to_csv(out / "tiles.csv", index=False)
        scores.to_csv(out / "phrase_scores.csv", index=False)
        (out / "summary.json").write_text(json.dumps(meta, indent=2))
        plt.imsave(out / "thumbnail.png", thumbnail(truth["label"]))
        print(f"wrote {out}  ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
