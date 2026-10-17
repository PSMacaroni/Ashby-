"""
Text phrases for the "look here" flags (section 4 of the plan), and how tiles are scored against them.

A phrase score is a text match: how close a tile's image embedding is to the phrase's text embedding
(cosine similarity), averaged over the positive phrases, minus the average over the negative phrases
(look-alikes). Higher = looks more like the phrase. It's a hint, never a measurement.

The tile embeddings and the phrases must come from the SAME model (CONCH image + CONCH text).

    scores = score_phrases(tile_embeddings, model)   # {"budding": [N], "nerve": [N], ...}
"""
import numpy as np

# key: (positive phrases, negative phrases). Wording from our features doc.
FLAG_PHRASES = {
    "budding": (["single tumor cells and small clusters of tumor cells budding at the invasive front",
                 "tumor budding in desmoplastic stroma"],
                ["intact tumor glands", "stroma without tumor cells"]),
    "lymphatic": (["cluster of tumor cells inside a thin-walled lymphatic space lined by endothelium"],
                  ["retraction artifact around tumor nests"]),
    "vein": (["tumor nodule inside a vein with a smooth muscle wall",
              "tumor in a vessel lumen next to a muscular artery"], []),
    "artery": (["thick-walled muscular artery in pericolic fat"], []),       # fallback if no artery detector
    "nerve": (["peripheral nerve bundle with wavy fibers"], ["smooth muscle bundle"]),  # fallback if no detector
    "pni": (["tumor cells surrounding a nerve"], []),
    "stroma_mature": (["mature fibrous stroma with fine collagen fibers"], []),
    "stroma_keloid": (["keloid-like thick hyalinized collagen bundles"], []),
    "stroma_immature": (["immature myxoid desmoplastic stroma"], []),
    "adenoma": (["tubular adenoma with low-grade dysplasia next to carcinoma"], []),
    # quality checks, for the tiling step
    "pen": (["pen ink marking"], []),
    "fold": (["tissue fold"], []),
    "blur": (["out of focus"], []),
}
PHRASES_VERSION = "v1"


def score_phrases(tile_embeddings, model, phrases=FLAG_PHRASES):
    """
    tile_embeddings: numpy [N, dim], normalized (what classify_images / encode_image return).
    Returns {key: numpy [N] score}. Needs the model's text side only (fine on a laptop CPU).
    """
    from src.models import encode_texts

    emb = np.asarray(tile_embeddings, dtype=np.float32)
    scores = {}
    for key, (positives, negatives) in phrases.items():
        pos = encode_texts(model, positives).float().cpu().numpy()
        score = (emb @ pos.T).mean(axis=1)
        if negatives:
            neg = encode_texts(model, negatives).float().cpu().numpy()
            score = score - (emb @ neg.T).mean(axis=1)
        scores[key] = score
    return scores
