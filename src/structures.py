"""
Side classifier for small structures that matter for risk but are not tissue-map classes:
nerves (perineural invasion) and blood vessels (venous / lymphovascular invasion).

It works on the same CONCH image embeddings as the tissue map and gives, per tile:
  p_nerve, p_vessel, p_other   logistic regression trained on SPIDER tiles (scripts/train_structures.py)
  p_artery                     zero-shot CONCH, artery vs vein, only meaningful where p_vessel is high.
                               EXPERIMENTAL: no labeled artery/vein data exists to train or test it.
Results go to structures.csv next to tiles.csv (see CLAUDE.md data contract), never into the 8 classes.

    bundle = load_structures("data/models/conch_structures_v1.joblib")
    out = predict_structures(bundle, image_embeddings)   # dict of numpy arrays, one value per tile
"""
import joblib
import numpy as np
import torch

from src.prompts import CLASS_NAMES, CLASSES, TEMPLATES

STRUCTURE_LABELS = ["other", "nerve", "vessel"]

# Zero-shot prompt sets. "other" is covered by our 8 tissue classes plus a few skin tissues
# (the training data includes skin), whose probabilities are summed into p_other.
STRUCTURE_NAMES = {
    "nerve": ["peripheral nerve", "nerve bundle", "nerve fascicles", "small nerve twig"],
    "vessel": ["blood vessel", "small blood vessels", "artery", "vein", "capillaries"],
}
EXTRA_OTHER_NAMES = {
    "epidermis": ["epidermis", "squamous epithelium"],
    "collagen": ["dermal collagen", "dense collagen"],
}
VESSEL_TYPE_NAMES = {
    "artery": ["artery", "muscular artery with a thick wall", "arteriole", "artery with a round lumen and thick media"],
    "vein": ["vein", "thin-walled vein", "venule", "vein with a thin wall and wide irregular lumen"],
}


def prompt_sets(names):
    return {k: [t.format(n) for n in v for t in TEMPLATES] for k, v in names.items()}


@torch.inference_mode()
def encode_prompt_sets(model, sets):
    """{name: [prompts]} -> [n_sets, 512] averaged, normalized CONCH text embeddings (dict order)."""
    from conch.open_clip_custom import get_tokenizer
    from src.models import tokenize

    tokenizer = get_tokenizer()
    device = next(model.parameters()).device
    rows = []
    for prompts in sets.values():
        emb = model.encode_text(tokenize(tokenizer, prompts).to(device)).mean(dim=0)
        rows.append(emb / emb.norm())
    return torch.stack(rows).float().cpu().numpy()


def zero_shot_sets():
    """Prompt sets for zero-shot nerve / vessel / other: names in order, and which ones count as other."""
    sets = {**prompt_sets(STRUCTURE_NAMES),
            **prompt_sets({c: CLASS_NAMES[c] for c in CLASSES}),
            **prompt_sets(EXTRA_OTHER_NAMES)}
    return sets


def softmax(logits):
    logits = logits - logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def zero_shot_structures(embeddings, text_emb, names, logit_scale):
    """Zero-shot p_other / p_nerve / p_vessel ([n, 3], STRUCTURE_LABELS order); other = sum of the rest."""
    p = softmax(logit_scale * embeddings @ text_emb.T)
    nerve, vessel = p[:, names.index("nerve")], p[:, names.index("vessel")]
    return np.stack([1 - nerve - vessel, nerve, vessel], axis=1)


def load_structures(path):
    return joblib.load(path)


def predict_structures(bundle, embeddings):
    """Per-tile structure probabilities from CONCH embeddings [n, 512]."""
    p = np.zeros((len(embeddings), len(STRUCTURE_LABELS)))
    p[:, bundle["clf"].classes_] = bundle["clf"].predict_proba(embeddings)
    pv = softmax(bundle["logit_scale"] * embeddings @ bundle["vessel_type_text"].T)
    return {"p_other": p[:, 0], "p_nerve": p[:, 1], "p_vessel": p[:, 2],
            "structure": np.array(STRUCTURE_LABELS)[p.argmax(1)],
            "p_artery": pv[:, bundle["vessel_type_names"].index("artery")]}
