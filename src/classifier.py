"""
Small tissue classifier: logistic regression on CONCH image embeddings (trained by scripts/train_classifier.py).

It knows our 8 CLASSES plus "blood". Blood is not a map class: the classifier only recognizes it so
the pipeline can ignore those tiles (qc_ok = False, qc_reason = "blood") instead of calling them
necrosis. The 8 tissue probabilities are re-normalized without blood, so they still sum to 1.

    clf = load_classifier("data/models/conch_lr_starc9.joblib")
    probs, is_blood = predict(clf, image_embeddings)   # embeddings from classify_images()
"""
import joblib
import numpy as np

from src.prompts import CLASSES

LABEL_NAMES = CLASSES + ["blood"]
BLOOD = LABEL_NAMES.index("blood")


def load_classifier(path):
    return joblib.load(path)


def predict_all(clf, embeddings):
    """Probabilities for all 9 labels [n, 9] (LABEL_NAMES order). Labels the model never saw get 0."""
    p = np.zeros((len(embeddings), len(LABEL_NAMES)))
    p[:, clf.classes_] = clf.predict_proba(embeddings)
    return p


def predict(clf, embeddings):
    """
    Returns (probs, is_blood):
      probs    [n, 8] tissue probabilities in CLASSES order, each row sums to 1
      is_blood [n] True where blood is the most likely label (pipeline: skip these tiles)
    """
    p = predict_all(clf, embeddings)
    is_blood = p.argmax(1) == BLOOD
    tissue = p[:, :len(CLASSES)]
    return tissue / tissue.sum(1, keepdims=True).clip(min=1e-12), is_blood
