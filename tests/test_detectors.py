"""Checks for the blood check helpers in src/detectors.py (no GPU, no slides).  Run: pytest -q"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import cutoffs
from src.detectors import class_probs, is_blood, load_bundle, predict_blood

MODELS = Path(__file__).resolve().parent.parent / "models"


def test_is_blood_needs_the_cutoff_and_spares_fat():
    p = np.array([0.95, 0.95, cutoffs.BLOOD_PROB, cutoffs.BLOOD_PROB - 0.01])
    labels = np.array(["necrosis", "fat", "tumor", "necrosis"])
    assert is_blood(p, labels).tolist() == [True, False, True, False]


def test_class_probs_keeps_label_order_and_fills_unseen_labels():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    clf = LogisticRegression().fit(X, [0, 0, 2, 2])            # label 1 never seen in training
    p = class_probs({"clf": clf, "labels": ["a", "b", "c"]}, X)
    assert p.shape == (4, 3)
    assert np.allclose(p[:, 1], 0) and np.allclose(p.sum(1), 1)
    assert p[0, 0] > p[0, 2] and p[3, 2] > p[3, 0]


def test_saved_musk_models_load_and_predict():
    E = np.random.default_rng(0).normal(size=(5, 1024)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)              # like MUSK's normalized embeddings
    p_blood = predict_blood(load_bundle(MODELS / "structures_musk_v1.joblib"), E)
    assert p_blood.shape == (5,) and np.all((p_blood >= 0) & (p_blood <= 1))
    probs8 = class_probs(load_bundle(MODELS / "tissue8_musk_v1.joblib"), E)
    assert probs8.shape == (5, 8) and np.allclose(probs8.sum(1), 1)
