"""Candidate models and the comparison harness."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_features


def candidates(seed: int) -> dict[str, tuple[str, object]]:
    """name -> (feature kind, estimator)."""
    return {
        "majority class": ("word", DummyClassifier(strategy="prior")),
        "complement NB (word)": ("word", ComplementNB(alpha=0.1)),
        "kNN cosine (k=5)": (
            "both",
            KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance"),
        ),
        "ridge classifier": ("both", RidgeClassifier(alpha=1.0)),
        "SGD modified huber": (
            "both",
            SGDClassifier(loss="modified_huber", alpha=2e-5, random_state=seed, max_iter=60),
        ),
        "linear SVM (word only)": ("word", LinearSVC(C=0.5)),
        "linear SVM (char only)": ("char", LinearSVC(C=0.5)),
        "linear SVM (word+char)": ("both", LinearSVC(C=0.5)),
        "logistic regression (word+char)": (
            "both",
            LogisticRegression(C=30, max_iter=200),
        ),
    }


def make_pipeline(kind: str, estimator) -> Pipeline:
    return Pipeline([("features", build_features(kind)), ("clf", estimator)])


def compare(train: pd.DataFrame, val: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fit every candidate on train, score on the validation fold."""
    rows = []
    for name, (kind, est) in candidates(seed).items():
        pipe = make_pipeline(kind, est)
        t0 = time.perf_counter()
        pipe.fit(train["text"], train["intent"])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = pipe.predict(val["text"])
        ms = (time.perf_counter() - t0) / len(val) * 1000
        rows.append(
            {
                "model": name,
                "val_accuracy": accuracy_score(val["intent"], pred),
                "val_macro_f1": f1_score(val["intent"], pred, average="macro"),
                "fit_seconds": fit_s,
                "predict_ms_per_query": ms,
            }
        )
    return pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)


def scores(pipe: Pipeline, texts) -> np.ndarray:
    """Raw per-class scores, shape (n, n_classes), columns in pipe.classes_ order."""
    clf = pipe.named_steps["clf"]
    feats = pipe.named_steps["features"].transform(texts)
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(feats))
    return np.log(np.clip(clf.predict_proba(feats), 1e-12, None))
