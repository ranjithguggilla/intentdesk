"""Label-quality audit with confident learning (cleanlab)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def out_of_fold_probs(pipe, texts, labels, seed: int, folds: int = 5) -> np.ndarray:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(clone(pipe), texts, labels, cv=cv, method="predict_proba", n_jobs=1)


def flag_issues(pred_probs: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Indices of suspected label errors, worst first."""
    return find_label_issues(
        labels=y_idx,
        pred_probs=pred_probs,
        filter_by="prune_by_noise_rate",
        return_indices_ranked_by="self_confidence",
    )


def issue_table(df: pd.DataFrame, pred_probs: np.ndarray, issues: np.ndarray, classes) -> pd.DataFrame:
    classes = np.asarray(classes)
    rows = []
    for i in issues:
        rows.append(
            {
                "text": df["text"].iloc[i],
                "given": df["intent"].iloc[i],
                "suggested": classes[int(np.argmax(pred_probs[i]))],
                "p_given": float(pred_probs[i][list(classes).index(df["intent"].iloc[i])]),
                "p_suggested": float(pred_probs[i].max()),
            }
        )
    return pd.DataFrame(rows, columns=["text", "given", "suggested", "p_given", "p_suggested"])


def confused_pairs(y_true: np.ndarray, y_pred: np.ndarray, top: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df[df.true != df.pred]
    df["pair"] = [" <-> ".join(sorted(p)) for p in zip(df.true, df.pred, strict=True)]
    out = df.groupby("pair").size().sort_values(ascending=False).head(top)
    return out.rename("errors").reset_index()
