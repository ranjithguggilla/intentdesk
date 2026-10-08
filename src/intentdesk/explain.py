"""Plain-language reasons for a routing decision, taken from linear weights."""

from __future__ import annotations

import numpy as np


def top_terms(pipe, text: str, class_index: int, k: int = 4) -> list[str]:
    """Word-level n-grams in `text` that pushed hardest toward the class."""
    feats = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    word = feats.transformer_list[0][1]
    x = word.transform([text])
    names = word.get_feature_names_out()
    coef = np.asarray(clf.coef_)
    row = coef[class_index] if coef.shape[0] > 1 else coef[0]
    idx = x.indices
    if len(idx) == 0:
        return []
    contrib = x.data * row[idx]
    order = np.argsort(contrib)[::-1][:k]
    return [names[idx[i]] for i in order if contrib[i] > 0]


def reason(intent: str, terms: list[str], confidence: float, routed: bool) -> str:
    label = intent.replace("_", " ")
    if terms:
        quoted = ", ".join(f'"{t}"' for t in terms)
        why = f"the wording ({quoted}) points to {label}"
    else:
        why = f"the overall phrasing is closest to {label}"
    tail = "sent straight to the queue" if routed else "held for a person to confirm"
    return f"{why}; confidence {confidence:.0%}, {tail}."
