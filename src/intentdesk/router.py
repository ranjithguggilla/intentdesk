"""Scoring function shared by the app, the CLI and the tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import probabilities
from .explain import reason, top_terms
from .models import scores


@dataclass
class Bundle:
    pipeline: object
    temperature: float
    threshold: float


@dataclass
class Decision:
    text: str
    intent: str
    confidence: float
    routed: bool
    top3: list[tuple[str, float]]
    reason: str


def route(bundle: Bundle, text: str) -> Decision:
    text = (text or "").strip()
    if not text:
        raise ValueError("query is empty")
    pipe = bundle.pipeline
    p = probabilities(scores(pipe, [text]), bundle.temperature)[0]
    order = np.argsort(p)[::-1][:3]
    classes = pipe.classes_
    best = int(order[0])
    conf = float(p[best])
    routed = conf >= bundle.threshold
    terms = top_terms(pipe, text, best)
    return Decision(
        text=text,
        intent=str(classes[best]),
        confidence=conf,
        routed=routed,
        top3=[(str(classes[i]), float(p[i])) for i in order],
        reason=reason(str(classes[best]), terms, conf, routed),
    )
