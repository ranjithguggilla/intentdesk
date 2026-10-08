"""Text features: word n-grams plus character n-grams, fitted inside the pipeline."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


def build_features(kind: str = "both") -> FeatureUnion:
    parts = []
    if kind in ("word", "both"):
        parts.append(("word", word_vectorizer()))
    if kind in ("char", "both"):
        parts.append(("char", char_vectorizer()))
    if not parts:
        raise ValueError(f"unknown feature kind {kind!r}")
    return FeatureUnion(parts)
