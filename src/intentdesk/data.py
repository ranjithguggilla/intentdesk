"""Loading, validating and splitting the BANKING77 query data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, SEED, VAL_FRACTION


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"category": "intent"})
    missing = [c for c in ("text", "intent") if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    df["text"] = df["text"].astype(str).str.strip()
    if df["text"].eq("").any():
        raise ValueError(f"{path.name} contains empty queries")
    return df


def load_intents(data_dir: Path = DATA_DIR) -> list[str]:
    return json.loads((data_dir / "categories.json").read_text())


def load_splits(data_dir: Path = DATA_DIR, seed: int = SEED):
    """Return train, validation and the official test split.

    The validation fold is carved out of the official training file with
    stratification. The test file is never touched until final reporting.
    """
    full = load_frame(data_dir / "train.csv")
    test = load_frame(data_dir / "test.csv")
    train, val = train_test_split(full, test_size=VAL_FRACTION, stratify=full["intent"], random_state=seed)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def exact_overlap(train: pd.DataFrame, other: pd.DataFrame) -> int:
    """Count queries of `other` that also appear verbatim in `train`."""
    norm = lambda s: s.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()  # noqa: E731
    return int(norm(other["text"]).isin(set(norm(train["text"]))).sum())
