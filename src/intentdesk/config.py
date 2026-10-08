"""Paths, seeds and the cost assumptions used for the business numbers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
ASSET_DIR = ROOT / "assets"

SEED = 7
VAL_FRACTION = 0.2


@dataclass(frozen=True)
class CostModel:
    """Per-query costs in dollars. These are assumptions, not measurements.

    A query that is auto-routed to the wrong queue gets bounced between teams
    before somebody fixes it. A query sent to the human triage queue is read
    and routed by a person, who is right most of the time but not always.
    """

    auto_route: float = 0.05
    human_triage: float = 1.20
    misroute: float = 6.00
    human_accuracy: float = 0.95
    queries_per_year: int = 1_000_000


COSTS = CostModel()
