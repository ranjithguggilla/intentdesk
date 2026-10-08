"""Temperature scaling and abstention policy for routing."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from .config import CostModel


def fit_temperature(scores: np.ndarray, y_idx: np.ndarray) -> float:
    """Find T minimising validation log loss of softmax(scores / T)."""

    def nll(log_t: float) -> float:
        logp = log_softmax(scores / np.exp(log_t), axis=1)
        return float(-logp[np.arange(len(y_idx)), y_idx].mean())

    res = minimize_scalar(nll, bounds=(-6, 6), method="bounded")
    return float(np.exp(res.x))


def probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(scores / temperature, axis=1)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def coverage_curve(conf: np.ndarray, correct: np.ndarray, grid: np.ndarray | None = None):
    """Accuracy on the auto-routed share as the confidence threshold rises."""
    grid = np.linspace(0.0, 0.99, 100) if grid is None else grid
    cov, acc = [], []
    for t in grid:
        m = conf >= t
        cov.append(m.mean())
        acc.append(correct[m].mean() if m.any() else np.nan)
    return grid, np.array(cov), np.array(acc)


def policy_cost(conf: np.ndarray, correct: np.ndarray, t: float, c: CostModel) -> float:
    """Average cost per query when confidence >= t is auto-routed."""
    auto = conf >= t
    cost = np.where(
        auto,
        c.auto_route + (~correct) * c.misroute,
        c.human_triage + (1 - c.human_accuracy) * c.misroute,
    )
    return float(cost.mean())


def best_threshold(conf: np.ndarray, correct: np.ndarray, c: CostModel) -> float:
    grid = np.round(np.linspace(0.0, 0.99, 100), 2)
    costs = [policy_cost(conf, correct, t, c) for t in grid]
    return float(grid[int(np.argmin(costs))])
