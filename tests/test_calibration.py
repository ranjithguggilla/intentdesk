import numpy as np

from intentdesk.calibration import (
    best_threshold,
    coverage_curve,
    expected_calibration_error,
    fit_temperature,
    policy_cost,
    probabilities,
)
from intentdesk.config import CostModel


def test_temperature_shrinks_overconfident_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    sc = rng.normal(size=(600, 3)) * 3
    sc[np.arange(600), y] += 2.5  # right class usually wins, but scores are far too sharp
    t = fit_temperature(sc, y)
    assert t > 1.0


def test_probabilities_rows_sum_to_one():
    p = probabilities(np.random.default_rng(1).normal(size=(5, 4)), 0.7)
    assert np.allclose(p.sum(1), 1)


def test_ece_zero_when_perfectly_calibrated():
    conf = np.full(1000, 0.8)
    correct = np.array([True] * 800 + [False] * 200)
    assert expected_calibration_error(conf, correct) < 1e-9


def test_coverage_curve_monotone_coverage():
    conf = np.linspace(0.1, 0.99, 50)
    correct = conf > 0.5
    grid, cov, acc = coverage_curve(conf, correct)
    assert np.all(np.diff(cov) <= 1e-12)
    assert acc[np.nanargmax(grid > 0.6)] == 1.0


def test_policy_cost_extremes():
    c = CostModel()
    conf = np.array([0.9, 0.9])
    correct = np.array([True, False])
    assert policy_cost(conf, correct, 0.0, c) == (c.auto_route * 2 + c.misroute) / 2
    human = c.human_triage + (1 - c.human_accuracy) * c.misroute
    assert policy_cost(conf, correct, 1.0, c) == human


def test_best_threshold_sends_unreliable_queries_to_humans():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.05, 1, 4000)
    correct = rng.uniform(size=4000) < conf  # confidence is honest
    t = best_threshold(conf, correct, CostModel())
    # break-even: accept when error cost is below the human-queue cost
    assert 0.55 < t < 0.9
