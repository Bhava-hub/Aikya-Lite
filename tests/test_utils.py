import numpy as np
import pytest

from utils import find_optimal_thresholds, total_cost


def _make_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    probs = np.clip(rng.random(n) * 0.6 + y * rng.random(n) * 0.4, 0.001, 0.999)
    amounts = rng.uniform(1, 500, n)
    return probs, y, amounts


def test_total_cost_hand_computed():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])
    amounts = np.array([100.0, 50.0, 20.0, 10.0])
    total, fn, fp = total_cost(y_true, y_pred, amounts)
    assert fn == pytest.approx(100 * 3.75)  # one missed fraud
    assert fp == pytest.approx(32.50)       # one false alarm
    assert total == pytest.approx(375 + 32.50)


def test_perfect_predictions_cost_nothing():
    y = np.array([1, 0, 1, 0])
    total, _, _ = total_cost(y, y.copy(), np.array([10.0, 20.0, 30.0, 40.0]))
    assert total == 0


def test_flagging_nothing_costs_all_fraud():
    y = np.array([1, 0, 1, 0])
    amounts = np.array([10.0, 20.0, 30.0, 40.0])
    total, fn, fp = total_cost(y, np.zeros(4, dtype=int), amounts)
    assert fp == 0
    assert fn == pytest.approx((10 + 30) * 3.75)
    assert total == pytest.approx(150.0)  # hand-computed: 150 + 0


def test_flagging_everything_costs_only_false_alarms():
    y = np.array([1, 0, 1, 0])
    total, fn, fp = total_cost(y, np.ones(4, dtype=int), np.array([10.0, 20.0, 30.0, 40.0]))
    assert fn == 0
    assert fp == pytest.approx(2 * 32.50)
    assert total == pytest.approx(65.0)  # hand-computed: 0 + 65


def test_optimal_threshold_matches_brute_force():
    probs, y, amounts = _make_data()
    brute_best = min(
        total_cost(y, (probs >= t).astype(int), amounts)[0] for t in np.unique(probs)
    )
    result = find_optimal_thresholds(probs, y, amounts)
    assert result["Cost-optimal"]["total_cost"] == pytest.approx(brute_best)


def test_reported_threshold_reproduces_reported_cost():
    probs, y, amounts = _make_data()
    result = find_optimal_thresholds(probs, y, amounts)
    for name in ("F1-optimal", "Cost-optimal"):
        r = result[name]
        preds = (probs >= r["threshold"]).astype(int)
        assert total_cost(y, preds, amounts)[0] == pytest.approx(r["total_cost"])


def test_cost_optimal_never_costs_more_than_f1_optimal():
    # The exact bug the dashboard showed: this must hold for any data
    for seed in range(20):
        probs, y, amounts = _make_data(seed=seed)
        r = find_optimal_thresholds(probs, y, amounts)
        assert r["Cost-optimal"]["total_cost"] <= r["F1-optimal"]["total_cost"] + 1e-9


def test_tied_probabilities_do_not_crash():
    probs = np.ones(10)  # everything saturated at 1.0, like the real model
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    result = find_optimal_thresholds(probs, y, np.full(10, 100.0))
    assert result["Cost-optimal"]["threshold"] == 1.0
    assert result["Cost-optimal"]["recall"] == 1.0