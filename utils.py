import numpy as np

FN_MULTIPLIER = 3.75  # LexisNexis: each $1 of fraud costs ~$3.75 in total
FP_COST = 32.50       # flat cost to investigate one false alarm


def total_cost(y_true, y_pred, amounts, fn_multiplier=FN_MULTIPLIER, fp_cost=FP_COST):
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_cost = (amounts[fn_mask] * fn_multiplier).sum()
    fp_cost_total = fp_mask.sum() * fp_cost
    return fn_cost + fp_cost_total, fn_cost, fp_cost_total


def find_optimal_thresholds(probs, y, amounts, fn_multiplier=FN_MULTIPLIER, fp_cost=FP_COST):
    """Return F1-optimal and cost-optimal thresholds, computed from the given probabilities."""
    order = np.argsort(-probs)
    p_sorted, y_sorted, amt_sorted = probs[order], y[order], amounts[order]

    # Only cut where the probability changes, so tied scores are never split
    last_of_group = np.append(p_sorted[:-1] != p_sorted[1:], True)

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    fraud_amt_caught = np.cumsum(np.where(y_sorted == 1, amt_sorted, 0.0))
    total_fraud = (y == 1).sum()
    total_fraud_amt = amounts[y == 1].sum()
    flagged = np.arange(1, len(y) + 1)

    fn_cost = (total_fraud_amt - fraud_amt_caught) * fn_multiplier
    fp_cost_arr = fp * fp_cost
    cost = np.where(last_of_group, fn_cost + fp_cost_arr, np.inf)
    f1 = np.where(last_of_group, 2 * tp / (flagged + total_fraud), -np.inf)

    results = {}
    for name, i in [("F1-optimal", int(np.argmax(f1))), ("Cost-optimal", int(np.argmin(cost)))]:
        results[name] = {
            "threshold": float(p_sorted[i]),
            "total_cost": float(cost[i]),
            "fn_cost": float(fn_cost[i]),
            "fp_cost": float(fp_cost_arr[i]),
            "recall": float(tp[i] / total_fraud),
        }
    return results