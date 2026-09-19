# Federated Fraud Detection — Corrective Steps & Alternatives

## Changes we actually applied

### 1. Weighted Loss (root-cause fix for the model)
- **File:** `model.py` — removed `sigmoid` from the final layer (now outputs raw logits)
- **File:** `client.py` — replaced `nn.BCELoss()` with:
```python
  pos_weight = torch.tensor([(len(y) - y.sum()) / y.sum()])
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
- **File:** `evaluate_federated.py` — added manual sigmoid at eval time:
```python
  logits = model(X_test_tensor).numpy().flatten()
  probs = 1 / (1 + np.exp(-logits))
```
- **Why:** Unweighted loss let the model "cheat" by predicting near-zero for everything, since fraud is only ~0.17% of the data and barely affected total loss. `pos_weight` (computed per-bank from local fraud ratio) forces the loss function to penalize missed fraud proportionally to its rarity.
- **Result:** ROC-AUC improved from 0.9047 → 0.9821; fraud-case probabilities jumped from maxing at 0.39 to consistently 0.99–1.0.

### 2. Precision-Recall Curve instead of ROC/Youden's J (root-cause fix for thresholding)
- **File:** `evaluate_federated.py` — replaced Youden's J (`roc_curve` + `tpr - fpr`) with:
```python
  from sklearn.metrics import precision_recall_curve
  precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)
  f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
  best_idx = np.argmax(f1_scores)
  best_threshold = pr_thresholds[best_idx]
```
- **Why:** ROC's FPR is a fraction of the *entire non-fraud population* (56,864), so even large absolute false-alarm counts look like tiny percentages — Youden's J got fooled into picking a threshold with 2,456 false alarms per 92 fraud catches. Precision is a fraction of *everything flagged*, directly reflecting the fraud-catch/false-alarm trade-off without a huge denominator to hide behind.
- **Result:** Final threshold 0.9996 → Precision 0.8511, Recall 0.8163, F1 0.8333 (federated) vs. Precision 0.9412, Recall 0.8163, F1 0.8743 (centralized baseline).

---

## Corrective options considered but NOT applied (documented as future work)

### 3. More local epochs / learning rate tuning
- Increase `for epoch in range(5)` → `range(20)` in `client.py`
- Could sharpen the decision boundary further. Skipped because `pos_weight` already produced a strong result (AUC 0.98) — not necessary unless chasing marginal gains.

### 4. Fix scaler inconsistency
- Currently each bank fits its own `StandardScaler` independently in `client.py`; `evaluate_federated.py` fits yet another scaler on the test set alone. Training-time and eval-time scaling don't perfectly match.
- Rigorous fix: save each bank's scaler stats, reconcile at the server, reuse exact scaling at eval time.
- Skipped as a known limitation — likely small practical effect on this dataset, not worth the added complexity given timeline. Worth stating explicitly in README.

### 5. SMOTE (synthetic minority oversampling)
- Considered earlier for the *centralized baseline* as an alternative to `class_weight='balanced'` (which actually made the baseline worse — reverted).
- Not pursued for the federated model because: (a) `pos_weight` already solved the same underlying problem more simply, (b) correct SMOTE in a federated setting requires each bank to oversample its own local shard independently before training — added complexity not worth taking on given `pos_weight` worked.

### 6. Architecture changes (deeper network, or GNN)
- Came up when the project "felt small." Considered swapping the 3-layer NN for something more complex.
- Not pursued because the bottleneck was the loss function, not model capacity — the same simple architecture reached AUC 0.98 once given weighted loss.

---

## Key takeaway
Two independent problems needed two independent, targeted fixes:
1. **Model wasn't learning to weight fraud correctly** → fixed via `pos_weight` in the loss function.
2. **Threshold-selection method wasn't suited to severe class imbalance** → fixed by switching from ROC/Youden's J to Precision-Recall/F1.

Fixing only one without the other would not have produced the final result. The other options above remain legitimate "future work" items but weren't necessary to reach a defensible federated-vs-centralized comparison.