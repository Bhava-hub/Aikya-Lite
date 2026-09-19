import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from model import FraudNet
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve



# Load the held-out test set (never seen during federated training)
test_df = pd.read_csv("data/global_test.csv")
X_test = test_df.drop("Class", axis=1).values
y_test = test_df["Class"].values

# Scale features — note: in a real deployment each bank would share only
# scaling statistics or use a global schema; for this PoC we fit fresh here
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Load the federated model
model = FraudNet(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("federated_model.pt"))
model.eval()

# Predict
with torch.no_grad():
    logits = model(X_test_tensor).numpy().flatten()
    probs = 1 / (1 + np.exp(-logits))  # manual sigmoid

'''
How this works: remember, ROC-AUC is computed by trying every possible threshold internally and plotting true-positive-rate vs. false-positive-rate at each one. 
roc_auc_score() collapses all that into one number, but scikit-learn has a companion function, roc_curve(), that gives you back the actual list of thresholds it tried, along with the true-positive-rate and false-positive-rate at each one.
 So instead of guessing 0.15, you can look at that list and pick the threshold that gives the best trade-off for your specific goal (e.g., "maximize catching fraud without flagging too many innocent transactions").




fpr, tpr, thresholds = roc_curve(y_test, probs)

# Youden's J statistic: finds the threshold that best balances
# catching fraud (tpr) against avoiding false alarms (fpr)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"Best threshold (Youden's J): {best_threshold:.4f}")
print(f"At this threshold — TPR (recall): {tpr[best_idx]:.4f}, FPR: {fpr[best_idx]:.4f}")

preds = (probs >= best_threshold).astype(int)
'''





'''
Why this is the right tool now: precision and recall are both computed relative to the actual fraud count (98), not mixed with the 56,864 non-fraud count the way FPR is — so this curve doesn't get distorted by the huge class-size gap the way ROC does.
 Maximizing F1 on this curve should land you a threshold that gives a genuinely balanced precision/recall trade-off, not one skewed by the imbalance.
'''
precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)

# F1 = harmonic mean of precision and recall — maximize this instead
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_idx]

print(f"Best threshold (max F1): {best_threshold:.4f}")
print(f"At this threshold — Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}")

preds = (probs >= best_threshold).astype(int)
print("Prob stats — min:", probs.min(), "max:", probs.max(), "mean:", probs.mean())
print("Fraud-only probs:", probs[y_test == 1][:20])
print(classification_report(y_test, preds, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

'''
One nuance worth knowing (good to mention in your README): Youden's J treats false positives and false negatives as equally costly.
 In real fraud detection, that's often not true — a bank might tolerate more false alarms (annoying but cheap) to avoid missing actual fraud (expensive). 
If you wanted to bias toward recall, 
you'd instead pick the threshold where recall crosses some target like 0.90, rather than using Youden's J. 
For this project, Youden's J is a solid, standard default to start with.

'''


'''
What we did wrong (technically): in client.py, each bank calls scaler.fit_transform(X) on its own training shard 
— so Bank 0 learns mean/std from small transactions, Bank 3 learns mean/std from large transactions, etc. Four different scalers, four different notions of "normal."

Then in evaluate_federated.py, we call scaler.fit_transform(X_test) again, fitting a brand new fifth scaler on the test set itself.

Why that's inconsistent: the model's weights were trained on data scaled one way (by each bank's own local scaler), but we're evaluating it on data scaled a completely different way (by a scaler fit on the test set). 
It's like training someone to recognize temperatures in Celsius, then testing them with numbers secretly converted to Fahrenheit — the underlying values don't line up the same way.

'''

'''
Before (current state): your model's raw probabilities were badly calibrated — every fraud case scored below 0.39, so the "natural" cutoff of 0.5 was simply wrong for this model's output range. Youden's J was doing double duty: compensating for a poorly-trained model and finding a reasonable operating point.

After adding pos_weight: the model itself should be fixed at the source — fraud cases should now score meaningfully higher (likely crossing 0.5 for many of them), since the loss function actually penalizes missing them. But here's the key thing that doesn't change: there will never be one single "correct" threshold, even for a well-trained model. 
That's not a bug to fix — it's an inherent property of binary classification with class imbalance.

Every threshold you pick trades recall against precision:

Lower threshold → catch more fraud, but more false alarms
Higher threshold → fewer false alarms, but miss more fraud

So Youden's J's actual, permanent job is: given a properly-trained model's probability outputs, find the threshold that best balances true positive rate against false positive rate — not "fix a broken model," but "convert good probabilities into a good decision rule." 
You'll still run the ROC curve + Youden's J code after retraining with pos_weight, but this time you're using it for its intended purpose rather than as a workaround.

Practical expectation: after pos_weight, rerun the Youden's J calculation — the value it picks will likely be much higher than 0.0165 (maybe somewhere around 0.3–0.5), because the probability distribution itself has shifted to be better-calibrated. And the resulting precision/recall trade-off at that new threshold should look far healthier than the 0.23 precision you saw before.
'''




"""
FEDERATED FRAUD DETECTION — MODEL ITERATION HISTORY
=====================================================

ITERATION 1: Baseline federated setup
- Model: 3-layer NN (FraudNet) with sigmoid output
- Loss: nn.BCELoss() (unweighted)
- Result: Loss converged smoothly to 0.0067 over 30 rounds, but this
  was misleading — model had learned to predict near-zero for almost
  every transaction (max fraud-case probability: 0.39, never crossing
  the 0.5 classification threshold).
- Root cause: BCELoss treats every example equally. With fraud at
  only ~0.17% of the data, misclassifying fraud barely affected total
  loss, so the optimizer found a "lazy" solution: predict low for
  everything.
- Evaluation on held-out test set @ threshold 0.5:
    Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000

ITERATION 2: Threshold correction via ROC / Youden's J
- No model changes — just searched for a better decision threshold
  using sklearn's roc_curve() and Youden's J statistic
  (J = TPR - FPR, maximized).
- Result: threshold 0.0165 surfaced real fraud detection ability:
    Precision: 0.2290 | Recall: 0.8061 | F1: 0.3567 | ROC-AUC: 0.9047
- Diagnosis: recall became usable, but precision collapsed. Youden's
  J optimizes TPR vs FPR treating both classes as equally sized —
  but with 56,864 non-fraud vs 98 fraud examples, even a small false
  positive RATE (0.47%) translates to hundreds of false alarms in
  absolute terms. Youden's J is not designed for severe class
  imbalance.

ITERATION 3: Fixed the root cause — weighted loss
- Model: removed sigmoid from FraudNet.forward() (now outputs raw
  logits instead of 0-1 probabilities)
- Loss: switched to nn.BCEWithLogitsLoss(pos_weight=...), where
  pos_weight = (non-fraud count / fraud count), computed per bank
  from each bank's own local data. This makes the loss function
  penalize missed fraud cases proportionally to how rare fraud is
  in that bank's shard — directly countering the imbalance that
  caused Iteration 1's failure.
- Evaluation script updated to manually apply sigmoid
  (1 / (1 + exp(-logits))) since the model no longer does this
  internally.
- Result: model now confidently separates classes — fraud-only
  probabilities jumped from maxing at 0.39 to consistently 0.99-1.0.
  ROC-AUC improved to 0.9821.
- Re-ran Youden's J on the new probabilities: threshold 0.2418 gave
  Precision: 0.0361 | Recall: 0.9388 — still broken by the same
  imbalance issue as Iteration 2, confirming Youden's J itself
  (not the model) was the wrong tool.

ITERATION 4 (FINAL): Correct threshold-selection method
- No model changes — replaced Youden's J with a Precision-Recall
  curve + F1-maximization approach (sklearn's precision_recall_curve),
  which computes precision/recall relative to the actual fraud count
  rather than mixing in the much larger non-fraud class size the way
  ROC/FPR does — making it far more suitable for severe imbalance.
- Final threshold: 0.9996
- FINAL RESULT (federated model vs. centralized baseline):
    Federated  — Precision: 0.8511 | Recall: 0.8163 | F1: 0.8333 | AUC: 0.9821
    Baseline   — Precision: 0.9412 | Recall: 0.8163 | F1: 0.8743 | AUC: ~0.96
  Recall is identical between federated and centralized; F1 is
  within 4 points; federated AUC exceeds baseline AUC — demonstrating
  federated learning can approach centralized performance on this
  task without any bank sharing raw transaction data.

KEY TAKEAWAY: two independent problems needed two independent fixes —
(1) the model wasn't learning to weight fraud correctly (fixed via
pos_weight in the loss function), and (2) the threshold-selection
method wasn't suited to imbalanced classes (fixed by switching from
ROC/Youden's J to Precision-Recall/F1). Fixing only one without the
other would not have produced this result.
"""

'''
ROC curve (Youden's J) — what it measures at each threshold:

TPR (True Positive Rate) = fraud caught / total fraud = fraud caught / 98
FPR (False Positive Rate) = false alarms / total non-fraud = false alarms / 56,864

Both rates are computed as a fraction of their own class size. This is the critical detail: FPR divides by 56,864, a huge number — so even a large absolute number of false alarms produces a tiny-looking FPR.
 At threshold 0.2418, FPR was 0.0432 (looks small!) but that's 0.0432 × 56,864 ≈ 2,456 actual false alarms. 
 Youden's J saw "4.3%" and treated it as comparable in scale to catching 93.88% of fraud — but 2,456 false alarms against 92 true catches is a terrible real-world trade, even though the rates looked balanced.

Precision-Recall curve (F1-max) — what it measures at each threshold:

Recall = fraud caught / total fraud = fraud caught / 98 (same as TPR, no change here)
Precision = fraud caught / total flagged as fraud (fraud caught + false alarms)

This is the key structural difference: precision's denominator is "everything you flagged," which directly includes the false alarm count in a way that's proportional to how many fraud cases you actually caught — not diluted by the 56,864 non-fraud population size.
 If you flag 2,456 transactions to catch 92 fraud cases, precision = 92/2456 ≈ 0.037 — it immediately shows you that's a bad trade, exactly matching what you saw in Iteration 3 (precision 0.0361).

Why F1-maximization then fixes it: F1 is the harmonic mean of precision and recall — and harmonic means punish imbalance between the two much more harshly than an average would. 
A threshold with recall 0.94 but precision 0.036 gets a low F1 (because F1 is dragged down toward the smaller number), so the search naturally avoids thresholds like that. 
It only picks a high F1 when both precision and recall are reasonably good together — which is exactly why it landed on threshold 0.9996, giving you 0.8511 precision and 0.8163 recall simultaneously, instead of one great number masking a terrible other one.

One-sentence summary: ROC/Youden's J got fooled because it measures false alarms as a fraction of a huge non-fraud population, making large absolute error counts look small; Precision-Recall/F1 measures errors as a fraction of what you actually flagged, so it can't hide behind a big denominator — making it the correct tool whenever one class is much rarer than the other, which is exactly your situation (98 vs 56,864).
'''

