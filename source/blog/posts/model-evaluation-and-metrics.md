---
title: "ML Metrics: Pick the Wrong One and Ship a Broken Model (2026)"
description: "Accuracy looks great, model fails in production. Learn when F1, ROC-AUC, precision, and recall actually matter — with sklearn code and real scenarios."
date: "2026-03-10"
updatedAt: "2026-03-28"
slug: "model-evaluation-and-metrics"
keywords: ["model evaluation", "ML metrics", "precision recall F1", "AUC ROC", "model performance"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
level: "intermediate"
time: "11 min"
stack: ["Python", "scikit-learn"]
---

## Learning Objectives

- Understand why accuracy alone is misleading
- Choose the right metric for regression and classification tasks
- Read and interpret a confusion matrix
- Use AUC-ROC curves to compare classifiers
- Diagnose overfitting and underfitting with learning curves
- Apply cross-validation correctly

---

## Why Metrics Matter More Than You Think

Choosing the wrong metric can make a terrible model look great. Consider a fraud detection model where 99% of transactions are legitimate. A model that always predicts "not fraud" achieves **99% accuracy** but catches zero fraud cases. Accuracy is useless here.

Good evaluation starts with understanding your problem: **What kind of errors are most costly?**

---

## Classification Metrics

### The Confusion Matrix

```
                Predicted Positive    Predicted Negative
Actual Positive      TP (True Pos)        FN (False Neg)
Actual Negative      FP (False Pos)       TN (True Neg)
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fraud', 'Fraud'])
disp.plot(cmap='Blues')
plt.show()
```

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**Use only when:** classes are balanced (roughly equal positive/negative counts).

### Precision
```
Precision = TP / (TP + FP)
```
"Of all the predictions that said positive, how many were actually positive?"

**Use when:** false positives are costly. Example: spam filter — you don't want to block legitimate emails.

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
"Of all actual positives, how many did the model find?"

**Use when:** false negatives are costly. Example: cancer screening — you don't want to miss real cases.

### F1 Score
```
F1 = 2 × Precision × Recall / (Precision + Recall)
```
Harmonic mean of precision and recall. Balances both concerns.

**Use when:** you have imbalanced classes and care about both false positives and false negatives.

### F-beta Score
When you want to weight precision vs recall differently:
```
F_beta = (1 + β²) × P × R / (β²×P + R)
```
- `beta > 1` → weights recall higher (catching positives matters more)
- `beta < 1` → weights precision higher (avoiding false alarms matters more)

```python
from sklearn.metrics import fbeta_score
f2 = fbeta_score(y_test, y_pred, beta=2)  # recall-focused
```

### AUC-ROC

The ROC curve plots **True Positive Rate (Recall)** vs **False Positive Rate** at all classification thresholds. AUC (Area Under Curve) summarizes this in a single number.

- AUC = 1.0 → perfect classifier
- AUC = 0.5 → random chance
- AUC = 0.0 → perfectly wrong (but flip predictions and it's perfect)

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**Use AUC-ROC when** you want to compare classifiers independent of the decision threshold.

### Precision-Recall Curve

For heavily imbalanced datasets, the PR curve is more informative than ROC.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

---

## Regression Metrics

### Mean Absolute Error (MAE)
```
MAE = mean(|y - ŷ|)
```
Average absolute prediction error. Same units as target variable. Robust to outliers.

### Mean Squared Error (MSE) / RMSE
```
MSE  = mean((y - ŷ)²)
RMSE = sqrt(MSE)
```
Penalizes large errors more. RMSE is in the same units as target — easier to interpret than MSE.

### R² (Coefficient of Determination)
```
R² = 1 - SS_res / SS_tot
```
- R² = 1.0 → model explains all variance (perfect)
- R² = 0.0 → model just predicts the mean (useless)
- R² < 0 → model is worse than predicting the mean

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.3f}")
```

### Mean Absolute Percentage Error (MAPE)
```
MAPE = mean(|y - ŷ| / |y|) × 100
```
Useful when target values vary by orders of magnitude. Undefined when actual values are 0.

---

## Learning Curves: Diagnosing Overfitting and Underfitting

Learning curves show training and validation performance as a function of training set size.

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training F1')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation F1')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend()
plt.show()
```

**Reading the curve:**
- **High bias (underfitting):** Both curves are low and converging at a low value. → Use a more complex model or add features.
- **High variance (overfitting):** Training score is high, validation score is much lower with a large gap. → Get more data, add regularization, or reduce model complexity.
- **Good fit:** Both curves converge at a high value.

---

## Cross-Validation Strategies

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_validate

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(model, X, y, cv=kf,
                         scoring=['accuracy', 'f1', 'roc_auc'],
                         return_train_score=True)

for metric in ['accuracy', 'f1', 'roc_auc']:
    val = results[f'test_{metric}']
    print(f"{metric}: {val.mean():.3f} ± {val.std():.3f}")
```

### Stratified K-Fold (Classification)
Ensures each fold has the same class proportions. **Always use this for classification.**

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Time Series Split
For temporal data — always train on the past, validate on the future.

```python
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=5)
```

---

## Metric Selection Guide

| Problem | Balanced Classes | Imbalanced Classes |
|---------|-----------------|-------------------|
| Binary Classification | Accuracy, AUC-ROC | F1, AUC-ROC, PR-AUC |
| Multi-class Classification | Macro F1 | Weighted F1 |
| Regression | RMSE, R² | MAE (robust to outliers) |
| Ranking/Probability | AUC-ROC | PR-AUC |

---

## Troubleshooting

**Model shows 99% accuracy but isn't useful**
→ Check class balance. Switch to F1 or AUC-ROC.

**Validation score jumps around a lot between folds**
→ Small dataset. Try Leave-One-Out CV or more folds. Get more data if possible.

**R² is negative**
→ Your model is worse than predicting the mean. Something is wrong — check for data leakage, target encoding, or train/test contamination.

**High AUC but poor performance at the default 0.5 threshold**
→ Tune the classification threshold for your specific precision/recall requirement using the ROC or PR curve.

---

## Key Takeaways

- Accuracy is only valid as a metric when classes are balanced — on a dataset where 95% of examples belong to one class, a constant-prediction model achieves 95% accuracy while being completely useless
- Precision and recall describe different failure modes: high precision means few false positives; high recall means few false negatives — which matters more depends on the cost of each error type in your application
- F1 score is the harmonic mean of precision and recall — it is zero if either is zero, penalizing models that sacrifice one for the other; use it as the primary metric for imbalanced classification
- AUC-ROC measures classification quality across all possible decision thresholds — it equals the probability that the model ranks a random positive example higher than a random negative example
- Learning curves diagnose overfitting vs underfitting — plot training and validation loss vs training set size; a large gap at full data size indicates overfitting; both curves high and converging indicates underfitting
- RMSE penalizes large errors more than MAE because it squares residuals — use RMSE when large errors are disproportionately costly; use MAE when outliers in the target should not dominate the metric
- Always report metrics on a held-out test set that was never used during development — validation set performance after hyperparameter tuning is optimistically biased
- A negative R² means your model is worse than predicting the mean — this signals data leakage, target encoding issues, or train/test contamination, not a numerical edge case

## FAQ

**What is a good AUC score?**
Context-dependent. For fraud detection: 0.95+ is expected. For some medical screening problems: 0.75 can be clinically meaningful. Always compare to a baseline — a trivial classifier that always predicts the majority class typically achieves AUC of 0.5.

**When should I use macro vs micro vs weighted F1?**
Macro F1 averages F1 per class treating all classes equally — use for class-balanced evaluation where you care equally about performance on rare and common classes. Micro F1 aggregates TP/FP/FN across all classes and is dominated by majority class performance. Weighted F1 weights by class frequency — use for imbalanced multi-class problems where you care more about performance on common classes.

**Should I report metrics on validation or test set?**
Report final metrics on the test set — held out throughout the entire development process. Use the validation set only for model selection and hyperparameter tuning. Reporting validation performance after tuning overstates real-world performance because you have implicitly optimized for it.

**What is the difference between ROC-AUC and PR-AUC?**
ROC-AUC measures performance at all thresholds using true positive rate vs false positive rate. PR-AUC measures precision vs recall at all thresholds. For highly imbalanced datasets (fraud, anomaly detection), PR-AUC is more informative — ROC-AUC can look artificially high because the large number of true negatives inflates TPR. PR-AUC is sensitive to the positive class size and reflects the practical difficulty of finding positives.

**How do I set the right classification threshold?**
Plot the precision-recall curve and pick the threshold that matches your business requirements. If false negatives are costly (missing fraud), choose a threshold that maximizes recall. If false positives are costly (alerting users unnecessarily), choose a threshold that maximizes precision. The default 0.5 threshold is rarely optimal for imbalanced problems.

**What is a confusion matrix and how do I read it?**
A confusion matrix is a 2x2 table for binary classification: true positives (correctly predicted positive), false positives (wrongly predicted positive), true negatives (correctly predicted negative), false negatives (wrongly predicted negative). Read row-by-row: the first row is actual negatives, the second row is actual positives. High off-diagonal values indicate systematic prediction errors worth investigating.

**What metrics should I use for a regression problem?**
RMSE (root mean squared error) for problems where large errors should be penalized more heavily — it squares residuals. MAE (mean absolute error) for problems where all errors should be treated equally and outliers in the target should not dominate. R² for interpretability — it measures what fraction of variance your model explains. Report multiple metrics and choose the primary one based on the business cost of different error magnitudes.

---

## What to Learn Next

- [Feature Engineering Guide: Encoding, Normalization, and Selection](/blog/feature-engineering-guide/)
- [Supervised Learning Guide: Algorithms, Training, and Evaluation](/blog/supervised-learning-guide/)
- [Machine Learning Roadmap: 6-Month Learning Path](/blog/machine-learning-roadmap/)
- [Python for Machine Learning: NumPy, Pandas, and scikit-learn](/blog/python-for-machine-learning/)
- [How LLMs Work: Evaluation at Scale for Language Models](/blog/how-llms-work/)
