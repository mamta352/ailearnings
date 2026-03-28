---
title: "Python for ML: The Stack Engineers Actually Use (2026)"
description: "Most Python ML tutorials teach wrong priorities. Focus on NumPy, pandas, scikit-learn, and matplotlib — the stack that gets you building models today."
date: "2026-03-10"
updatedAt: "2026-03-28"
slug: "python-for-machine-learning"
keywords: ["python machine learning", "numpy pandas scikit-learn", "python for ML"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
level: "beginner"
time: "11 min"
stack: ["Python", "NumPy", "pandas", "scikit-learn"]
---

## Learning Objectives

- Use NumPy for fast numerical computation
- Manipulate and analyze data with pandas
- Visualize distributions and relationships with matplotlib/seaborn
- Apply scikit-learn's consistent API for training and evaluation
- Write clean, reproducible ML code

---

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

---

## NumPy: Fast Numerical Computation

NumPy is the foundation of scientific Python. Everything in ML ultimately runs on NumPy arrays.

### Creating Arrays

```python
import numpy as np

# From lists
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])  # 2D

# Special arrays
zeros = np.zeros((3, 4))      # 3×4 matrix of zeros
ones  = np.ones((2, 3))       # 2×3 matrix of ones
eye   = np.eye(4)              # 4×4 identity matrix
rand  = np.random.randn(3, 3)  # random normal

print(a.shape, b.shape, a.dtype)
```

### Array Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise (no loops needed)
print(a + b)      # [11 22 33 44]
print(a * b)      # [10 40 90 160]
print(a ** 2)     # [1 4 9 16]
print(np.sqrt(a)) # [1.   1.41 1.73 2.  ]

# Matrix multiplication
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
C = A @ B   # (3, 2) result — use @ not np.dot for readability
```

### Slicing and Indexing

```python
X = np.random.randn(100, 5)

X[0]       # first row
X[:, 2]    # third column (all rows)
X[10:20]   # rows 10-19
X[X > 0]   # all positive values (boolean indexing)

# Fancy indexing
indices = np.array([0, 5, 10])
X[indices]  # rows 0, 5, 10
```

### Broadcasting

```python
# Subtract column means from each column
X = np.random.randn(100, 5)
col_means = X.mean(axis=0)  # shape (5,)
X_centered = X - col_means  # broadcasts: (100,5) - (5,) → (100,5)
```

### Statistics

```python
print(X.mean(axis=0))   # mean of each column
print(X.std(axis=0))    # std of each column
print(X.min(), X.max())
print(np.percentile(X, [25, 50, 75]))
```

---

## Pandas: Data Manipulation

Pandas DataFrames are how you'll work with tabular data before feeding it to ML models.

### Loading Data

```python
import pandas as pd

df = pd.read_csv('data.csv')
df = pd.read_json('data.json')
df = pd.read_excel('data.xlsx')

# Quick overview
print(df.shape)        # (rows, cols)
print(df.head())       # first 5 rows
print(df.info())       # column types and null counts
print(df.describe())   # statistics for numeric columns
```

### Selection and Filtering

```python
# Select columns
df['age']                        # single column → Series
df[['age', 'income', 'churn']]  # multiple columns → DataFrame

# Filter rows
df[df['age'] > 30]
df[(df['age'] > 30) & (df['churn'] == 1)]
df.query('age > 30 and churn == 1')  # cleaner syntax

# .loc (label-based) and .iloc (integer-based)
df.loc[0:5, 'age':'income']   # rows 0-5, columns age through income
df.iloc[0:5, 2:6]             # rows 0-4, columns 2-5
```

### Common Operations

```python
# Sorting
df.sort_values('income', ascending=False)

# New columns
df['income_per_age'] = df['income'] / df['age']

# Apply function
df['name_upper'] = df['name'].apply(lambda x: x.upper())
df['age_group'] = df['age'].apply(lambda x: 'senior' if x > 60 else 'adult')

# String operations
df['email_domain'] = df['email'].str.split('@').str[1]
df[df['name'].str.contains('Smith', case=False)]
```

### GroupBy and Aggregation

```python
# Average income by city
df.groupby('city')['income'].mean()

# Multiple aggregations
df.groupby('city').agg({
    'income': ['mean', 'median', 'std'],
    'age':    ['mean', 'min', 'max'],
    'churn':  'sum'
}).round(2)

# Pivot table
pd.pivot_table(df, values='income', index='city', columns='age_group', aggfunc='mean')
```

### Handling Missing Data

```python
print(df.isnull().sum())              # null count per column
print(df.isnull().sum() / len(df))    # null percentage

df.dropna(subset=['income'])          # drop rows where income is null
df['age'].fillna(df['age'].median(), inplace=True)  # fill with median
```

### Merging DataFrames

```python
users    = pd.read_csv('users.csv')
orders   = pd.read_csv('orders.csv')

merged = pd.merge(users, orders, on='user_id', how='left')
```

---

## Data Visualization

### Matplotlib Basics

```python
import matplotlib.pyplot as plt

# Histogram
plt.figure(figsize=(8, 4))
plt.hist(df['income'], bins=30, edgecolor='black')
plt.xlabel('Income')
plt.ylabel('Count')
plt.title('Income Distribution')
plt.tight_layout()
plt.show()

# Scatter plot
plt.scatter(df['age'], df['income'], alpha=0.3, c=df['churn'], cmap='coolwarm')
plt.colorbar(label='Churn')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
```

### Seaborn for Statistical Plots

```python
import seaborn as sns

# Distribution + box plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['income'], kde=True, ax=axes[0])
sns.boxplot(x='churn', y='income', data=df, ax=axes[1])
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.show()

# Pairplot (shows all pairwise relationships)
sns.pairplot(df[['age', 'income', 'tenure', 'churn']], hue='churn')
plt.show()
```

---

## Scikit-learn Core API

Scikit-learn has a consistent API: every estimator has `fit()`, `predict()`, and `score()`.

### The Estimator Pattern

```python
from sklearn.ensemble import RandomForestClassifier

# 1. Instantiate with hyperparameters
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# 2. Fit on training data
model.fit(X_train, y_train)

# 3. Predict on new data
y_pred  = model.predict(X_test)        # class labels
y_proba = model.predict_proba(X_test)  # class probabilities

# 4. Score
accuracy = model.score(X_test, y_test)
```

### Transformers Follow the Same Pattern

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)              # learn mean and std from training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # use training statistics

# Shortcut: fit_transform on training data
X_train_scaled = scaler.fit_transform(X_train)
```

### Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression()),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
# The pipeline applies scaler.transform automatically at prediction time
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Reproducibility Checklist

```python
import numpy as np
import random

# Fix random seeds
np.random.seed(42)
random.seed(42)

# Always set random_state in scikit-learn estimators
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Save and load models
import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')

# Save DataFrames
df.to_csv('processed_data.csv', index=False)
df.to_parquet('processed_data.parquet')  # faster for large files
```

---

## Troubleshooting

**Shape mismatch error**
```python
print(X_train.shape, y_train.shape)  # check dimensions
# Sklearn expects X shape (n_samples, n_features)
# Ensure y is 1D: y.reshape(-1) if needed
```

**ValueError: Input contains NaN**
→ Check `np.isnan(X_train).sum()` and impute missing values before training.

**ConvergenceWarning in LogisticRegression**
→ Increase `max_iter=1000` or standardize features.

---

## Key Takeaways

- NumPy's array operations are 10–1000x faster than equivalent Python loops because they are implemented in C and operate on contiguous memory — always vectorize instead of iterating over array elements
- Pandas DataFrames are the standard input format for scikit-learn pipelines — learn groupby, merge, pivot, and fillna early; these operations cover 90% of real data wrangling work
- scikit-learn's fit/transform/predict API is consistent across all estimators — learn it once and you can use any algorithm, preprocessing step, or pipeline without relearning the interface
- Always separate exploratory notebooks from production code — use Jupyter for EDA and visualization, then migrate working logic to .py files for reproducibility and version control
- Setting random seeds (`numpy.random.seed`, `random_state=` in sklearn) is not optional for reproducible results — experiments that cannot be reproduced are impossible to debug
- Use joblib for saving models, not pickle directly — it handles numpy arrays more efficiently and is the standard for scikit-learn model persistence
- Matplotlib creates publication-quality plots but seaborn produces better-looking statistical visualizations with fewer lines — use seaborn for correlation matrices, distribution plots, and categorical comparisons
- Virtual environments (venv or conda) are mandatory for ML projects — ML dependency conflicts are common and a clean environment per project prevents hours of debugging

## FAQ

**Python 2 or Python 3?**
Python 3 only. Python 2 reached end-of-life in 2020 and is unsupported by all major ML libraries. Use Python 3.10 or 3.11 for the best compatibility with current ML libraries.

**Should I use Jupyter notebooks or Python scripts?**
Both, for different purposes. Notebooks are ideal for exploration, EDA, visualization, and sharing results — the cell-by-cell execution model matches the exploratory workflow. Scripts are required for reusable pipeline code, scheduled jobs, and anything that goes into production. A common pattern: prototype in a notebook, then refactor the working logic into clean .py modules.

**What about GPU acceleration?**
For classical ML (sklearn, XGBoost), CPU is fine. For deep learning, use PyTorch with CUDA. For large-scale tabular data on GPU, consider RAPIDS cuML — it provides a scikit-learn compatible API that runs on NVIDIA GPUs, with 10–100x speedups for common algorithms.

**What is the difference between fit_transform and fit + transform separately?**
fit_transform is equivalent to calling fit and then transform on the same data, combined for convenience. Use fit_transform on training data. Use only transform (not fit_transform) on validation and test data — you want to apply the same transformation parameters learned from training, not refit on the new data. Fitting on test data is data leakage.

**When should I use Pandas vs NumPy directly?**
Use Pandas for labeled, heterogeneous data — DataFrames handle mixed types, named columns, and missing values well. Use NumPy for homogeneous numerical arrays where performance matters — matrix operations, custom distance computations, and feeding data into PyTorch or TensorFlow. Most ML workflows start in Pandas and convert to NumPy arrays before model training.

**What is the best way to handle missing data in Python?**
Use sklearn.impute.SimpleImputer inside a Pipeline. For numeric columns, median imputation is more robust than mean imputation when outliers are present. For categorical columns, use most_frequent or a constant placeholder. For time-series data, forward-fill or backward-fill is often more appropriate. Always impute after splitting data — fit the imputer on training data only.

**How do I profile which part of my ML code is slow?**
Use line_profiler or cProfile for Python code. The most common bottlenecks are: applying Python functions row-by-row in Pandas (use vectorized operations instead), loading data in a loop (batch-load with pd.read_csv once), and repeated model re-fitting (cache fitted models with joblib). For data loading, switching from CSV to Parquet typically provides 5–10x read speed improvement.

---

## What to Learn Next

- [Machine Learning Basics for Developers: Build Your First Model](/blog/machine-learning-basics-for-developers/)
- [Supervised Learning Guide: scikit-learn Algorithms and Workflows](/blog/supervised-learning-guide/)
- [Feature Engineering Guide: Encoding, Normalization, and Selection](/blog/feature-engineering-guide/)
- [Model Evaluation and Metrics: F1, AUC-ROC, and RMSE](/blog/model-evaluation-and-metrics/)
- [Transformer Architecture Explained: From ML to Modern LLMs](/blog/transformer-architecture-explained/)
