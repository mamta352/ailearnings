---
title: "Feature Engineering: Fix Bad Data Before It Ruins Models (2026)"
description: "Bad features = bad models, every time. Learn encoding, normalization, interaction features, and datetime extraction."
date: "2026-03-10"
updatedAt: "2026-03-28"
slug: "feature-engineering-guide"
keywords: ["feature engineering", "feature engineering machine learning", "ML feature engineering"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
level: "intermediate"
time: "13 min"
stack: ["Python", "scikit-learn", "pandas"]
---

## Learning Objectives

- Handle missing values, categorical variables, and outliers
- Apply the right encoding strategy for different categorical types
- Create new features that improve model performance
- Scale and normalize features correctly
- Select the most informative features and remove noise

---

## Why Feature Engineering Matters

In practice, feature engineering often contributes more to model performance than algorithm selection. A simple model on great features beats a complex model on poor features.

The feature engineering pipeline is: **raw data → cleaned data → encoded data → scaled data → selected features → model input**.

---

## Handling Missing Values

### Strategy 1: Drop
Drop rows or columns with missing values. Only safe if missingness is rare and random.

```python
df.dropna(subset=['important_column'], inplace=True)  # drop rows
df.drop(columns=['mostly_null_column'], inplace=True)  # drop columns
```

### Strategy 2: Impute with Statistics

```python
from sklearn.impute import SimpleImputer
import pandas as pd

# Numerical: fill with median (robust to outliers)
num_imputer = SimpleImputer(strategy='median')
df[['age', 'income']] = num_imputer.fit_transform(df[['age', 'income']])

# Categorical: fill with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['city']] = cat_imputer.fit_transform(df[['city']])
```

### Strategy 3: Add a Missingness Indicator
Missing data itself can be informative. Add a binary flag before imputing.

```python
df['age_was_missing'] = df['age'].isna().astype(int)
df['age'].fillna(df['age'].median(), inplace=True)
```

### Strategy 4: Model-Based Imputation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iter_imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(iter_imputer.fit_transform(df), columns=df.columns)
```

---

## Encoding Categorical Variables

### Label Encoding
Assigns each category an integer. Only appropriate for ordinal categories where order matters.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
# "high school" → 0, "bachelor" → 1, "master" → 2
```

**Warning:** Don't use label encoding for nominal categories with tree-ensemble models — it implies a false ordering.

### One-Hot Encoding
Creates a binary column for each category. Use for nominal categories with low cardinality.

```python
df_encoded = pd.get_dummies(df, columns=['city', 'department'], drop_first=True)

# Or with scikit-learn:
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded = ohe.fit_transform(df[['city']])
```

### Ordinal Encoding
For ordered categories (low/medium/high, bronze/silver/gold).

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['risk_encoded'] = oe.fit_transform(df[['risk_level']])
```

### Target Encoding (Mean Encoding)
Replace each category with the mean of the target variable for that category. Powerful for high-cardinality features but prone to overfitting — use with cross-validation.

```python
# Manual implementation
target_mean = df.groupby('city')['churn'].mean()
df['city_target_encoded'] = df['city'].map(target_mean)
```

### Frequency Encoding
Replace each category with how often it appears in the dataset.

```python
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)
```

---

## Scaling Numerical Features

### StandardScaler (Z-score normalization)
Centers to mean=0, std=1. Best for normally distributed features and algorithms sensitive to scale (linear models, SVM, neural networks).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use fit from training set only
```

### MinMaxScaler
Scales to a fixed range [0, 1]. Preserves zero values. Sensitive to outliers.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```

### RobustScaler
Uses median and IQR instead of mean and std. Best when your data has many outliers.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
```

**Tree-based models (Random Forest, XGBoost) don't require scaling.** Neural networks, linear models, and SVMs do.

---

## Handling Outliers

### Detect Outliers

```python
# IQR method
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['income'] < lower) | (df['income'] > upper)]
print(f"Outliers: {len(outliers)}")
```

### Treatment Options
- **Cap (Winsorize):** Clip values to [lower, upper]
- **Log transform:** Compress skewed distributions
- **Remove:** Only if outliers are data entry errors

```python
# Winsorize
df['income'] = df['income'].clip(lower=lower, upper=upper)

# Log transform (for right-skewed data)
import numpy as np
df['income_log'] = np.log1p(df['income'])  # log1p handles zeros
```

---

## Creating New Features

### Date/Time Features

```python
df['date'] = pd.to_datetime(df['date'])
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
df['hour']        = df['date'].dt.hour
```

### Interaction Features

```python
df['income_per_dependent'] = df['income'] / (df['dependents'] + 1)
df['age_times_income']     = df['age'] * df['income']
```

### Binning / Discretization

```python
# Manual bins
df['age_group'] = pd.cut(df['age'],
    bins=[0, 18, 35, 50, 65, 100],
    labels=['teen', 'young_adult', 'adult', 'middle_age', 'senior'])

# Quantile bins (equal-frequency)
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4'])
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X[['age', 'income']])
```

---

## Feature Selection

### Filter Methods — Statistical Tests

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Select top 10 features by ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(selected_features.tolist())
```

### Wrapper Method — Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=10)
rfe.fit(X_train, y_train)
print(X.columns[rfe.support_].tolist())
```

### Embedded Method — Feature Importance

```python
import pandas as pd
model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.nlargest(15)
print(top_features)
```

### Permutation Importance (Most Reliable)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
print(perm_imp.head(15))
```

---

## Building a Feature Pipeline

Combine all preprocessing steps into a reproducible pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

numeric_features = ['age', 'income', 'tenure']
categorical_features = ['city', 'plan_type']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier()),
])

full_pipeline.fit(X_train, y_train)
score = full_pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")
```

---

## Troubleshooting

**Model performance doesn't improve after adding features**
→ Check correlation with target. Low correlation = low predictive power. Use `SelectKBest` to filter.

**One-hot encoding creates too many columns**
→ Use target encoding or frequency encoding for high-cardinality categoricals (>20 unique values).

**Train/test performance gap is large despite regularization**
→ Check for data leakage — ensure no future information sneaks into features.

---

## Key Takeaways

- Feature quality determines model ceiling — a perfect algorithm on bad features underperforms a simple algorithm on good features; fix data before tuning hyperparameters
- Fit all preprocessing (scalers, imputers, encoders) on training data only — applying fit_transform on the full dataset leaks test set statistics into training, inflating performance estimates
- Use `sklearn.pipeline.Pipeline` to chain preprocessing and modeling — it prevents data leakage automatically and makes deployment reproducible
- One-hot encoding is correct for nominal categories with no ordering; ordinal encoding is correct for ordered categories (low/medium/high); high-cardinality categories (>20 values) need target encoding or frequency encoding
- Log-transform right-skewed features (income, price, count) before training linear models or neural networks — they are sensitive to scale and skew in ways tree-based models are not
- Interaction features (multiplying or concatenating two features) can capture non-linear relationships that individual features miss — but validate each with cross-validation before keeping
- Datetime features should be decomposed into components (hour, day of week, month, is_weekend) — the raw timestamp integer is meaningless to most models
- Remove features with more than 50% missing values before imputing — imputing highly missing features introduces more noise than signal

## FAQ

**When should I do feature selection?**
After basic preprocessing. Use feature importance from a quick baseline model to identify candidates for removal. Always validate that removing a feature does not hurt performance before dropping it permanently.

**Does feature engineering matter for deep learning?**
Less so for raw data like images and text — deep learning learns features automatically from the raw signal. For tabular data, yes. Good feature engineering still matters even with deep learning; neural networks on tabular data often benefit from the same encoding and normalization steps used for classical models.

**What is the difference between StandardScaler and MinMaxScaler?**
StandardScaler transforms to zero mean and unit variance (z-score normalization). MinMaxScaler scales to a fixed range, typically [0, 1]. Use StandardScaler for most ML models — it is robust when the distribution has outliers. Use MinMaxScaler when your model requires inputs in a specific range, such as a neural network with sigmoid activations.

**How do I handle missing values in test data when I fitted an imputer on training data?**
Call `imputer.transform(X_test)` — not `imputer.fit_transform(X_test)`. The imputer was already fit on training statistics. Calling fit_transform on test data re-estimates statistics from test data, which constitutes leakage. Using a Pipeline ensures this is handled correctly automatically.

**What is target encoding and when should I use it?**
Target encoding replaces a categorical value with the mean of the target variable for that category. For example, city "New York" gets encoded as the mean churn rate of New York customers. Use it for high-cardinality categoricals where one-hot encoding would create hundreds of columns. Always compute target encoding means on training data only, and apply a smoothing factor to prevent overfitting to small categories.

**How many features is too many?**
There is no universal threshold, but more features requires more training data to avoid overfitting. A rough heuristic: if you have fewer than 10 samples per feature, you likely have too many. Use variance inflation factor (VIF) to detect multicollinearity and SelectKBest or recursive feature elimination to reduce dimensionality.

**Should I always normalize features?**
Not always. Tree-based models (decision trees, random forests, gradient boosting) are scale-invariant — they split on feature values, not magnitudes. Linear models, SVMs, and neural networks are scale-sensitive and require normalization. Always check which algorithm you are using before adding a scaling step.

---

## What to Learn Next

- [Python for Machine Learning: NumPy, Pandas, and scikit-learn](/blog/python-for-machine-learning/)
- [Model Evaluation and Metrics: F1, AUC-ROC, and RMSE](/blog/model-evaluation-and-metrics/)
- [Supervised Learning Guide: Algorithms and Best Practices](/blog/supervised-learning-guide/)
- [Machine Learning Roadmap: 6-Month Learning Path](/blog/machine-learning-roadmap/)
- [How LLMs Work: Transformers and Training at Scale](/blog/how-llms-work/)
