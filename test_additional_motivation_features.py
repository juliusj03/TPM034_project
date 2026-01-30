#!/usr/bin/env python3
"""
Test if additional motivation features improve model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load data
script_dir = Path(__file__).parent
data_path = script_dir / 'Data files' / 'fulldataset.xlsx'
df = pd.read_excel(data_path)

# Identify columns
ct_cols = [c for c in df.columns if c.startswith("ct1_") or c.startswith("ct2_")]
mct_cols = [c for c in df.columns if c.startswith("motct")]

# Ensure numeric
for c in ct_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Create target
def add_deviation_scores(group):
    med = group[ct_cols].median(numeric_only=True)
    deviation = group[ct_cols].sub(med, axis=1)
    abs_deviation = deviation.abs()
    mad = abs_deviation.mean(axis=1, skipna=True)
    group = group.copy()
    group["ct_mad_from_median"] = mad
    return group

df = df.groupby("study", group_keys=False).apply(add_deviation_scores)
study_median_mad = df.groupby("study")["ct_mad_from_median"].transform("median")
df["silent_middle"] = (df["ct_mad_from_median"] <= study_median_mad).astype(int)

# Original features
df["motivationamount"] = df[mct_cols].notna().sum(axis=1)
df["motivation_length"] = (
    df[mct_cols]
    .fillna("")
    .astype(str)
    .apply(lambda row: sum(len(s.split()) for s in row), axis=1)
)
df["logmotivationlength"] = np.log1p(df["motivation_length"])

print("="*70)
print("TESTING ADDITIONAL MOTIVATION FEATURES")
print("="*70)

# New features
print("\nCreating new motivation features...")

# 1. Response rate (% of answered questions with motivation)
tasks_answered = df[ct_cols].notna().sum(axis=1)
df['motivation_response_rate'] = df['motivationamount'] / tasks_answered.replace(0, np.nan)

# 2. Average words per motivation (when provided)
df['avg_words_per_motivation'] = df['motivation_length'] / df['motivationamount'].replace(0, np.nan)

# 3. Binary: provided any motivation
df['provided_any_motivation'] = (df['motivationamount'] > 0).astype(int)

# 4. Motivation variance (how consistent is length across motivations?)
def calc_motivation_variance(row):
    lengths = []
    for col in mct_cols:
        val = row[col]
        if pd.notna(val) and val not in ['', '""']:
            lengths.append(len(str(val).split()))
    if len(lengths) > 1:
        return np.std(lengths)
    return 0

print("  Computing motivation variance...")
df['motivation_variance'] = df[mct_cols].apply(calc_motivation_variance, axis=1)

# 5. Max motivation length (longest single response)
def calc_max_motivation(row):
    lengths = []
    for col in mct_cols:
        val = row[col]
        if pd.notna(val) and val not in ['', '""']:
            lengths.append(len(str(val).split()))
    return max(lengths) if lengths else 0

print("  Computing max motivation length...")
df['max_motivation_length'] = df[mct_cols].apply(calc_max_motivation, axis=1)

print("\nNew features created:")
print("  - motivation_response_rate")
print("  - avg_words_per_motivation")
print("  - provided_any_motivation")
print("  - motivation_variance")
print("  - max_motivation_length")

# Setup for modeling
y = df["silent_middle"].copy()
groups = df["study"].copy()

exclude = set(ct_cols + ["ct_mad_from_median", "silent_middle"] + mct_cols +
              ["study", "ct1type", "ct2type", "id", "time", "motivation_length"])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def create_pipeline(X_data):
    cat_cols = X_data.select_dtypes(include=["object", "string", "category"]).columns
    num_cols = X_data.columns.difference(cat_cols)

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=4,
        min_samples_leaf=80,
        min_samples_split=30,
        max_features=0.2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("prep", preprocess), ("rf", rf)])

# Test different feature combinations
feature_sets = {
    "Original (2 features)": ["motivationamount", "logmotivationlength"],

    "+ response_rate": ["motivationamount", "logmotivationlength", "motivation_response_rate"],

    "+ avg_words": ["motivationamount", "logmotivationlength", "avg_words_per_motivation"],

    "+ binary": ["motivationamount", "logmotivationlength", "provided_any_motivation"],

    "+ variance": ["motivationamount", "logmotivationlength", "motivation_variance"],

    "+ max_length": ["motivationamount", "logmotivationlength", "max_motivation_length"],

    "All new features": [
        "motivationamount", "logmotivationlength",
        "motivation_response_rate", "avg_words_per_motivation",
        "provided_any_motivation", "motivation_variance", "max_motivation_length"
    ],

    "No motivation": []
}

results = []

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

for name, motiv_features in feature_sets.items():
    print(f"\n{name}...")

    # Build feature set
    base_features = [c for c in df.columns if c not in exclude]

    if name == "No motivation":
        # Remove all motivation-related features
        X = df[base_features].copy()
        X = X.drop(columns=[c for c in X.columns if 'motivation' in c.lower() or 'motiv' in c.lower()], errors='ignore')
    else:
        # Add specified motivation features
        X = df[base_features + motiv_features].copy()

    # Remove duplicates
    X = X.loc[:, ~X.columns.duplicated()]

    print(f"  Features: {X.shape[1]} ({len([c for c in X.columns if 'motiv' in c.lower()])} motivation-related)")

    # Train and evaluate
    model = create_pipeline(X)

    try:
        scores = cross_val_score(
            model, X, y, cv=cv, groups=groups,
            scoring='roc_auc', n_jobs=-1
        )
        auc_mean = scores.mean()
        auc_std = scores.std()

        print(f"  ROC AUC: {auc_mean:.4f} ± {auc_std:.4f}")

        results.append({
            'Feature Set': name,
            'N Features': X.shape[1],
            'N Motivation': len([c for c in X.columns if 'motiv' in c.lower()]),
            'ROC AUC': auc_mean,
            'AUC Std': auc_std
        })
    except Exception as e:
        print(f"  Failed: {e}")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)

# Calculate improvement over baseline
baseline_auc = results_df[results_df['Feature Set'] == 'Original (2 features)']['ROC AUC'].values[0]
no_motiv_auc = results_df[results_df['Feature Set'] == 'No motivation']['ROC AUC'].values[0]

results_df['Δ vs Original'] = results_df['ROC AUC'] - baseline_auc
results_df['Δ vs No Motiv'] = results_df['ROC AUC'] - no_motiv_auc

print("\nRanked by ROC AUC:")
print(results_df.sort_values('ROC AUC', ascending=False).to_string(index=False))

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

best_idx = results_df['ROC AUC'].idxmax()
best_set = results_df.loc[best_idx]

improvement_vs_original = best_set['ROC AUC'] - baseline_auc
improvement_vs_no_motiv = best_set['ROC AUC'] - no_motiv_auc

print(f"""
Best feature set: {best_set['Feature Set']}
  ROC AUC: {best_set['ROC AUC']:.4f}
  Improvement vs original: {improvement_vs_original:+.4f} ({improvement_vs_original/baseline_auc*100:+.2f}%)
  Improvement vs no motivation: {improvement_vs_no_motiv:+.4f}

Original (2 features): {baseline_auc:.4f}
No motivation: {no_motiv_auc:.4f}

{'⚠️  Adding new features HELPS' if improvement_vs_original > 0.005 else '✓ Original features are sufficient'}
{'   Consider adding: ' + best_set['Feature Set'] if improvement_vs_original > 0.005 else '   Stick with original 2 features'}
""")

print("\nKey insights:")
if improvement_vs_original > 0.005:
    print("  • New motivation features improve prediction")
    print("  • Worth adding to model for better performance")
else:
    print("  • New features provide minimal improvement (< 0.005 AUC)")
    print("  • Original features (motivationamount + logmotivationlength) capture most signal")
    print("  • Adding more features increases complexity without meaningful gain")
