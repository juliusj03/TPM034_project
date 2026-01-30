#!/usr/bin/env python3
"""
Test alternative definitions of "silent middle" to see which correlates
better with motivation and other engagement indicators.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
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

# Calculate motivation features
df["motivationamount"] = df[mct_cols].notna().sum(axis=1)
df["motivation_length"] = (
    df[mct_cols]
    .fillna("")
    .astype(str)
    .apply(lambda row: sum(len(s.split()) for s in row), axis=1)
)
df["motivation_response_rate"] = df["motivationamount"] / df[ct_cols].notna().sum(axis=1).replace(0, np.nan)

print("="*70)
print("TESTING ALTERNATIVE SILENT MIDDLE DEFINITIONS")
print("="*70)

# ============================================================================
# Define multiple target definitions
# ============================================================================

def add_deviation_scores(group):
    """Helper for study-relative calculations"""
    med = group[ct_cols].median(numeric_only=True)
    deviation = group[ct_cols].sub(med, axis=1)
    abs_deviation = deviation.abs()
    mad = abs_deviation.mean(axis=1, skipna=True)
    group = group.copy()
    group["ct_mad_from_median"] = mad
    return group

df = df.groupby("study", group_keys=False).apply(add_deviation_scores)

targets = {}

# 1. Current definition (study-relative MAD ≤ median)
study_median_mad = df.groupby("study")["ct_mad_from_median"].transform("median")
targets['Current (study MAD ≤ median)'] = (df["ct_mad_from_median"] <= study_median_mad).astype(int)

# 2. Study-relative MAD ≤ 25th percentile (stricter)
study_q25_mad = df.groupby("study")["ct_mad_from_median"].transform(lambda x: x.quantile(0.25))
targets['Strict (study MAD ≤ Q25)'] = (df["ct_mad_from_median"] <= study_q25_mad).astype(int)

# 3. Study-relative MAD ≤ 75th percentile (broader)
study_q75_mad = df.groupby("study")["ct_mad_from_median"].transform(lambda x: x.quantile(0.75))
targets['Broad (study MAD ≤ Q75)'] = (df["ct_mad_from_median"] <= study_q75_mad).astype(int)

# 4. Low variance (people who give similar scores across all questions)
df['response_std'] = df[ct_cols].std(axis=1, skipna=True)
median_std = df['response_std'].median()
targets['Low variance (std ≤ median)'] = (df['response_std'] <= median_std).astype(int)

# 5. Absolute middle (responses close to 50 on 0-100 scale)
df['mean_response'] = df[ct_cols].mean(axis=1, skipna=True)
targets['Absolute middle (40-60)'] = ((df['mean_response'] >= 40) & (df['mean_response'] <= 60)).astype(int)

# 6. Low engagement (bottom 50% of motivation)
median_motivation = df['motivationamount'].median()
targets['Low engagement (motiv ≤ median)'] = (df['motivationamount'] <= median_motivation).astype(int)

# 7. Very low engagement (bottom 25% of motivation)
q25_motivation = df['motivationamount'].quantile(0.25)
targets['Very low engagement (motiv ≤ Q25)'] = (df['motivationamount'] <= q25_motivation).astype(int)

# 8. Combined: Low MAD AND low motivation
targets['Combined (MAD+motiv low)'] = (
    (df["ct_mad_from_median"] <= study_median_mad) &
    (df['motivationamount'] <= median_motivation)
).astype(int)

print(f"\nCreated {len(targets)} alternative target definitions")

# ============================================================================
# Evaluate each target definition
# ============================================================================

results = []

for target_name, target_values in targets.items():
    print(f"\n{'='*70}")
    print(f"Target: {target_name}")
    print(f"{'='*70}")

    # Prevalence
    prevalence = target_values.mean()
    print(f"Prevalence: {prevalence:.1%}")

    # Correlation with motivation features
    from scipy.stats import pointbiserialr

    corr_amount, p_amount = pointbiserialr(target_values, df['motivationamount'])
    corr_length, p_length = pointbiserialr(target_values, df['motivation_length'])
    corr_rate, p_rate = pointbiserialr(target_values.dropna(), df['motivation_response_rate'].dropna())

    print(f"\nCorrelation with motivation:")
    print(f"  Amount: r={corr_amount:+.3f} (p={p_amount:.4f})")
    print(f"  Length: r={corr_length:+.3f} (p={p_length:.4f})")
    print(f"  Rate:   r={corr_rate:+.3f} (p={p_rate:.4f})")

    # Group comparison (Mann-Whitney U)
    group_0 = df[target_values == 0]
    group_1 = df[target_values == 1]

    stat, p = mannwhitneyu(
        group_1['motivationamount'].dropna(),
        group_0['motivationamount'].dropna()
    )

    mean_0 = group_0['motivationamount'].mean()
    mean_1 = group_1['motivationamount'].mean()
    diff = mean_1 - mean_0
    pct_diff = (diff / mean_0 * 100) if mean_0 > 0 else 0

    # Cohen's d
    pooled_std = np.sqrt(
        (group_0['motivationamount'].std()**2 + group_1['motivationamount'].std()**2) / 2
    )
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    print(f"\nGroup differences (motivationamount):")
    print(f"  Target=0: {mean_0:.2f}")
    print(f"  Target=1: {mean_1:.2f}")
    print(f"  Diff: {diff:+.2f} ({pct_diff:+.1f}%)")
    print(f"  Cohen's d: {cohens_d:+.3f}")
    print(f"  p-value: {p:.4f}")

    # Store results
    results.append({
        'Target': target_name,
        'Prevalence': prevalence,
        'Corr_Amount': corr_amount,
        'Corr_Length': corr_length,
        'Corr_Rate': corr_rate,
        'Mean_Diff': diff,
        'Pct_Diff': pct_diff,
        'Cohens_d': cohens_d,
        'P_value': p
    })

# ============================================================================
# Summary comparison
# ============================================================================

print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Cohens_d', key=abs, ascending=False)

print("\nRanked by effect size (Cohen's d - absolute value):")
print(results_df[['Target', 'Prevalence', 'Cohens_d', 'Pct_Diff', 'Corr_Amount']].to_string(index=False))

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

best_idx = results_df['Cohens_d'].abs().idxmax()
best_target = results_df.loc[best_idx]

print(f"""
Strongest correlation with motivation: {best_target['Target']}
  - Cohen's d: {best_target['Cohens_d']:.3f}
  - Difference: {best_target['Pct_Diff']:+.1f}%
  - Prevalence: {best_target['Prevalence']:.1%}

This definition shows the clearest relationship between "silent middle"
and engagement/motivation metrics.

Current definition ranks #{results_df[results_df['Target'] == 'Current (study MAD ≤ median)'].index[0] + 1}
by effect size.
""")

print("\n" + "="*70)
print("PREDICTIVE PERFORMANCE COMPARISON")
print("="*70)
print("\nTraining models with each target definition...")

# Quick model comparison (using same features as notebook)
exclude = set(ct_cols + ["ct_mad_from_median"] + mct_cols +
              ["study", "ct1type", "ct2type", "id", "time", "motivation_length",
               "response_std", "mean_response", "motivation_response_rate"])

X = df[[c for c in df.columns if c not in exclude]].copy()
groups = df["study"].copy()

cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

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
        n_estimators=100,  # Reduced for speed
        max_depth=4,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("prep", preprocess), ("rf", rf)])

model_results = []

for target_name, target_values in targets.items():
    # Skip targets with too low/high prevalence (can't train properly)
    prevalence = target_values.mean()
    if prevalence < 0.1 or prevalence > 0.9:
        print(f"  Skipping {target_name} (prevalence {prevalence:.1%} too extreme)")
        continue

    print(f"  Training: {target_name}...")
    model = create_pipeline(X)

    try:
        scores = cross_val_score(
            model, X, target_values, cv=cv, groups=groups,
            scoring='roc_auc', n_jobs=-1
        )
        auc = scores.mean()
        print(f"    AUC: {auc:.4f}")

        model_results.append({
            'Target': target_name,
            'ROC_AUC': auc,
            'AUC_std': scores.std()
        })
    except Exception as e:
        print(f"    Failed: {e}")

print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

if model_results:
    model_results_df = pd.DataFrame(model_results).sort_values('ROC_AUC', ascending=False)
    print("\nRanked by ROC AUC:")
    print(model_results_df.to_string(index=False))

    print(f"\nBest predictive performance: {model_results_df.iloc[0]['Target']}")
    print(f"  ROC AUC: {model_results_df.iloc[0]['ROC_AUC']:.4f}")

print("\n" + "="*70)
print("FINAL RECOMMENDATION")
print("="*70)
print("""
Consider these factors when choosing target definition:

1. **Conceptual alignment**: What does "silent middle" mean?
   - Low deviation from median? (current)
   - Low engagement? (motivation-based)
   - Both? (combined)

2. **Predictability**: Which target can be predicted best from demographics/attitudes?

3. **Effect size with motivation**: Which shows clearest engagement differences?

4. **Practical utility**: Which definition helps policymakers most?
""")
