#!/usr/bin/env python3
"""
Verify motivation feature calculation and explore alternative measures.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set working directory
script_dir = Path(__file__).parent
data_path = script_dir / 'Data files' / 'fulldataset.xlsx'

# Load data
df = pd.read_excel(data_path)

# Identify motivation columns
mct_cols = [c for c in df.columns if c.startswith("motct")]
print(f"Total motivation columns: {len(mct_cols)}")

# Recreate target (simplified version)
ct_cols = [c for c in df.columns if c.startswith("ct1_") or c.startswith("ct2_")]
for c in ct_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

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

print("\n" + "="*70)
print("DIAGNOSTIC: Checking motivation calculation")
print("="*70)

# Current calculation
df["motivationamount"] = df[mct_cols].notna().sum(axis=1)
df["motivation_length"] = (
    df[mct_cols]
    .fillna("")
    .astype(str)
    .apply(lambda row: sum(len(s.split()) for s in row), axis=1)
)

# Sample inspection
print("\n1. Sample of raw motivation data (first 5 rows):")
print(df[['id', 'motct1_1', 'motct1_2', 'motivationamount', 'motivation_length']].head())

# Check empty string issue
print("\n2. Checking empty string handling:")
test_vals = ['""', "", None, "hello world", "test"]
for val in test_vals:
    words = len(str(val).split()) if pd.notna(val) else 0
    print(f"  Value: {repr(val):20s} → Words: {words}")

# Distribution of motivationamount
print("\n3. Distribution of motivationamount:")
print(df['motivationamount'].describe())

# How many people provide NO motivation?
no_motivation = (df['motivationamount'] == 0).sum()
print(f"\nRespondents with 0 motivations: {no_motivation:,} ({no_motivation/len(df)*100:.1f}%)")

# Average words per motivation (when provided)
df['avg_words_per_motivation'] = df['motivation_length'] / df['motivationamount'].replace(0, np.nan)
print("\n4. Average words per motivation (when provided):")
print(df['avg_words_per_motivation'].describe())

print("\n" + "="*70)
print("ALTERNATIVE MOTIVATION METRICS")
print("="*70)

# Alternative 1: Response rate
ct1_count = df[[c for c in ct_cols if c.startswith('ct1_')]].notna().sum(axis=1)
ct2_count = df[[c for c in ct_cols if c.startswith('ct2_')]].notna().sum(axis=1)
df['tasks_answered'] = ct1_count + ct2_count
df['motivations_provided'] = df['motivationamount']
df['motivation_response_rate'] = df['motivations_provided'] / df['tasks_answered'].replace(0, np.nan)

print("\n1. Response rate (% of tasks with motivation):")
print(df['motivation_response_rate'].describe())

# Alternative 2: Binary indicator
df['provided_any_motivation'] = (df['motivationamount'] > 0).astype(int)

# Alternative 3: Detailed word count (excluding empty strings)
def count_words_carefully(row):
    """Count words only in non-empty motivation fields"""
    total = 0
    for col in mct_cols:
        val = row[col]
        if pd.notna(val) and val not in ['', '""', None]:
            total += len(str(val).split())
    return total

print("\n2. Recalculating word count (excluding empty strings)...")
df['motivation_length_fixed'] = df[mct_cols].apply(count_words_carefully, axis=1)

print("\nComparison of word counts:")
comparison = pd.DataFrame({
    'Original': df['motivation_length'].describe(),
    'Fixed (no empty)': df['motivation_length_fixed'].describe()
})
print(comparison)

print("\n" + "="*70)
print("COMPARISON BY SILENT MIDDLE (with alternative metrics)")
print("="*70)

metrics = {
    'motivationamount': 'Count of motivations',
    'motivation_length': 'Total words (original)',
    'motivation_length_fixed': 'Total words (fixed)',
    'motivation_response_rate': 'Response rate (%)',
    'avg_words_per_motivation': 'Avg words per motivation',
    'provided_any_motivation': 'Provided ANY motivation (binary)'
}

for metric, label in metrics.items():
    print(f"\n{label}:")
    by_group = df.groupby('silent_middle')[metric].agg(['mean', 'median', 'std']).round(2)
    print(by_group)

    # Statistical test
    from scipy.stats import mannwhitneyu
    silent = df[df['silent_middle'] == 1][metric].dropna()
    not_silent = df[df['silent_middle'] == 0][metric].dropna()

    if len(silent) > 0 and len(not_silent) > 0:
        stat, p = mannwhitneyu(silent, not_silent)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  p-value: {p:.4f} {sig}")

        # Effect size (Cohen's d approximation)
        mean_diff = not_silent.mean() - silent.mean()
        pooled_std = np.sqrt((silent.std()**2 + not_silent.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
Based on this analysis, consider using:
1. **motivation_response_rate**: Shows engagement as % of tasks with motivation
2. **avg_words_per_motivation**: Shows depth when people DO engage
3. **provided_any_motivation**: Binary indicator for basic engagement

These may show clearer differences than raw counts.
""")
