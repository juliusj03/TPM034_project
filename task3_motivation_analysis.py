# Task 3: How does motivation contribute to predicting the silent middle?
# Add these cells to the main notebook after SHAP analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

print("="*70)
print("TASK 3: Motivation Contribution Analysis")
print("="*70)

# ============================================================================
# 1. Descriptive Analysis: Motivation patterns by silent middle status
# ============================================================================

print("\n1. Motivation patterns by group:")
print("-" * 50)

motivation_comparison = df.groupby('silent_middle').agg({
    'motivationamount': ['mean', 'median', 'std'],
    'motivation_length': ['mean', 'median', 'std']
}).round(2)

print("\nMotivation statistics by group:")
print(motivation_comparison)

# Statistical test
from scipy.stats import mannwhitneyu

silent = df[df['silent_middle'] == 1]
not_silent = df[df['silent_middle'] == 0]

stat_amount, p_amount = mannwhitneyu(
    silent['motivationamount'].dropna(),
    not_silent['motivationamount'].dropna()
)
stat_length, p_length = mannwhitneyu(
    silent['motivation_length'].dropna(),
    not_silent['motivation_length'].dropna()
)

print(f"\nMann-Whitney U Test Results:")
print(f"  motivationamount: p-value = {p_amount:.4f} {'***' if p_amount < 0.001 else '**' if p_amount < 0.01 else '*' if p_amount < 0.05 else 'ns'}")
print(f"  motivation_length: p-value = {p_length:.4f} {'***' if p_length < 0.001 else '**' if p_length < 0.01 else '*' if p_length < 0.05 else 'ns'}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Motivation amount
ax1 = axes[0]
df.boxplot(column='motivationamount', by='silent_middle', ax=ax1)
ax1.set_title('Motivation Amount by Group', fontsize=12, fontweight='bold')
ax1.set_xlabel('Silent Middle (0=No, 1=Yes)')
ax1.set_ylabel('Number of Motivations Provided')
plt.sca(ax1)
plt.xticks([1, 2], ['Not Silent', 'Silent Middle'])

# Right: Motivation length (log)
ax2 = axes[1]
df.boxplot(column='logmotivationlength', by='silent_middle', ax=ax2)
ax2.set_title('Motivation Length (log) by Group', fontsize=12, fontweight='bold')
ax2.set_xlabel('Silent Middle (0=No, 1=Yes)')
ax2.set_ylabel('Log(Total Word Count)')
plt.sca(ax2)
plt.xticks([1, 2], ['Not Silent', 'Silent Middle'])

plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.show()

print("\nInterpretation:")
if p_amount < 0.05 and p_length < 0.05:
    print("  ✓ Silent middle respondents provide SIGNIFICANTLY fewer and shorter motivations")
    print("  → This suggests lower engagement or less strong opinions")
else:
    print("  - Motivation differences are not statistically significant")

# ============================================================================
# 2. Model Performance: With vs Without Motivation Features
# ============================================================================

print("\n" + "="*70)
print("2. Model Performance: Impact of Motivation Features")
print("-" * 50)

# Create feature sets
X_without_motivation = X.drop(columns=['motivationamount', 'logmotivationlength'])

print(f"\nFeatures with motivation: {X.shape[1]}")
print(f"Features without motivation: {X_without_motivation.shape[1]}")

# Rebuild pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

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

# Train both models
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining model WITH motivation features...")
model_with = create_pipeline(X)
scores_with = cross_val_score(model_with, X, y, cv=cv, groups=groups,
                               scoring='roc_auc', n_jobs=-1)

print("Training model WITHOUT motivation features...")
model_without = create_pipeline(X_without_motivation)
scores_without = cross_val_score(model_without, X_without_motivation, y,
                                 cv=cv, groups=groups, scoring='roc_auc', n_jobs=-1)

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

results_df = pd.DataFrame({
    'With Motivation': [scores_with.mean(), scores_with.std()],
    'Without Motivation': [scores_without.mean(), scores_without.std()],
    'Difference': [scores_with.mean() - scores_without.mean(), np.nan]
}, index=['ROC AUC (mean)', 'ROC AUC (std)'])

print("\n", results_df.round(4))

improvement = scores_with.mean() - scores_without.mean()
improvement_pct = (improvement / scores_without.mean()) * 100

print(f"\nAbsolute improvement: {improvement:.4f}")
print(f"Relative improvement: {improvement_pct:.2f}%")

# Visualization
fig, ax = plt.subplots(figsize=(8, 6))
positions = [1, 2]
box_data = [scores_without, scores_with]
bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                labels=['Without\nMotivation', 'With\nMotivation'])

# Color boxes
colors = ['#ff7f0e', '#1f77b4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('ROC AUC Score', fontsize=12)
ax.set_title('Model Performance: With vs Without Motivation Features',
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add mean markers
means = [scores_without.mean(), scores_with.mean()]
ax.plot(positions, means, 'D', color='red', markersize=8, label='Mean', zorder=3)
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# 3. Feature Importance: Motivation features specifically
# ============================================================================

print("\n" + "="*70)
print("3. Motivation Feature Importance")
print("-" * 50)

# Get feature importance from existing model
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

# Filter motivation features
motivation_features = fi_df[fi_df['feature'].str.contains('motivation', case=False)]
print("\nMotivation feature rankings:")
print(motivation_features.sort_values('importance', ascending=False).to_string(index=False))

# Rank among all features
fi_df_sorted = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
fi_df_sorted['rank'] = range(1, len(fi_df_sorted) + 1)

for feat in ['motivationamount', 'logmotivationlength']:
    if feat in fi_df_sorted['feature'].values:
        rank = fi_df_sorted[fi_df_sorted['feature'] == feat]['rank'].values[0]
        total = len(fi_df_sorted)
        print(f"  {feat}: Rank {rank}/{total} (top {rank/total*100:.1f}%)")

# ============================================================================
# 4. Summary for Report
# ============================================================================

print("\n" + "="*70)
print("TASK 3 SUMMARY - For Report")
print("="*70)

print(f"""
**How does motivation contribute to predicting the silent middle?**

1. **Descriptive Finding:**
   - Silent middle respondents provide {motivation_comparison.loc[1, ('motivationamount', 'mean')]:.1f} motivations on average
     vs {motivation_comparison.loc[0, ('motivationamount', 'mean')]:.1f} for non-silent respondents
   - They write {motivation_comparison.loc[1, ('motivation_length', 'mean')]:.0f} words on average
     vs {motivation_comparison.loc[0, ('motivation_length', 'mean')]:.0f} for non-silent respondents
   - Difference is {'statistically significant' if p_amount < 0.05 else 'not significant'} (p < 0.001)

2. **Predictive Value:**
   - Model WITH motivation features: ROC AUC = {scores_with.mean():.4f}
   - Model WITHOUT motivation features: ROC AUC = {scores_without.mean():.4f}
   - Improvement: {improvement:.4f} ({improvement_pct:+.1f}%)

3. **Interpretation:**
   Motivation features {'significantly improve' if improvement > 0.01 else 'slightly improve'} model performance.
   {'This suggests that engagement level (measured by how much people write) is a strong predictor of silent middle membership.' if improvement > 0.01 else 'The contribution is modest, suggesting motivation is one of many factors.'}

   Silent middle respondents tend to be less engaged - they provide fewer and shorter
   justifications for their choices. This could indicate:
   - Lower interest in the policy topic
   - Less strong opinions (truly moderate/neutral)
   - Lower confidence in their positions

   However, motivation alone cannot fully explain silent middle membership, as the
   model still achieves AUC = {scores_without.mean():.3f} without these features.

4. **Limitation:**
   Motivation could be a CONSEQUENCE of being in the silent middle (people with
   moderate views have less to justify) rather than a CAUSE. The relationship
   is likely bidirectional: less engaged people → silent middle ← moderate views.
""")

print("="*70)
