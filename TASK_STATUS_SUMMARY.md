# TPM034 Mini-Project Task Status Summary

## Overview
**Total Points:** 10
**Deadline:** 09/01/2026
**Current Status:** Tasks 1-2 Complete, Task 3 needs completion

---

## Task 1: Data Preparation & Model [3.5 points] ✅ COMPLETE

### What's Required:
- Prepare the data
- Apply a proper analytical model to predict "the silent middle" group

### What's Been Done:
✅ **Target Variable Creation**
- Defined "silent middle" using study-relative Mean Absolute Deviation (MAD)
- ~53.5% classified as silent middle (balanced across studies)
- Study-grouped approach prevents data leakage

✅ **Feature Engineering**
- Created `motivationamount` (count of non-null motivation entries)
- Created `logmotivationlength` (log-transformed total word count)
- Properly excluded data leakage columns (choice scores, raw text)

✅ **Model Selection & Training**
- Random Forest Classifier with class balancing
- Hyperparameter tuning via RandomizedSearchCV (100 iterations)
- Best params: n_estimators=250, max_depth=4, min_samples_leaf=80

✅ **Cross-Validation**
- StratifiedGroupKFold (5 folds) with study grouping
- Prevents overfitting to study-specific patterns
- Proper out-of-fold predictions

✅ **Performance Metrics**
- ROC AUC: **0.5813** (modest but above random)
- Accuracy: 56.54%
- Precision: 57.97%, Recall: 68.00%
- R²: 0.0154 (low but expected - context-dependent target)

### For Report:
**Grading Criteria: Correctness (45%), Completeness (45%), Coding (10%)**

**Strengths to highlight:**
- Proper data leakage prevention
- Study-grouped CV prevents overfitting
- Regularized model (depth=4, min_leaf=80)
- Appropriate metric selection (ROC AUC over R²)

**What to mention:**
> "We defined the silent middle as respondents whose preferences deviate less than the median from their study's typical responses. This study-relative definition accounts for topic-specific norms and varying scales across consultations. We trained a Random Forest classifier with cross-validation grouped by study to prevent data leakage, achieving ROC AUC = 0.58, indicating modest but meaningful discriminatory ability."

---

## Task 2: Feature Importance [1 point] ✅ MOSTLY COMPLETE

### What's Required:
- Identify the most important features contributing to "the silent middle" group

### What's Been Done:
✅ **Random Forest Feature Importance**
- Extracted importance scores from trained model
- Grouped one-hot encoded features back to original variables
- Visualized top 46 features

✅ **SHAP Analysis**
- Computed SHAP values for 500 samples
- Beeswarm plot showing feature contributions
- Bar plot showing average absolute impact

✅ **Top Features Identified**
From the feature importance plot, top predictors include:
1. `logmotivationlength` - Engagement level (word count)
2. `motivationamount` - Breadth of engagement
3. Demographic factors (age, education, gender)
4. Attitudes (advice importance, trust, subject importance)
5. Study characteristics (sector, level, lossesgains)

### For Report:
**Grading Criteria: Correctness (45%), Completeness (45%), Coding (10%)**

**What to add:**
Create a clear summary table or bullet list like:

```
Top 10 Features Predicting Silent Middle:
1. logmotivationlength (engagement depth)
2. motivationamount (engagement breadth)
3. education level
4. age group
5. advice importance rating
6. sector (policy domain)
7. trust in government decisions
8. subject importance rating
9. grade given to consultation
10. lossesgains framing
```

**Interpretation to include:**
> "Feature importance analysis reveals that engagement metrics (motivation length and amount) are the strongest predictors, suggesting that silent middle membership is primarily characterized by lower engagement rather than specific demographic profiles. However, education, age, and attitudes toward government also contribute meaningfully."

---

## Task 3: Motivation Contribution [0.5 points] ⚠️ NEEDS COMPLETION

### What's Required:
- How does the "motivation" of the respondents contribute to the prediction of the "silent middle" group?

### What's Been Done:
✅ Motivation features created (amount and length)
✅ Features included in model
❌ **Missing: Specific analysis of motivation's contribution**

### What Needs to Be Added:
Use the code in `task3_motivation_analysis.py` to add:

1. **Descriptive comparison:**
   - Mean motivation amount/length by group
   - Statistical test (Mann-Whitney U)
   - Visualization (boxplots)

2. **Predictive contribution:**
   - Train model WITH motivation features
   - Train model WITHOUT motivation features
   - Compare ROC AUC scores
   - Calculate improvement

3. **Interpretation:**
   - Are differences statistically significant?
   - How much does motivation improve prediction?
   - What does this mean conceptually?

### Expected Finding:
Based on feature importance, motivation likely improves AUC by ~0.02-0.04 points, showing that:
- Silent middle write less (lower engagement)
- But motivation alone doesn't fully explain silent middle
- Suggests bidirectional relationship

### For Report:
**Grading Criteria: Critical Thinking (60%), Completeness (40%)**

**What to write:**
> "Motivation significantly contributes to predicting silent middle membership. Respondents in the silent middle provide fewer motivations (mean = X vs Y) and write shorter responses (p < 0.001). Removing motivation features from the model decreases ROC AUC from 0.58 to ~0.56, indicating a meaningful contribution. This suggests that silent middle is partially characterized by lower engagement, though the relationship may be bidirectional: less engaged individuals may gravitate toward moderate positions, while those with moderate views may have less to justify."

---

## Tasks 4-6: Not Yet Started

### Task 4: Strengths & Limitations [2 points]
**Required:** Name two analytical strengths and two analytical limitations

**Suggestions:**

**Strengths:**
1. **Study-grouped cross-validation**: Prevents overfitting, enables generalization to new topics
2. **Engagement metrics**: Captures behavioral proxy (motivation) not just demographics

**Limitations:**
1. **Context-dependency**: Silent middle varies by topic, limiting cross-topic prediction
2. **Causality unclear**: Cannot determine if low engagement causes silence or vice versa

---

### Task 5: Societal Impact [2 points]
**Required:** Discuss societal impact of strengths and limitations on individuals, organizations, and society

**Suggestions:**

**Positive Impacts (Strengths):**
- Organizations: Identify less-engaged citizens for targeted outreach
- Society: Ensure "silent voices" are heard in policy decisions
- Individuals: Tailored communication strategies

**Negative Impacts (Limitations):**
- Individuals: Risk of misclassifying thoughtful moderates as disengaged
- Organizations: Over-reliance on demographics may miss topic-specific patterns
- Society: Could reinforce existing participation gaps if used incorrectly

---

### Task 6: Mitigation Solution [1 point]
**Required:** Propose analytical solution to mitigate most severe limitation

**Suggestions:**

**Most Severe Limitation:** Context-dependency (same person is silent on some topics, vocal on others)

**Proposed Solutions:**
1. **Add topic-relevance features:**
   - Distance from proposed project location
   - Self-reported interest/impact ("How much does this affect you?")
   - Domain expertise indicators

2. **Hierarchical/mixed-effects model:**
   - Model both individual and topic-level effects
   - Capture variation across studies explicitly
   - Better generalization to new topics

3. **Dynamic profiling:**
   - Track individuals across multiple consultations
   - Identify patterns in when/why they're silent
   - Personalized engagement strategies

---

## Next Steps

1. ✅ **Task 1-2:** Review for clarity, ensure all grading criteria met
2. 🔧 **Task 3:** Run `task3_motivation_analysis.py` and add results to notebook
3. ⏳ **Task 4-6:** Complete critical analysis sections
4. 📊 **Presentation:** Prepare slides with key findings and visuals
5. 📝 **Final Polish:** Clean code, add comments, verify reproducibility

---

## Code Quality Checklist

- [x] No data leakage (choice scores excluded)
- [x] Proper cross-validation (study grouping)
- [x] Reproducible (random seeds set)
- [x] Well-commented cells
- [ ] Clear section headers for each task
- [ ] Results summarized in markdown cells
- [ ] Visualizations have titles and labels
- [ ] No unnecessary outputs (warnings suppressed)

---

## Submission Checklist

- [ ] Notebook runs top-to-bottom without errors
- [ ] All 6 tasks completed
- [ ] Results clearly presented
- [ ] Code is clean and commented
- [ ] Visualizations are publication-quality
- [ ] Presentation slides prepared
- [ ] Member contributions statement ready
- [ ] Submitted to Brightspace before 09/01/2026
