# Practical Application III: Comparing Classifiers

**Author**: Jelani Gould-Bailey

## Overview

This project compares the performance of four classification models — Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM) — on a bank marketing dataset from the UCI Machine Learning Repository. The goal is to predict whether a customer will subscribe to a term deposit based on demographic and campaign interaction data.

[View the full analysis in the Jupyter Notebook](prompt_III.ipynb)

## Dataset

The dataset contains 41,188 records and 20 features from a Portuguese banking institution, covering multiple marketing campaigns. The target variable `y` indicates whether a client subscribed to a term deposit. The data is significantly imbalanced: ~89% of customers did not subscribe.

## Key Findings

- **Best Model: Decision Tree** (with tuned hyperparameters) achieved the strongest performance across all meaningful metrics: ROC-AUC of 0.74, F1 of 0.57, Recall of 0.51, and Test Accuracy of 91.8%.
- **Class Imbalance**: Accuracy alone is misleading on this dataset — a model that always predicts "no" would score 88.7%. ROC-AUC, F1, Precision, and Recall were used for a more complete evaluation.
- **Hyperparameter Tuning**: `RandomizedSearchCV` was used in place of `GridSearchCV` due to computational constraints on the ~41k row dataset. Tuning successfully resolved the Decision Tree's overfitting (100% train accuracy with default params) and improved all metrics.
- **Key Predictors**: Call duration, prior campaign outcome, and macroeconomic indicators (euribor rate, employment variation) are the strongest predictors of subscription.

## Model Comparison Summary

| Model | Precision | Recall | F1 | ROC-AUC | Test Accuracy |
|-------|-----------|--------|----|---------|---------------|
| Logistic Regression | 0.6629 | 0.4033 | 0.5014 | 0.6886 | 91.2% |
| KNN | 0.5550 | 0.3419 | 0.4231 | 0.6535 | 90.2% |
| **Decision Tree** | **0.6345** | **0.5148** | **0.5678** | **0.7386** | **91.8%** |
| SVM | 0.6523 | 0.3370 | 0.4443 | 0.6571 | 91.2% |

## Actionable Recommendations

1. **Call duration matters** — Longer calls are strongly associated with success. Coach sales teams to keep prospects engaged.
2. **Re-target prior subscribers** — Customers who subscribed in a prior campaign are significantly more likely to subscribe again.
3. **Time campaigns to economic conditions** — Macroeconomic features (interest rates, employment) are highly predictive; favorable windows improve conversion.
4. **Prioritize cellular contact** — Cellular outreach yields higher subscription rates than landline telephone.

## Next Steps

- Apply `class_weight='balanced'` or SMOTE to improve recall on the minority class
- Tune the decision threshold below 0.5 to optimize the precision/recall tradeoff for business needs
- Evaluate ensemble methods (Random Forest, XGBoost) as natural next candidates
