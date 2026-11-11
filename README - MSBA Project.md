# Should This Loan Be Approved? Predicting SBA Loan Default

**2024 CSU Systemwide Business Analytics Competition** · M.S. Business Analytics, California State University, Sacramento

Team: James Pease, Natalie Mukoda, Angela Flores Figueroa

## The Problem

The U.S. Small Business Administration guarantees part of each small-business loan, but banks still lose money when a borrower defaults. Using SBA national data, we built models to classify loan applications as lower risk (approve) or higher risk (deny), choosing the model and approval cutoff that **maximize the bank's net profit**, not just accuracy.

## Data

- **Source:** U.S. SBA National Data, 1987 to 2014, published by Li, Mickel, and Taylor (2018) for teaching and publicly available on Kaggle (not included here due to size)
- **Size:** 899,164 loans, 27 variables
- **Target:** `MIS_Status`, paid in full (0) or charged off (1)

## Approach

| Step | Notebook |
| --- | --- |
| Data cleaning: removed identifiers, redundant fields, and anything unknown at approval time (which would leak the outcome); addressed multicollinearity | `1 - Cleaning Data` |
| Classification trees, plus bagging, boosting, and random forest; tree feature importance used to trim weak predictors | `2 - Classification Trees` |
| k-Nearest Neighbors | `3 - KNN Group Project` |
| Logistic regression (statsmodels, Ridge, Lasso), plus Elastic Net in R | `4 - Logistic Regression`, `4-Logit-Models(Elastic-Net).pdf` |
| Discriminant analysis | `5 - Discriminant Analysis` |
| Neural networks | `6 - Neural Networks` |
| Model comparison on precision, recall, F1, specificity, and net profit | `7 - Model Comparison` |
| Final model: performance, lift and ROC charts, and the profit-maximizing cutoff | `8 - Question 3` |

Key design choices:

- **60/40 train/validation split**, predictors normalized where needed, and hyperparameters tuned with cross-validation
- **Class weights** (5 to 1 on defaults) because approving a loan that defaults costs far more than turning away a good one
- **Profit-based decision rule:** approving a loan that is repaid earns 5% of the disbursed amount; approving one that defaults loses 25%; denying a loan earns nothing. Each model's cutoff was set to maximize total profit.

## Results

| Model | Strength |
| --- | --- |
| Random Forest | Highest net profit (cutoff 0.43); recommended model |
| Bagging Tree | Highest F1 score (0.49) |
| Logistic Regression (Ridge) | Highest recall (0.93), catching the most defaults |
| Neural Network | Highest precision (0.52), but negative net profit |

The final Random Forest uses a compact set of predictors (loan amount, SBA-guaranteed amount, revolving credit, loan term, a recession flag, and urban or rural location) and reached 68% accuracy on the validation set. A full summary is in `SBA_Executive Summary.pdf`.

## Lessons Learned

Revisiting this project after graduation, I found an issue worth documenting:

**Net profit was measured on the training set instead of the validation set.** Each model's profit and optimal cutoff were calculated from its predictions on the same data it was trained on. That inflates results for flexible models that partly memorize their training data. The final Random Forest shows the gap clearly: 82% accuracy on training data versus 68% on validation data. Tree ensembles and KNN, which overfit most easily, ranked highest on profit, so the recommendation of Random Forest may partly reflect overfitting rather than true out-of-sample performance.

**What I'd do differently:**

1. Calculate net profit and choose each model's cutoff on the validation set, or better, tune the cutoff with cross-validation and report final profit on a held-out test set that no tuning decision ever touched.
2. Treat a large gap between training and validation performance as a red flag before comparing models, and regularize or constrain the model (for example, limit tree depth or require more samples per leaf) until the gap narrows.
3. Compare models on profit per loan or as a share of the best possible profit, which is easier to interpret than a total that depends on how many loans happen to be in a split.
4. Keep one shared pipeline for splitting, weighting, and scoring so every model is evaluated the same way. With several team members building models in separate notebooks, small differences in setup can creep in.

The broader takeaway: choosing a model on a business metric like profit was the right instinct, and it matters just as much *which data* that metric is measured on.
