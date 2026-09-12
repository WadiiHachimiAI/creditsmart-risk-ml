# CreditSmart Analytics — Bank Credit Risk ML

Regression and classification models for a Moroccan fintech use case: estimating credit amounts and predicting default risk from 612 real credit applications, built with scikit-learn.

## Overview

Two complementary models on the same credit application dataset:

- **Regression** — estimate the appropriate credit amount (`montant_credit_mad`) for an applicant
- **Classification** — predict whether an applicant will default on payment (`defaut_paiement`)

The goal: faster, more objective, less risky credit decisions for a fintech serving banks and microfinance institutions.

## Dataset

612 credit applications with 16 features: demographics (age, gender, city, marital status, education), employment (sector, years employed, monthly income), credit history, existing debt ratio, and the two prediction targets.

## Data Preparation

- Missing numerical values (years employed, income, dependents, debt ratio) filled with the **median**
- Missing categorical values (credit history) filled with the **mode**
- No duplicate rows found

## Results

### Regression — Credit Amount

| Metric | Value |
|---|---|
| MAE | 30,443 MAD |
| RMSE | 49,482 MAD |
| **R²** | **0.533** (test) |

R² of 0.53 means the model captures roughly half the variance in credit amounts — a moderate result, suggesting credit amount depends on factors beyond what's in this feature set (or a non-linear relationship a simple Linear Regression can't fully capture).

### Classification — Default Risk

Compared four models, evaluated not just on accuracy but on **recall for the default class** — in a lending context, failing to catch an actual defaulter (a false negative) is far more costly than a false alarm.

| Model | Accuracy | Recall (Default) | Est. Cost of Missed Defaults |
|---|---|---|---|
| SVM | 62.6% | **84.1%** | 1,155,000 MAD |
| KNN | 48.0% | 58.0% | 3,045,000 MAD |
| Decision Tree | 65.0% | 66.7% | 2,415,000 MAD |
| Random Forest | **72.4%** | 72.5% | 1,995,000 MAD |

**SVM was selected as the best model** — despite having the lowest raw accuracy, it catches the most actual defaulters (84% recall), translating to the lowest estimated financial loss from missed defaults (assuming ~105,000 MAD cost per missed default). This reflects a real business tradeoff: in credit risk, optimizing for accuracy alone can hide a model that lets expensive defaults slip through.

5-fold cross-validation (recall-focused) confirmed SVM as the most *stable* model (lowest variance across folds), while Decision Tree and Random Forest showed signs of overfitting (perfect recall on training data, notably lower on test/cross-validation).

**Top 3 features driving predictions:** monthly income, repayment capacity, and risk score.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## What I'd Improve Next

- Address class imbalance (56% default vs. 44% non-default) with SMOTE or class weighting
- Try a non-linear regression approach (Random Forest Regressor, Gradient Boosting) to improve on the R² of 0.53
- Tune the classification threshold on SVM specifically to balance recall against precision, rather than using the default 0.5 cutoff
