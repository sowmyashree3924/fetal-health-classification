# Fetal Health Classification

Classifies cardiotocography (CTG) recordings into three fetal-health classes from 21 numeric
features, comparing Logistic Regression, Decision Tree, Random Forest and XGBoost.

> **Research / educational project. Not a medical device and not for clinical decisions.**

## Data

`data/processed/` holds the ready-to-use split: **1,690 training rows and 423 test rows**
(2,113 total, stratified so class proportions match in both files, no duplicate rows, no missing values).

The raw file and the preprocessing that produced these CSVs are **not part of this repository**
(`data/raw/` is empty). The data appears to be the UCI Cardiotocography dataset (2,126 records,
which would be 2,113 after removing 13 duplicates). Please add the preprocessing step (a
`01_Data_Preparation` notebook or a script in `src/`) so the split can be rebuilt.

### Class labels

The target `Fetal_Health` holds the integers 0, 1, 2, and the files do not record what they mean.
The class sizes (1,646 / 175 / 292) are consistent with the UCI classes
Normal (1,655), Pathological (176) and Suspect (295) after removing 13 duplicates, which would be
produced by an alphabetical label encoding:

| id | likely meaning | rows (train + test) |
|---|---|---|
| 0 | Normal | 1,646 |
| 1 | Pathological | 175 |
| 2 | Suspect | 292 |

**This mapping is inferred, not confirmed** - check it against your original preprocessing code and
correct this table if it is wrong. Note that the ids are *not* ordered by severity. Plots and
tables use the raw ids for that reason.

## Project structure

```
data/processed/   train.csv, test.csv, selected_features.txt
data/raw/         (empty - raw data and preprocessing not included)
notebooks/        02_Model_Training.ipynb
src/              evaluation.py  (metrics, bootstrap CI, confusion-matrix helper)
models/           tuned_model.pkl, reduced_feature_model.pkl, model_metadata.json
results/          CSV result tables and figures/
requirements.txt
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/02_Model_Training.ipynb   # Run > Restart Kernel and Run All Cells
```

The notebook runs top to bottom and regenerates everything in `models/` and `results/`.
Pickled models are tied to the library versions in `requirements.txt`
(see `models/model_metadata.json`).

## Method

1. Models are compared with **5x3 repeated stratified cross-validation on the training set**
   (Macro F1, because class 0 is about 78% of the data).
2. The best model is tuned with `GridSearchCV` on the training set only.
3. The **test set is used once**, for the final report, with per-class metrics and a bootstrap
   confidence interval.
4. Feature selection uses Random Forest importance (training set only) to keep the top 10 features,
   compared like-for-like against all 21 features.

## Results

Test set = 423 rows (only 35 and 58 rows in the two minority classes).

| Model | CV Macro F1 | Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression | 0.7802 | 0.9362 | 0.8745 |
| Decision Tree | 0.8547 | 0.9078 | 0.8389 |
| Random Forest | 0.8808 | 0.9669 | 0.9383 |
| XGBoost | 0.8969 | 0.9645 | 0.9358 |
| XGBoost (tuned) | 0.9005 | 0.9716 | 0.9511 |

Tuned XGBoost (`{'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 200}`): test Macro F1 **0.9511**
(95% bootstrap CI 0.921-0.977),
test accuracy 0.9716. Cross-validated Macro F1 is lower (0.901), so
treat the CV figure as the more conservative estimate.

Per-class test recall for the tuned model is 0.997 (class 0), 0.943 (class 1) and 0.845 (class 2).
Nine of the 58 class-2 cases were predicted as class 0, which is the most important error to
reduce in any follow-up work.

### 21 vs 10 features (same settings)

| Model | Feature set | CV Macro F1 (mean) | Test Accuracy | Test Macro F1 |
|---|---|---|---|---|
| Random Forest (200 trees) | All 21 features | 0.8808 | 0.9669 | 0.9383 |
| Random Forest (200 trees) | Top 10 features | 0.8784 | 0.9574 | 0.9227 |
| XGBoost (tuned params) | All 21 features | 0.8981 | 0.9716 | 0.9511 |
| XGBoost (tuned params) | Top 10 features | 0.8932 | 0.9669 | 0.9379 |

Dropping to 10 features changes cross-validated Macro F1 by 0.002-0.005, well inside its
fold-to-fold spread (about 0.02-0.03), while test Macro F1 falls by 0.013-0.016 (a handful of
test samples). So 10 features is *about* as good, but this experiment cannot prove it is equal.
Permutation importance agrees with the impurity-based top 10 on only 6 of 10 features, partly
because `Mean`, `Mode` and `Median` are highly correlated, so the exact subset is somewhat arbitrary.

**Recommended model:** `models/tuned_model.pkl` (tuned XGBoost, all 21 features). Use
`reduced_feature_model.pkl` only if fewer input features matter more than about 0.01 Macro F1.

## Known limitations

- Single train/test split with very few minority-class test samples: differences below roughly
  0.02 Macro F1 are not reliable.
- The three classes are imbalanced; `class_weight="balanced"` did not improve cross-validated
  Macro F1 for Random Forest in a quick check, so it is not used.
- No probability calibration or decision-threshold tuning has been done.
