# Fetal Health Classification

Machine-learning classification of cardiotocography (CTG) recordings into three
fetal-health classes using 21 numeric features. The project compares Logistic
Regression, Decision Tree, Random Forest, and XGBoost models and selects a tuned
XGBoost model for the final result.

> **Research/educational project.** This software is not a medical device and
> must not be used for clinical decisions.

## Project status

- Processed data, trained models, evaluation tables, and plots are included.
- The raw UCI file and the preprocessing script are **not** included;
  `data/raw/` is empty.
- The notebooks are the reproducible analysis and training workflow. The
  training notebook can be run directly with the included processed CSV files.
- The recommended saved model is `models/tuned_model.pkl`.

## Dataset

The project uses the [UCI Cardiotocography dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography).
The original data contained 2,126 observations and 21 CTG-derived input
features. Data inspection found no missing values and 13 duplicate rows.
After removing duplicates, 2,113 unique observations remained.

| Property | Value |
|---|---:|
| Original observations | 2,126 |
| Duplicate rows removed | 13 |
| Final observations | 2,113 |
| Numeric input features | 21 |
| Missing values | 0 |
| Training rows | 1,690 |
| Test rows | 423 |

The train/test split is stratified and uses a fixed random seed of 42. The
class distribution in the processed data is:

| Stored ID | Likely source label | Rows | Percentage |
|---:|---|---:|---:|
| 0 | Normal | 1,646 | 77.90% |
| 1 | Pathological | 175 | 8.28% |
| 2 | Suspect | 292 | 13.82% |

The semantic mapping above is inferred from the source class counts and is not
stored in the processed CSV files. The integer IDs are not severity rankings.
Use the raw IDs when reproducing the plots and metrics unless the mapping has
been confirmed against the original preprocessing.

## Repository structure

```text
data/
  processed/
    train.csv                 # 1,690 rows, 21 features + Fetal_Health
    test.csv                  # 423 rows, 21 features + Fetal_Health
    selected_features.txt    # top 10 features used in the reduced model
  raw/                        # intentionally empty in this repository
notebooks/
  01_data_analysis.ipynb     # exploratory analysis and data summaries
  02_Model_Training.ipynb    # model comparison, tuning, evaluation
src/
  evaluation.py              # metrics, bootstrap CI, confusion-matrix helper
models/
  tuned_model.pkl
  reduced_feature_model.pkl
  model_metadata.json
results/
  *.csv                      # metrics, tuning, EDA, and feature summaries
  *.png                      # generated figures
fetal_health_classification_report.pdf
fetal_health_classification_one_page_summary.pdf
requirements.txt
```

## Installation and setup

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use a process-scoped policy change:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pinned package versions are recorded in `requirements.txt` and the
environment used to create the saved models is recorded in
`models/model_metadata.json`. The metadata identifies Python 3.12.3,
scikit-learn 1.9.1, XGBoost 3.2.0, pandas 3.0.6, and NumPy 2.4.6.

## Running the notebooks

Start Jupyter from the project root:

```bash
jupyter lab
```

Then run each notebook with **Kernel > Restart Kernel and Run All Cells**.

1. `notebooks/01_data_analysis.ipynb` performs exploratory analysis. It
   expects access to the original UCI data. Because the raw data and
   preprocessing are not included, this notebook may require downloading the
   dataset and installing the notebook's data-access dependency separately.
2. `notebooks/02_Model_Training.ipynb` loads `data/processed/train.csv` and
   `data/processed/test.csv`, compares models, tunes XGBoost, evaluates the
   held-out test set, and regenerates files under `models/` and `results/`.

The test set must not be used during model selection. The supplied workflow
uses it once for final reporting.

## Method

1. Inspect the data, remove duplicate rows, and retain the 21 numeric CTG
   features.
2. Use an 80:20 stratified train/test split with `random_seed = 42`.
3. Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost
   using 5x3 repeated stratified cross-validation.
4. Use Macro F1 as the primary selection metric because the classes are
   imbalanced.
5. Tune the strongest candidate with `GridSearchCV` using training data only.
6. Evaluate the selected model on the untouched test set, including
   per-class metrics and a 2,000-sample percentile bootstrap confidence
   interval for Macro F1.
7. Compare all 21 features with a top-10 subset selected from Random Forest
   feature importance.

## Results

The final test set contains 423 rows, including only 35 and 58 examples in
the two minority classes.

| Model | CV Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.7802 | 0.9362 | 0.8745 |
| Decision Tree | 0.8547 | 0.9078 | 0.8389 |
| Random Forest | 0.8808 | 0.9669 | 0.9383 |
| XGBoost | 0.8969 | 0.9645 | 0.9358 |
| **XGBoost (tuned)** | **0.9005** | **0.9716** | **0.9511** |

The tuned XGBoost parameters are:

```python
{
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 1,
    "n_estimators": 200,
}
```

Its test Macro F1 is 0.9511 with a 95% bootstrap confidence interval of
0.9205–0.9765. The cross-validated Macro F1 of 0.9005 is the more conservative
generalization estimate. Test recall by stored class ID is 0.997 for class 0,
0.943 for class 1, and 0.845 for class 2. Nine class-2 cases were predicted
as class 0.

### Feature-selection comparison

| Model | Feature set | CV Macro F1 | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|---:|
| Random Forest | All 21 | 0.8808 | 0.9669 | 0.9383 |
| Random Forest | Top 10 | 0.8784 | 0.9574 | 0.9227 |
| Tuned XGBoost | All 21 | 0.8981 | 0.9716 | 0.9511 |
| Tuned XGBoost | Top 10 | 0.8932 | 0.9669 | 0.9379 |

Use the full-feature `tuned_model.pkl` when predictive performance is the
priority. Use `reduced_feature_model.pkl` only when requiring fewer inputs is
more important than approximately 0.01 Macro F1.

## Using the saved model

The tuned model expects the 21 feature columns listed in
`models/model_metadata.json`. A minimal prediction check is:

```python
import json
import joblib
import pandas as pd

model = joblib.load("models/tuned_model.pkl")
metadata = json.load(open("models/model_metadata.json", encoding="utf-8"))
test = pd.read_csv("data/processed/test.csv")

features = metadata["tuned_model"]["features"]
predictions = model.predict(test[features])
print(predictions[:10])
print("Predictions generated:", len(predictions))
```

For a formal evaluation, compare `predictions` with
`test["Fetal_Health"]` using the helpers in `src/evaluation.py`, rather than
using predictions as a clinical conclusion.

## Generated outputs

The training notebook writes model and analysis outputs to:

- `models/model_metadata.json`: target name, random seed, parameters, features,
  metrics, and library versions.
- `results/model_results.csv`: held-out metrics for each classifier.
- `results/cv_model_comparison.csv`: cross-validation means and standard
  deviations.
- `results/tuning_results.csv`: tuned XGBoost parameters and confidence
  interval.
- `results/feature_selection_comparison.csv`: 21-feature versus top-10
  comparison.
- `results/eda_summary.csv`, `feature_summary.csv`, and
  `class_feature_means.csv`: exploratory summaries.
- `results/*.png`: class distribution, feature analysis, model evaluation,
  and confusion-matrix figures.

## Limitations and responsible use

- Results rely on one stratified train/test split and small minority-class
  test supports; differences below roughly 0.02 Macro F1 should not be
  over-interpreted.
- The data are class-imbalanced. A quick Random Forest check found that
  `class_weight="balanced"` did not improve cross-validated Macro F1, so it
  is not used in the reported model.
- Probability calibration and decision-threshold tuning were not performed.
- The target-label mapping in this repository is inferred rather than
  preserved by the raw preprocessing pipeline.
- The data and model are not validated for clinical deployment, prospective
  monitoring, demographic fairness, or generalization to other hospitals or
  populations.
- No clinical interpretation should be made from a model prediction.
