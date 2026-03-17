# end-to-end-mlflow-pipeline
## Diabetes Disease Progression — MLflow Random Forest Regressor

A machine learning pipeline that trains a `RandomForestRegressor` on the sklearn Diabetes dataset, tunes hyperparameters via `GridSearchCV`, and tracks experiments with **MLflow**.

---

## Overview

This project predicts a quantitative measure of diabetes disease progression one year after baseline using ten clinical features. The pipeline covers data preparation, hyperparameter tuning, model evaluation, and full experiment tracking with MLflow's Model Registry.

---

## Dataset

**Source:** [`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

| Property | Details |
|---|---|
| Samples | 442 |
| Features | 10 (numeric, mean-centered & scaled) |
| Target | Quantitative disease progression score |

**Features:** `age`, `sex`, `bmi`, `bp`, `s1` (total cholesterol), `s2` (LDL), `s3` (HDL), `s4` (TCH), `s5` (triglycerides), `s6` (blood sugar)

---

## Project Structure

```
.
├── notebook.ipynb          # Main pipeline (data prep → train → log)
├── mlflow.db               # SQLite backend store for MLflow
├── mlruns/                 # MLflow artifact root
└── README.md
```

---

## Requirements

```bash
pip install pandas scikit-learn mlflow
```

---

## Usage

### 1. Start the MLflow Tracking Server

```bash
python -m mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

### 2. Run the Pipeline

Execute the notebook or script. The pipeline will:

1. Load and prepare the Diabetes dataset
2. Split data into train/test sets (80/20)
3. Run `GridSearchCV` across the hyperparameter grid
4. Log the best parameters, MSE metric, and model to MLflow
5. Register the model as `best_regression_model` in the MLflow Model Registry

### 3. View Results

Open the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Hyperparameter Grid

| Parameter | Values |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 5, 10 |
| `min_samples_split` | 5, 10 |
| `min_samples_leaf` | 2, 5 |

> **Note:** `None` was removed from `min_samples_split` — sklearn's `RandomForestRegressor` requires an `int ≥ 2` or a `float` in `(0.0, 1.0]` for this parameter. Passing `None` causes a `InvalidParameterError`.

---

## Results (Sample Run)

| Parameter | Best Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| **MSE** | **3527.08** |

---

## MLflow Tracking

The following are logged per run:

- **Parameters:** `best_n_estimators`, `best_max_depth`, `best_min_samples_split`, `best_min_samples_leaf`
- **Metrics:** `mse`
- **Model:** Registered under `best_regression_model` in the MLflow Model Registry (versioned)