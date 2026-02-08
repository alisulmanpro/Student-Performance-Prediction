# Student Performance Prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)
[![Python](https://img.shields.io/badge/python-3.14%2B-green)](#)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)](#)

> Predict final student grades (G3) from demographic, school, and historical grade features.

---

<!-- PLACEHOLDER: Replace with project banner or demo GIF -->

<img width="1280" height="619" alt="Image" src="https://github.com/user-attachments/assets/e2f7ffe9-c093-4f93-bd74-a32a94bfd638" />

---

## Table of Contents

* [About](#about)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Dataset](#dataset)
* [Quickstart](#quickstart)
* [Preprocessing & Modeling](#preprocessing--modeling)
* [Evaluation](#evaluation)
* [API](#api)
* [Demo / Example Requests](#demo--example-requests)
* [Deployment](#deployment)
* [Roadmap & Improvements](#roadmap--improvements)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## About

This project trains and ships a production-ready machine learning pipeline to predict final student grades (`G3`) using the UCI Student Performance dataset. The delivered artifact is a saved sklearn `Pipeline` (preprocessing + model) and a FastAPI application for inference.

Key goals:

* Reproducible preprocessing with `ColumnTransformer` and `Pipeline`
* Robust baseline and ensemble models (Linear Regression, Random Forest)
* Clear evaluation and model persistence
* Lightweight FastAPI inference endpoint

---

## Features

* End-to-end pipeline: data → preprocessing → training → evaluation → model saving
* Preprocessing implemented with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`
* Baseline and Random Forest regression models
* Saved sklearn pipeline for zero-drift inference
* FastAPI server with Pydantic input validation

---

## Repository Structure

```
student-performance-prediction/
├── data/
│   └── raw/                 # Raw data files
│
├── models/                  # Saved models
│   └── student_performance_rf.pkl
├── notebooks/               # EDA and experiments
│   └── 01_eda.ipynb
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── main.py                  # FastAPI application entry point
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
├── uv.lock                  # Dependency lock file
├── .gitignore
├── .python-version
└── README.md
```

---

## Dataset

* Source: UCI Student Performance dataset (Kaggle mirror recommended)
* Files: `student-mat.csv` (semicolon-separated)
* Target column: `G3` (final grade, range 0–20)

**Note:** keep raw data under `data/raw/` and never commit sensitive/raw files to public repos.

---

## Quickstart

1. Create and activate a virtual environment

```bash
python -m venv .venv
# mac / linux
source .venv/bin/activate
# windows (powershell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

2. Place `student-mat.csv` in `data/raw/`
3. Run training (example)

```bash
python src/train.py --data_path data/raw/student-mat.csv --output models/student_performance_rf.pkl
```

4. Start the API

```bash
fastapi dev main.py
# open http://127.0.0.1:8000/docs
```

---

## Preprocessing & Modeling

* Numerical features: median imputation + `StandardScaler`
* Categorical features: most-frequent imputation + `OneHotEncoder(handle_unknown='ignore')`
* Preprocessing implemented via `build_preprocessor(cat_cols, num_cols)` in `src/preprocessing.py`
* Models available in `src/train.py` (Linear Regression baseline, Random Forest)

---

## Evaluation

Model evaluation scripts (`src/evaluate.py`) produce MAE, RMSE and R² metrics.

**Example baseline results (expected range):**

* MAE ≈ 1.2–1.8
* RMSE ≈ 2.0–2.5
* R² ≈ 0.6–0.85

<!-- PLACEHOLDER: Add model performance chart -->

<img width="1086" height="460" alt="Image" src="https://github.com/user-attachments/assets/89be779a-cadd-4435-9e40-b0e849a87606" />

---

## API

The FastAPI application loads the serialized sklearn Pipeline and exposes a `/predict` endpoint.

**Input**: JSON matching `app/schema.py` (Pydantic model)

**Output**: `{ "predicted_G3": float }

**Example request payload** (replace with realistic values):

1. The High-Achiever (Positive Test) `output: 18 - 20`

``` json
{
  "school": "GP", 
  "sex": "M", 
  "age": 17, 
  "address": "U", 
  "famsize": "LE3", 
  "Pstatus": "T",
  "Medu": 4, 
  "Fedu": 4, 
  "Mjob": "health", 
  "Fjob": "services", 
  "reason": "reputation",
  "guardian": "mother", 
  "traveltime": 1, 
  "studytime": 4, 
  "failures": 0, 
  "schoolsup": "no",
  "famsup": "yes", 
  "paid": "yes", 
  "activities": "yes", 
  "nursery": "yes", 
  "higher": "yes",
  "internet": "yes", 
  "romantic": "no", 
  "famrel": 5, 
  "freetime": 2, 
  "goout": 2,
  "Dalc": 1, 
  "Walc": 1, 
  "health": 5, 
  "absences": 0, 
  "G1": 18, 
  "G2": 19,
  "Dalc": 1, 
  "Walc": 1, 
  "health": 5, 
  "absences": 0, 
  "G1": 18, 
  "G2": 19
}
```

2. The At-Risk Student (Negative Test) `output: 0 - 6`

``` json
{
  "school": "MS", 
  "sex": "M", 
  "age": 19, 
  "address": "R", 
  "famsize": "GT3", 
  "Pstatus": "T",
  "Medu": 1, 
  "Fedu": 1, 
  "Mjob": "other", 
  "Fjob": "other", 
  "reason": "course",
  "guardian": "other", 
  "traveltime": 3, 
  "studytime": 1, 
  "failures": 3, 
  "schoolsup": "no",
  "famsup": "no", 
  "paid": "no", 
  "activities": "no", 
  "nursery": "no", 
  "higher": "no",
  "internet": "no", 
  "romantic": "yes", 
  "famrel": 2, 
  "freetime": 4, 
  "goout": 5,
  "Dalc": 3, 
  "Walc": 4, 
  "health": 2, 
  "absences": 20, 
  "G1": 5, 
  "G2": 4
}
```

3. The "Average" Student (Boundary Test) `output: 10 - 12`

```json
{
  "school": "GP", 
  "sex": "F", 
  "age": 16, 
  "address": "U", 
  "famsize": "GT3", 
  "Pstatus": "T",
  "Medu": 2, 
  "Fedu": 2, 
  "Mjob": "services", 
  "Fjob": "other", 
  "reason": "home",
  "guardian": "father", 
  "traveltime": 1, 
  "studytime": 2, 
  "failures": 0, 
  "schoolsup": "yes",
  "famsup": "yes", 
  "paid": "no", 
  "activities": "yes", 
  "nursery": "yes", 
  "higher": "yes",
  "internet": "yes", 
  "romantic": "no", 
  "famrel": 4, 
  "freetime": 3, 
  "goout": 3,
  "Dalc": 1, 
  "Walc": 2, 
  "health": 4, 
  "absences": 6, 
  "G1": 11, 
  "G2": 10
}
```

<img width="1086" height="576" alt="Image" src="https://github.com/user-attachments/assets/f92e316f-fc74-411e-b9d2-9b789b670894" />

---


## Roadmap & Improvements

* Hyperparameter tuning (Grid / Random / Bayesian)
* Cross-validation & CI checks
* Model explainability (SHAP) and fairness checks
* Monitoring: latency, error-rate, prediction drift
* Add unit & integration tests for API

---

## Contributing

Contributions are welcome. Please open an issue or PR. Follow the code style and add tests for new functionality.

---

## License

MIT License — see `LICENSE` file.

---

## Contact

Ali Sulman — [https://github.com/alisulmanpro](https://github.com/alisulmanpro)

<!-- END OF README -->
