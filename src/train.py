from src.preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def build_baseline_model(cat_cols, num_cols):
    preprocessor = build_preprocessor(cat_cols, num_cols)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ])

    return model

def build_rf_model(cat_cols, num_cols):
    preprocessor = build_preprocessor(cat_cols, num_cols)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    return model