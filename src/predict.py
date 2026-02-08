import joblib

def save_model(model, path: str):
    joblib.dump(model, path)