import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "marine_10year_forecast_alerts.csv"
MODELS_DIR = ROOT / "models"

WINDOW_SIZE = 3
FEATURE_CANDIDATES = [
    ["chl", "o2", "thetao", "so"],
    ["chl", "o2", "sst", "salinity"],
]


def pick_feature_columns(df):
    for candidate in FEATURE_CANDIDATES:
        if set(candidate).issubset(df.columns):
            return candidate
    raise ValueError("Expected feature columns were not found in dataset.")


def build_windowed_xy(df, feature_cols, horizon_months=1):
    X = []
    y = []

    target_shift = horizon_months - 1
    last_index = len(df) - target_shift

    for end_idx in range(WINDOW_SIZE, last_index + 1):
        feature_window = df[feature_cols].iloc[end_idx - WINDOW_SIZE:end_idx].values.flatten()
        target_idx = end_idx - 1 + target_shift
        target_value = float(df["Predicted_MEHI"].iloc[target_idx])
        X.append(feature_window)
        y.append(target_value)

    return np.asarray(X), np.asarray(y)


def build_windowed_env_xy(df, feature_cols):
    X = []
    y = []

    for end_idx in range(WINDOW_SIZE, len(df)):
        feature_window = df[feature_cols].iloc[end_idx - WINDOW_SIZE:end_idx].values.flatten()
        next_feature_row = df[feature_cols].iloc[end_idx].values.astype(float)
        X.append(feature_window)
        y.append(next_feature_row)

    return np.asarray(X), np.asarray(y)


def train_and_evaluate(X, y, split_ratio=0.8):
    split_idx = max(int(len(X) * split_ratio), 10)
    split_idx = min(split_idx, len(X) - 1)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "pred_test_min": float(np.min(test_pred)),
        "pred_test_max": float(np.max(test_pred)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    return model, scaler, metrics


def train_environment_model(X, y, split_ratio=0.8):
    split_idx = max(int(len(X) * split_ratio), 10)
    split_idx = min(split_idx, len(X) - 1)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_r2": float(r2_score(y_train, train_pred, multioutput="uniform_average")),
        "test_r2": float(r2_score(y_test, test_pred, multioutput="uniform_average")),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    return model, scaler, metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "Predicted_MEHI"]).sort_values("time").reset_index(drop=True)

    feature_cols = pick_feature_columns(df)
    required = feature_cols + ["Predicted_MEHI"]
    df = df.dropna(subset=required).reset_index(drop=True)

    X_1m, y_1m = build_windowed_xy(df, feature_cols, horizon_months=1)
    model_1m, scaler_1m, metrics_1m = train_and_evaluate(X_1m, y_1m)

    X_3m, y_3m = build_windowed_xy(df, feature_cols, horizon_months=3)
    model_3m, scaler_3m, metrics_3m = train_and_evaluate(X_3m, y_3m)

    X_env, y_env = build_windowed_env_xy(df, feature_cols)
    env_model, env_scaler, env_metrics = train_environment_model(X_env, y_env)

    joblib.dump(model_1m, MODELS_DIR / "final_1month_model.pkl")
    joblib.dump(model_3m, MODELS_DIR / "final_3month_model.pkl")
    joblib.dump(scaler_1m, MODELS_DIR / "final_scaler_1month.pkl")
    joblib.dump(scaler_3m, MODELS_DIR / "final_scaler_3month.pkl")
    joblib.dump(env_model, MODELS_DIR / "env_1month_model.pkl")
    joblib.dump(env_scaler, MODELS_DIR / "env_scaler_1month.pkl")

    report = {
        "feature_columns": feature_cols,
        "window_size": WINDOW_SIZE,
        "model_type": "RandomForestRegressor",
        "one_month": metrics_1m,
        "three_month": metrics_3m,
        "environment_one_month": env_metrics,
    }

    with open(MODELS_DIR / "training_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("Training complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
