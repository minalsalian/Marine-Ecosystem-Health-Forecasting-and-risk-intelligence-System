from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = ROOT / "data" / "marine_10year_forecast_alerts.csv"
OUTPUT_DATA_2026 = ROOT / "data" / "processed" / "marine_full_2014_2026.csv"
OUTPUT_DATA_2025 = ROOT / "data" / "processed" / "marine_full_2014_2025.csv"

MODEL_PATH = ROOT / "models" / "final_1month_model.pkl"
SCALER_PATH = ROOT / "models" / "final_scaler_1month.pkl"
ENV_MODEL_PATH = ROOT / "models" / "env_1month_model.pkl"
ENV_SCALER_PATH = ROOT / "models" / "env_scaler_1month.pkl"

WINDOW_SIZE = 3
FEATURE_CANDIDATES = [
    ["chl", "o2", "thetao", "so"],
    ["chl", "o2", "sst", "salinity"],
]


def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    for candidate in FEATURE_CANDIDATES:
        if set(candidate).issubset(df.columns):
            return candidate
    raise ValueError("Input dataset is missing required feature columns for forecasting.")


def load_target_bounds(df: pd.DataFrame) -> tuple[float, float]:
    lower = float(df["Predicted_MEHI"].quantile(0.01))
    upper = float(df["Predicted_MEHI"].quantile(0.99))
    if lower >= upper:
        return (0.0, max(100.0, float(df["Predicted_MEHI"].max())))
    return lower, upper


def build_feature_bounds(reference_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    bounds = {}
    for field in feature_cols:
        low = float(reference_df[field].quantile(0.01))
        high = float(reference_df[field].quantile(0.99))
        pad = 0.1 * (high - low) if high > low else 1.0
        bounds[field] = (low - pad, high + pad)
    return bounds


def compute_monthly_climatology(dataframe: pd.DataFrame, column_name: str, month_value: int, fallback: float) -> float:
    monthly_values = dataframe[dataframe["time"].dt.month == month_value][column_name]
    if monthly_values.empty:
        return float(fallback)
    return float(monthly_values.mean())


def compute_z_score(dataframe: pd.DataFrame, column_name: str, raw_value: float) -> float:
    mean_value = float(dataframe[column_name].mean())
    std_value = float(dataframe[column_name].std())
    if not np.isfinite(std_value) or std_value == 0:
        return 0.0
    return float((raw_value - mean_value) / std_value)


def infer_alert_level(predicted_mehi: float, reference_df: pd.DataFrame) -> str:
    critical_data = reference_df[reference_df["Ecosystem_Alert"] == "CRITICAL"]["Predicted_MEHI"]
    warning_data = reference_df[reference_df["Ecosystem_Alert"] == "WARNING"]["Predicted_MEHI"]

    if not critical_data.empty and not warning_data.empty:
        critical_max = float(critical_data.max())
        warning_max = float(warning_data.max())
        if predicted_mehi <= critical_max:
            return "CRITICAL"
        if predicted_mehi <= warning_max:
            return "WARNING"
        return "STABLE"

    low_threshold = float(reference_df["Predicted_MEHI"].quantile(0.33))
    mid_threshold = float(reference_df["Predicted_MEHI"].quantile(0.66))
    if predicted_mehi <= low_threshold:
        return "CRITICAL"
    if predicted_mehi <= mid_threshold:
        return "WARNING"
    return "STABLE"


def build_next_row(
    working_df: pd.DataFrame,
    feature_cols: list[str],
    mehi_model,
    mehi_scaler,
    env_model,
    env_scaler,
    mehi_bounds: tuple[float, float],
    feature_bounds: dict[str, tuple[float, float]],
) -> pd.Series:
    latest = working_df.iloc[-1].copy()
    next_time = pd.to_datetime(latest["time"]) + pd.DateOffset(months=1)

    env_window = working_df[feature_cols].tail(WINDOW_SIZE).values.flatten().reshape(1, -1)
    env_window_scaled = env_scaler.transform(env_window)
    env_pred = np.asarray(env_model.predict(env_window_scaled)).reshape(-1)

    env_next = {}
    for idx, field in enumerate(feature_cols):
        low, high = feature_bounds[field]
        env_next[field] = float(np.clip(env_pred[idx], low, high))

    lag_window = working_df[feature_cols].tail(WINDOW_SIZE - 1).copy()
    candidate_feature_row = pd.DataFrame([env_next], columns=feature_cols)
    feature_window = pd.concat([lag_window, candidate_feature_row], ignore_index=True)

    X_input = feature_window[feature_cols].values.flatten().reshape(1, -1)
    X_model = mehi_scaler.transform(X_input)
    raw_pred_mehi = float(mehi_model.predict(X_model)[0])
    pred_mehi = float(np.clip(raw_pred_mehi, mehi_bounds[0], mehi_bounds[1]))

    next_row = latest.copy()
    next_row["time"] = next_time
    for field in feature_cols:
        next_row[field] = env_next[field]

    if "sst" not in next_row.index and "thetao" in next_row.index:
        next_row["sst"] = float(next_row["thetao"])
    if "salinity" not in next_row.index and "so" in next_row.index:
        next_row["salinity"] = float(next_row["so"])

    if "thetao" in next_row.index:
        next_row["thetao"] = float(next_row["sst"])
    if "so" in next_row.index:
        next_row["so"] = float(next_row["salinity"])

    next_row["month"] = int(next_time.month)
    next_row["sst_climatology"] = compute_monthly_climatology(working_df, "sst", next_row["month"], next_row["sst"])
    next_row["chl_climatology"] = compute_monthly_climatology(working_df, "chl", next_row["month"], next_row["chl"])
    next_row["sst_anomaly"] = float(next_row["sst"] - next_row["sst_climatology"])
    next_row["chl_anomaly"] = float(next_row["chl"] - next_row["chl_climatology"])
    next_row["sst_z"] = compute_z_score(working_df, "sst", float(next_row["sst"]))
    next_row["chl_z"] = compute_z_score(working_df, "chl", float(next_row["chl"]))
    next_row["o2_z"] = compute_z_score(working_df, "o2", float(next_row["o2"]))

    next_row["Predicted_MEHI"] = pred_mehi
    next_row["Ecosystem_Alert"] = infer_alert_level(pred_mehi, working_df)

    if pred_mehi >= 60:
        next_row["Ecosystem_Status"] = "Good"
    elif pred_mehi >= 45:
        next_row["Ecosystem_Status"] = "Moderate"
    else:
        next_row["Ecosystem_Status"] = "Poor"

    next_row["Bleaching_Risk"] = "HIGH" if float(next_row["sst_z"]) >= 1.0 else "NORMAL"
    next_row["Bloom_Risk"] = "HIGH" if float(next_row["chl_z"]) >= 1.0 else "NORMAL"
    next_row["Oxygen_Risk"] = "HIGH" if float(next_row["o2_z"]) <= -1.0 else "NORMAL"

    if "MS_MHHI_raw" in next_row.index:
        next_row["MS_MHHI_raw"] = pred_mehi
    if "MS_MHHI" in next_row.index:
        next_row["MS_MHHI"] = pred_mehi

    next_row["data_source"] = "forecast"
    return next_row


def main() -> None:
    df = pd.read_csv(INPUT_DATA)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    feature_cols = pick_feature_columns(df)

    if "Predicted_MEHI" not in df.columns:
        raise ValueError("Input dataset is missing 'Predicted_MEHI'.")

    if "data_source" not in df.columns:
        df["data_source"] = "actual"
    else:
        df["data_source"] = df["data_source"].fillna("actual")

    mehi_model = joblib.load(MODEL_PATH)
    mehi_scaler = joblib.load(SCALER_PATH)
    env_model = joblib.load(ENV_MODEL_PATH)
    env_scaler = joblib.load(ENV_SCALER_PATH)
    mehi_bounds = load_target_bounds(df)
    feature_bounds = build_feature_bounds(df, feature_cols)

    last_real_time = pd.to_datetime(df["time"]).max()
    target_end = pd.Timestamp("2026-12-01")
    if last_real_time >= target_end:
        OUTPUT_DATA_2026.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DATA_2026, index=False)
        df.to_csv(OUTPUT_DATA_2025, index=False)
        print(f"No generation needed: last record is {last_real_time.date()}")
        print(f"Saved existing dataset to: {OUTPUT_DATA_2026}")
        return

    months_to_generate = (target_end.year - last_real_time.year) * 12 + (target_end.month - last_real_time.month)
    if months_to_generate <= 0:
        months_to_generate = 0

    generated_rows = []
    working_df = df.copy()
    for _ in range(months_to_generate):
        next_row = build_next_row(
            working_df,
            feature_cols,
            mehi_model,
            mehi_scaler,
            env_model,
            env_scaler,
            mehi_bounds,
            feature_bounds,
        )
        generated_rows.append(next_row)
        working_df = pd.concat([working_df, pd.DataFrame([next_row])], ignore_index=True)

    future_df = pd.DataFrame(generated_rows)
    final_df = pd.concat([df, future_df], ignore_index=True)
    final_df = final_df.sort_values("time").reset_index(drop=True)

    OUTPUT_DATA_2026.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_DATA_2026, index=False)
    final_df.to_csv(OUTPUT_DATA_2025, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Generated rows: {len(future_df)}")
    print(f"Final rows: {len(final_df)}")
    print(f"Final time range: {final_df['time'].min().date()} -> {final_df['time'].max().date()}")
    print(f"Saved to: {OUTPUT_DATA_2026}")


if __name__ == "__main__":
    main()