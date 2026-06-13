from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.exceptions import HTTPException
import csv
from datetime import datetime
from io import StringIO
import pandas as pd
import joblib
import os
import numpy as np
import logging
from threading import Lock

try:
    import shap
except Exception:
    shap = None

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marine-dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

FORECAST_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "marine_full_2014_2026.csv")
FORECAST_DATA_PATH_OLD = os.path.join(BASE_DIR, "..", "data", "processed", "marine_full_2014_2025.csv")
LEGACY_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "marine_10year_forecast_alerts.csv")
if os.path.exists(FORECAST_DATA_PATH):
    DATA_PATH = FORECAST_DATA_PATH
elif os.path.exists(FORECAST_DATA_PATH_OLD):
    DATA_PATH = FORECAST_DATA_PATH_OLD
else:
    DATA_PATH = LEGACY_DATA_PATH

FINAL_MODEL_1M_PATH = os.path.join(MODELS_DIR, "final_1month_model.pkl")
LEGACY_MODEL_1M_PATH = os.path.join(MODELS_DIR, "linear_1month_model.pkl")

FINAL_MODEL_3M_PATH = os.path.join(MODELS_DIR, "final_3month_model.pkl")
LEGACY_MODEL_3M_PATH = os.path.join(MODELS_DIR, "linear_3month_model.pkl")

FINAL_SCALER_1M_PATH = os.path.join(MODELS_DIR, "final_scaler_1month.pkl")
LEGACY_SCALER_1M_PATH = os.path.join(MODELS_DIR, "scaler_X.pkl")

FINAL_SCALER_3M_PATH = os.path.join(MODELS_DIR, "final_scaler_3month.pkl")
LEGACY_SCALER_3M_PATH = os.path.join(MODELS_DIR, "scaler_X_3month.pkl")

WINDOW_SIZE = 3
FEATURE_COLS_PRIMARY = ["chl", "o2", "thetao", "so"]
FEATURE_COLS_FALLBACK = ["chl", "o2", "sst", "salinity"]
TARGET_OUTPUT_INDEX = 0
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")

SPECIES_METADATA = [
    {"key": "turtle", "label": "🐢 Turtle", "r2": 0.68, "reliability": "High"},
    {"key": "fish", "label": "🐟 Fish", "r2": 0.56, "reliability": "Moderate"},
    {"key": "plant", "label": "🌿 Plant", "r2": 0.78, "reliability": "Very High"},
]

SPECIES_CALIBRATION = None
DATA_LOCK = Lock()


def _safe_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for '{field_name}'") from exc


def _compute_monthly_climatology(dataframe, column_name, month_value, fallback_value):
    if column_name not in dataframe.columns:
        return fallback_value

    monthly_series = dataframe[dataframe["time"].dt.month == month_value][column_name]
    if monthly_series.empty:
        return fallback_value
    return float(monthly_series.mean())


def _compute_z_score(dataframe, column_name, raw_value):
    if column_name not in dataframe.columns:
        return 0.0

    mean_value = float(dataframe[column_name].mean())
    std_value = float(dataframe[column_name].std())
    if not np.isfinite(std_value) or std_value == 0:
        return 0.0
    return float((raw_value - mean_value) / std_value)


def build_live_ingest_row(payload):
    working_df = df.copy().sort_values("time")
    latest = working_df.iloc[-1]

    if payload.get("time"):
        timestamp = pd.to_datetime(payload.get("time"), errors="coerce")
        if pd.isna(timestamp):
            raise ValueError("Invalid 'time' format. Use ISO date format.")
    else:
        timestamp = pd.to_datetime(latest["time"]) + pd.DateOffset(months=1)

    sst_value = _safe_float(payload.get("sst"), "sst")
    chl_value = _safe_float(payload.get("chl"), "chl")
    o2_value = _safe_float(payload.get("o2"), "o2")
    salinity_value = _safe_float(payload.get("salinity"), "salinity")

    thetao_value = _safe_float(payload.get("thetao", sst_value), "thetao")
    so_value = _safe_float(payload.get("so", salinity_value), "so")

    month_value = int(timestamp.month)

    sst_climatology = _compute_monthly_climatology(working_df, "sst", month_value, sst_value)
    chl_climatology = _compute_monthly_climatology(working_df, "chl", month_value, chl_value)

    sst_anomaly = float(sst_value - sst_climatology)
    chl_anomaly = float(chl_value - chl_climatology)

    sst_z = _compute_z_score(working_df, "sst", sst_value)
    chl_z = _compute_z_score(working_df, "chl", chl_value)
    o2_z = _compute_z_score(working_df, "o2", o2_value)

    candidate = latest.to_dict()
    candidate.update(
        {
            "time": timestamp,
            "sst": sst_value,
            "chl": chl_value,
            "o2": o2_value,
            "salinity": salinity_value,
            "thetao": thetao_value,
            "so": so_value,
            "month": month_value,
            "sst_climatology": sst_climatology,
            "chl_climatology": chl_climatology,
            "sst_anomaly": sst_anomaly,
            "chl_anomaly": chl_anomaly,
            "sst_z": sst_z,
            "chl_z": chl_z,
            "o2_z": o2_z,
        }
    )

    feature_cols = get_feature_columns(working_df)
    feature_window = working_df[feature_cols].tail(WINDOW_SIZE - 1).copy()
    new_feature_row = {col: float(candidate.get(col, 0.0)) for col in feature_cols}
    feature_window = pd.concat([feature_window, pd.DataFrame([new_feature_row])], ignore_index=True)

    X_input = feature_window.values.flatten().reshape(1, -1)
    model_output = predict_output(X_input, horizon=1)
    predicted_mehi = extract_scalar_prediction(model_output, TARGET_OUTPUT_INDEX)

    candidate["Predicted_MEHI"] = float(predicted_mehi)
    candidate["Ecosystem_Alert"] = infer_alert_level(predicted_mehi)

    if predicted_mehi >= 60:
        candidate["Ecosystem_Status"] = "Good"
    elif predicted_mehi >= 45:
        candidate["Ecosystem_Status"] = "Moderate"
    else:
        candidate["Ecosystem_Status"] = "Poor"

    candidate["Bleaching_Risk"] = "HIGH" if sst_z >= 1.0 else "NORMAL"
    candidate["Bloom_Risk"] = "HIGH" if chl_z >= 1.0 else "NORMAL"
    candidate["Oxygen_Risk"] = "HIGH" if o2_z <= -1.0 else "NORMAL"

    if "MS_MHHI_raw" in candidate:
        candidate["MS_MHHI_raw"] = float(predicted_mehi)
    if "MS_MHHI" in candidate:
        candidate["MS_MHHI"] = float(predicted_mehi)

    return candidate


def ingest_live_record(payload):
    global df, SPECIES_CALIBRATION

    with DATA_LOCK:
        new_row = build_live_ingest_row(payload)
        appended_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        cleaned_df = clean_dataset(appended_df)

        cleaned_df.to_csv(DATA_PATH, index=False)

        df = cleaned_df
        SPECIES_CALIBRATION = None

    return new_row


def get_feature_columns(dataframe):
    if set(FEATURE_COLS_PRIMARY).issubset(dataframe.columns):
        return FEATURE_COLS_PRIMARY
    if set(FEATURE_COLS_FALLBACK).issubset(dataframe.columns):
        return FEATURE_COLS_FALLBACK
    raise ValueError(
        "Missing required feature columns. Expected either "
        f"{FEATURE_COLS_PRIMARY} or {FEATURE_COLS_FALLBACK}."
    )


def load_pickle_from_candidates(paths, label, required=False):
    for path in paths:
        if os.path.exists(path):
            try:
                logger.info("Loading %s from %s", label, path)
                return joblib.load(path)
            except Exception as exc:
                logger.warning("Failed loading %s from %s: %s", label, path, exc)

    if required:
        raise RuntimeError(f"Unable to load required artifact: {label}")

    logger.info("Optional artifact missing: %s", label)
    return None


def save_artifact_if_possible(artifact, output_path, label):
    if artifact is None:
        return

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(artifact, output_path)
        logger.info("Saved %s to %s", label, output_path)
    except Exception as exc:
        logger.warning("Could not save %s to %s: %s", label, output_path, exc)


def clean_dataset(raw_df):
    cleaned_df = raw_df.copy()
    cleaned_df.columns = [str(col).strip() for col in cleaned_df.columns]

    if "time" not in cleaned_df.columns:
        raise ValueError("Dataset is missing 'time' column.")
    if "Predicted_MEHI" not in cleaned_df.columns:
        raise ValueError("Dataset is missing 'Predicted_MEHI' column.")
    if "Ecosystem_Alert" not in cleaned_df.columns:
        raise ValueError("Dataset is missing 'Ecosystem_Alert' column.")

    cleaned_df["time"] = pd.to_datetime(cleaned_df["time"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["time"]).copy()

    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    if numeric_columns:
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].ffill().bfill()

    feature_cols = get_feature_columns(cleaned_df)
    required_subset = ["time", "Predicted_MEHI", "Ecosystem_Alert"] + feature_cols
    cleaned_df = cleaned_df.dropna(subset=required_subset)

    if cleaned_df.empty:
        raise ValueError("Dataset became empty after cleaning NaN values.")

    cleaned_df = cleaned_df.sort_values("time").reset_index(drop=True)
    return cleaned_df


def safe_transform(X_input, scaler):
    if scaler is None:
        return X_input

    try:
        return scaler.transform(X_input)
    except Exception as exc:
        logger.warning("Scaler transform failed, using raw features: %s", exc)
        return X_input


def extract_scalar_prediction(prediction, target_index=0):
    prediction_array = np.asarray(prediction)

    if prediction_array.ndim == 0:
        return float(prediction_array)
    if prediction_array.ndim == 1:
        safe_index = min(target_index, prediction_array.shape[0] - 1)
        return float(prediction_array[safe_index])

    safe_index = min(target_index, prediction_array.shape[1] - 1)
    return float(prediction_array[0, safe_index])


def score_to_status(score):
    if score >= 70:
        return "Stable"
    if score >= 45:
        return "Warning"
    return "Critical"


def score_to_status_class(score):
    if score >= 70:
        return "status-stable"
    if score >= 45:
        return "status-warning"
    return "status-critical"


def get_feature_labels(feature_cols):
    labels = []
    for row_idx in range(WINDOW_SIZE):
        lag = WINDOW_SIZE - 1 - row_idx
        lag_label = "t" if lag == 0 else f"t-{lag}"
        for feature in feature_cols:
            labels.append(f"{feature} ({lag_label})")
    return labels


def get_model_coefficients():
    if hasattr(model_1m, "coef_"):
        coefficients = np.asarray(model_1m.coef_)
        if coefficients.ndim == 1:
            return coefficients
        safe_index = min(TARGET_OUTPUT_INDEX, coefficients.shape[0] - 1)
        return coefficients[safe_index]

    if hasattr(model_1m, "feature_importances_"):
        return np.asarray(model_1m.feature_importances_, dtype=float)

    expected_width = int(getattr(model_1m, "n_features_in_", WINDOW_SIZE * len(FEATURE_COLS_FALLBACK)))
    return np.ones(expected_width, dtype=float)


def get_environment_input_snapshot(working_df):
    latest = working_df.iloc[-1]

    salinity_mean = working_df["salinity"].mean() if "salinity" in working_df.columns else 0
    salinity_std = working_df["salinity"].std() if "salinity" in working_df.columns else 1
    salinity_std = salinity_std if pd.notna(salinity_std) and salinity_std != 0 else 1

    salinity_z = (
        (latest["salinity"] - salinity_mean) / salinity_std
        if "salinity" in latest.index
        else 0
    )

    return {
        "sst_z": round(float(latest.get("sst_z", 0)), 3),
        "chl_z": round(float(latest.get("chl_z", 0)), 3),
        "o2_z": round(float(latest.get("o2_z", 0)), 3),
        "salinity_z": round(float(salinity_z), 3),
    }


def risk_score_to_level(score):
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MODERATE"
    return "LOW"


def wqi_score_to_status(score):
    if score >= 75:
        return "GOOD"
    if score >= 50:
        return "MODERATE"
    return "POOR"


def compute_environmental_indicators_from_z(sst_z, chl_z, o2_z, salinity_z):
    bloom_risk = float(np.clip(48 + (chl_z * 22) + (sst_z * 7) + (max(-o2_z, 0) * 4), 0, 100))
    hypoxia_risk = float(np.clip(42 + (max(-o2_z, 0) * 30) + (sst_z * 6) + (chl_z * 5), 0, 100))
    salinity_stress = float(np.clip(abs(salinity_z) * 20, 0, 100))
    water_quality_index = float(np.clip(100 - ((0.42 * bloom_risk) + (0.43 * hypoxia_risk) + (0.15 * salinity_stress)), 0, 100))

    return {
        "bloom_risk": round(bloom_risk, 1),
        "bloom_status": risk_score_to_level(bloom_risk),
        "hypoxia_risk": round(hypoxia_risk, 1),
        "hypoxia_status": risk_score_to_level(hypoxia_risk),
        "water_quality_index": round(water_quality_index, 1),
        "wqi_status": wqi_score_to_status(water_quality_index),
    }


def compute_wqi_trend(working_df):
    salinity_mean = working_df["salinity"].mean() if "salinity" in working_df.columns else 0
    salinity_std = working_df["salinity"].std() if "salinity" in working_df.columns else 1
    salinity_std = salinity_std if pd.notna(salinity_std) and salinity_std != 0 else 1

    trend_labels = []
    trend_values = []

    for _, row in working_df.iterrows():
        sst_z = float(row.get("sst_z", 0))
        chl_z = float(row.get("chl_z", 0))
        o2_z = float(row.get("o2_z", 0))
        salinity_value = float(row.get("salinity", salinity_mean))
        salinity_z = float((salinity_value - salinity_mean) / salinity_std)

        indicators = compute_environmental_indicators_from_z(sst_z, chl_z, o2_z, salinity_z)
        trend_labels.append(pd.to_datetime(row["time"]).strftime("%b %Y"))
        trend_values.append(indicators["water_quality_index"])

    return trend_labels, trend_values


def build_trend_meta(current_value, previous_value, precision=2):
    current_float = float(current_value)
    previous_float = float(previous_value)
    delta = current_float - previous_float

    if delta > 0.02:
        arrow = "↑"
        direction = "up"
    elif delta < -0.02:
        arrow = "↓"
        direction = "down"
    else:
        arrow = "→"
        direction = "flat"

    return {
        "value": round(current_float, precision),
        "delta": round(delta, precision),
        "arrow": arrow,
        "direction": direction,
    }


def build_live_parameter_cards(working_df):
    latest = working_df.iloc[-1]
    previous = working_df.iloc[-2] if len(working_df) > 1 else latest

    return {
        "sst": {
            **build_trend_meta(latest.get("sst", 0.0), previous.get("sst", 0.0), precision=2),
            "label": "SST",
            "unit": "deg C",
            "icon": "🌡",
        },
        "chlorophyll": {
            **build_trend_meta(latest.get("chl", 0.0), previous.get("chl", 0.0), precision=3),
            "label": "Chlorophyll",
            "unit": "mg/m3",
            "icon": "🟢",
        },
        "oxygen": {
            **build_trend_meta(latest.get("o2", 0.0), previous.get("o2", 0.0), precision=2),
            "label": "Oxygen",
            "unit": "mg/L",
            "icon": "💧",
        },
        "salinity": {
            **build_trend_meta(latest.get("salinity", 0.0), previous.get("salinity", 0.0), precision=2),
            "label": "Salinity",
            "unit": "PSU",
            "icon": "🌊",
        },
    }


def risk_level_to_ui(level_text):
    normalized = str(level_text).upper()
    if normalized in ("HIGH", "CRITICAL"):
        return "Critical", "risk-critical"
    if normalized in ("MODERATE", "WARNING"):
        return "Warning", "risk-warning"
    return "Stable", "risk-stable"


def build_live_risk_alerts(latest_alert, ecosystem_indicators):
    oxygen_level = ecosystem_indicators.get("hypoxia_status", "LOW")
    bloom_level = ecosystem_indicators.get("bloom_status", "LOW")

    oxygen_state, oxygen_class = risk_level_to_ui(oxygen_level)
    bloom_state, bloom_class = risk_level_to_ui(bloom_level)
    ecosystem_state, ecosystem_class = risk_level_to_ui(latest_alert)

    return [
        {
            "title": "Oxygen Depletion Risk",
            "level": str(oxygen_level).upper(),
            "state": oxygen_state,
            "state_class": oxygen_class,
        },
        {
            "title": "Bloom Risk",
            "level": str(bloom_level).upper(),
            "state": bloom_state,
            "state_class": bloom_class,
        },
        {
            "title": "Overall Ecosystem Alert",
            "level": str(latest_alert).upper(),
            "state": ecosystem_state,
            "state_class": ecosystem_class,
        },
    ]


def build_notification_feed(latest_alert, ecosystem_indicators, live_parameters):
    notifications = []
    latest_alert = str(latest_alert).upper()
    bloom_level = str(ecosystem_indicators.get("bloom_status", "LOW")).upper()
    hypoxia_level = str(ecosystem_indicators.get("hypoxia_status", "LOW")).upper()

    if latest_alert in ("WARNING", "CRITICAL"):
        notifications.append(
            {
                "title": "Ecosystem alert crossed threshold",
                "level": latest_alert,
                "message": f"Current ecosystem alert is {latest_alert}. Review the forecast before conditions worsen.",
                "badge_class": "notification-critical" if latest_alert == "CRITICAL" else "notification-warning",
                "timestamp": "Current",
                "source": "MEHI forecast",
            }
        )

    if bloom_level in ("MODERATE", "HIGH"):
        notifications.append(
            {
                "title": "Bloom risk elevated",
                "level": bloom_level,
                "message": f"Bloom risk is {bloom_level} at {live_parameters['chlorophyll']['value']} {live_parameters['chlorophyll']['unit']}.",
                "badge_class": "notification-warning" if bloom_level == "MODERATE" else "notification-critical",
                "timestamp": "Current",
                "source": "Chlorophyll monitor",
            }
        )

    if hypoxia_level in ("MODERATE", "HIGH"):
        notifications.append(
            {
                "title": "Oxygen depletion risk elevated",
                "level": hypoxia_level,
                "message": f"Hypoxia risk is {hypoxia_level} at {live_parameters['oxygen']['value']} {live_parameters['oxygen']['unit']}.",
                "badge_class": "notification-warning" if hypoxia_level == "MODERATE" else "notification-critical",
                "timestamp": "Current",
                "source": "Oxygen monitor",
            }
        )

    if not notifications:
        notifications.append(
            {
                "title": "System stable",
                "level": "LOW",
                "message": "No threshold crossings detected. Monitoring continues in the background.",
                "badge_class": "notification-stable",
                "timestamp": "Current",
                "source": "Monitoring system",
            }
        )

    return notifications


def build_live_log_rows(working_df, limit=8):
    latest_rows = working_df.tail(limit)
    records = []
    for _, row in latest_rows.iterrows():
        records.append(
            {
                "time": pd.to_datetime(row["time"]).strftime("%Y-%m-%d"),
                "sst": round(float(row.get("sst", 0.0)), 2),
                "o2": round(float(row.get("o2", 0.0)), 2),
                "risk": str(row.get("Ecosystem_Alert", "UNKNOWN")),
            }
        )
    return records


def predict_output(X_input, horizon):
    if horizon == 3:
        selected_model = model_3m
        selected_scaler = scaler_3m
    else:
        selected_model = model_1m
        selected_scaler = scaler_1m

    expected_width = getattr(selected_model, "n_features_in_", X_input.shape[1])
    if expected_width != X_input.shape[1]:
        selected_model = model_1m
        selected_scaler = scaler_1m

    X_model = safe_transform(X_input, selected_scaler)
    return np.asarray(selected_model.predict(X_model))


def infer_alert_level(predicted_mehi):
    critical_data = df[df["Ecosystem_Alert"] == "CRITICAL"]["Predicted_MEHI"]
    warning_data = df[df["Ecosystem_Alert"] == "WARNING"]["Predicted_MEHI"]

    if not critical_data.empty and not warning_data.empty:
        critical_max = critical_data.max()
        warning_max = warning_data.max()

        if predicted_mehi <= critical_max:
            return "CRITICAL"
        if predicted_mehi <= warning_max:
            return "WARNING"
        return "STABLE"

    low_threshold = df["Predicted_MEHI"].quantile(0.33)
    mid_threshold = df["Predicted_MEHI"].quantile(0.66)

    if predicted_mehi <= low_threshold:
        return "CRITICAL"
    if predicted_mehi <= mid_threshold:
        return "WARNING"
    return "STABLE"


def compute_species_calibration():
    working_df = df.copy()
    feature_cols = get_feature_columns(working_df)
    prediction_rows = []
    salinity_mean = float(working_df["salinity"].mean()) if "salinity" in working_df.columns else 0.0
    salinity_std = float(working_df["salinity"].std()) if "salinity" in working_df.columns else 1.0
    if not np.isfinite(salinity_std) or salinity_std == 0:
        salinity_std = 1.0

    for idx in range(WINDOW_SIZE, len(working_df) + 1):
        feature_window = working_df[feature_cols].iloc[idx - WINDOW_SIZE:idx]
        X_input = feature_window.values.flatten().reshape(1, -1)
        model_output = predict_output(X_input, horizon=1)

        env_row = working_df.iloc[idx - 1]
        salinity_z = float((float(env_row.get("salinity", salinity_mean)) - salinity_mean) / salinity_std)
        species_outputs = extract_species_outputs(
            model_output,
            {
                "sst_z": float(env_row.get("sst_z", 0.0)),
                "chl_z": float(env_row.get("chl_z", 0.0)),
                "o2_z": float(env_row.get("o2_z", 0.0)),
                "salinity_z": salinity_z,
            },
        )

        prediction_rows.append(species_outputs)

    if not prediction_rows:
        return {"mins": np.zeros(3), "maxs": np.ones(3), "spans": np.ones(3)}

    prediction_matrix = np.vstack(prediction_rows)
    mins = np.percentile(prediction_matrix, 5, axis=0)
    maxs = np.percentile(prediction_matrix, 95, axis=0)
    spans = np.where((maxs - mins) == 0, 1.0, (maxs - mins))

    return {"mins": mins, "maxs": maxs, "spans": spans}


def build_species_prediction_payload(raw_species_outputs, already_scaled=False):
    global SPECIES_CALIBRATION

    if already_scaled:
        normalized_scores = np.clip(np.asarray(raw_species_outputs, dtype=float), 0, 100)
    else:
        if SPECIES_CALIBRATION is None:
            SPECIES_CALIBRATION = compute_species_calibration()

        normalized_scores = (
            (raw_species_outputs - SPECIES_CALIBRATION["mins"]) / SPECIES_CALIBRATION["spans"]
        ) * 100
        normalized_scores = np.clip(normalized_scores, 0, 100)

    species_cards = []
    for idx, meta in enumerate(SPECIES_METADATA):
        score = round(float(normalized_scores[idx]), 1)
        species_cards.append(
            {
                "label": meta["label"],
                "score": score,
                "status": score_to_status(score),
                "status_class": score_to_status_class(score),
                "reliability": meta["reliability"],
                "r2": meta["r2"],
            }
        )

    avg_score = float(np.mean(normalized_scores))
    at_risk_count = int(sum(1 for card in species_cards if card["status"] != "Stable"))

    if avg_score >= 70:
        overall_condition = "GOOD"
    elif avg_score >= 45:
        overall_condition = "MODERATE"
    else:
        overall_condition = "CRITICAL"

    return {
        "species_cards": species_cards,
        "overall_condition": overall_condition,
        "at_risk_count": at_risk_count,
        "average_score": round(avg_score, 1),
        "species_labels": [card["label"] for card in species_cards],
        "species_scores": [card["score"] for card in species_cards],
    }


def extract_species_outputs(model_output, env_context=None):
    output_array = np.asarray(model_output).reshape(-1)

    if output_array.size >= 3:
        return output_array[:3].astype(float)

    base_value = float(output_array[0]) if output_array.size else 0.0
    env_context = env_context or {}

    sst_z = float(env_context.get("sst_z", 0.0))
    chl_z = float(env_context.get("chl_z", 0.0))
    o2_z = float(env_context.get("o2_z", 0.0))
    salinity_z = float(env_context.get("salinity_z", 0.0))

    # Keep species differentiated when a legacy single-output model is loaded.
    salinity_stress = abs(salinity_z)
    turtle_score = base_value + (0.85 * sst_z) + (0.28 * chl_z) - (1.10 * o2_z) - (0.20 * salinity_stress)
    fish_score = base_value + (0.35 * sst_z) + (0.22 * chl_z) - (1.65 * o2_z) - (0.35 * salinity_stress)
    plant_score = base_value - (0.20 * sst_z) + (1.55 * chl_z) - (0.35 * o2_z) - (0.10 * salinity_stress)

    return np.array([turtle_score, fish_score, plant_score], dtype=float)

def compute_species_health_from_raw(sst, chl, o2, salinity):
    def clamp_01(value):
        return float(np.clip(float(value), 0.0, 1.0))

    sst = float(sst)
    chl = float(chl)
    o2 = float(o2)
    salinity = float(salinity)

    # Normalize environmental inputs to 0-1 using realistic operating ranges.
    sst_norm = clamp_01((sst - 25.0) / (32.0 - 25.0))
    oxygen_norm = clamp_01(o2 / 7.0)
    chl_norm = clamp_01(chl / 2.0)
    sal_norm = clamp_01(1.0 - (abs(salinity - 35.0) / 5.0))

    # Weighted environmental health indices.
    turtle_health = (
        (0.35 * oxygen_norm)
        + (0.30 * sst_norm)
        + (0.20 * chl_norm)
        + (0.15 * sal_norm)
    )
    fish_health = (
        (0.50 * oxygen_norm)
        + (0.20 * sst_norm)
        + (0.20 * chl_norm)
        + (0.10 * sal_norm)
    )
    plant_health = (
        (0.50 * chl_norm)
        + (0.20 * oxygen_norm)
        + (0.20 * sst_norm)
        + (0.10 * sal_norm)
    )

    # Convert health to risk score, where higher value means higher risk.
    turtle_risk = 100.0 - (turtle_health * 100.0)
    fish_risk = 100.0 - (fish_health * 100.0)
    plant_risk = 100.0 - (plant_health * 100.0)

    # Slight sensitivity boost while keeping deterministic behavior across refreshes.
    turtle_risk *= 1.10
    fish_risk *= 1.15
    plant_risk *= 1.05

    turtle_risk = np.clip(turtle_risk, 0.0, 100.0)
    fish_risk = np.clip(fish_risk, 0.0, 100.0)
    plant_risk = np.clip(plant_risk, 0.0, 100.0)

    return np.array([turtle_risk, fish_risk, plant_risk], dtype=float)

def z_to_raw(column_name, z_value, working_df):
    if column_name not in working_df.columns:
        return 0.0

    mean_value = float(working_df[column_name].mean())
    std_value = float(working_df[column_name].std())
    if not np.isfinite(std_value) or std_value == 0:
        std_value = 1.0

    return float(mean_value + (z_value * std_value))


def predict_mehi_from_raw_inputs(sst, chl, o2, salinity):
    working_df = df.copy().sort_values("time")
    feature_cols = get_feature_columns(working_df)
    latest_row = working_df.iloc[-1]

    scenario_row = {}
    for col in feature_cols:
        if col == "chl":
            scenario_row[col] = float(chl)
        elif col == "o2":
            scenario_row[col] = float(o2)
        elif col in ("sst", "thetao"):
            scenario_row[col] = float(sst)
        elif col in ("salinity", "so"):
            scenario_row[col] = float(salinity)
        else:
            scenario_row[col] = float(latest_row.get(col, 0.0))

    feature_window = pd.DataFrame([scenario_row] * WINDOW_SIZE, columns=feature_cols)
    X_input = feature_window.values.flatten().reshape(1, -1)
    model_output = predict_output(X_input, horizon=1)
    return extract_scalar_prediction(model_output, TARGET_OUTPUT_INDEX)


def compute_z_inputs_from_raw(sst, chl, o2, salinity):
    working_df = df.copy().sort_values("time")
    return {
        "sst_z": _compute_z_score(working_df, "sst", sst),
        "chl_z": _compute_z_score(working_df, "chl", chl),
        "o2_z": _compute_z_score(working_df, "o2", o2),
        "salinity_z": _compute_z_score(working_df, "salinity", salinity),
    }


def predict_multi_species_from_z_inputs(sst_z, chl_z, o2_z, salinity_z):
    working_df = df.copy().sort_values("time")
    feature_cols = get_feature_columns(working_df)

    scenario_values = {
        "chl": z_to_raw("chl", chl_z, working_df),
        "o2": z_to_raw("o2", o2_z, working_df),
        "sst": z_to_raw("sst", sst_z, working_df),
        "thetao": z_to_raw("thetao", sst_z, working_df),
        "salinity": z_to_raw("salinity", salinity_z, working_df),
        "so": z_to_raw("so", salinity_z, working_df),
    }

    scenario_row = {}
    for col in feature_cols:
        if col in scenario_values and scenario_values[col] != 0:
            scenario_row[col] = scenario_values[col]
        elif col in working_df.columns:
            scenario_row[col] = float(working_df[col].iloc[-1])
        else:
            scenario_row[col] = 0.0

    feature_window = pd.DataFrame([scenario_row] * WINDOW_SIZE, columns=feature_cols)
    X_input = feature_window.values.flatten().reshape(1, -1)
    _ = predict_output(X_input, horizon=1)

    species_scores = compute_species_health_from_raw(
        sst=float(scenario_values.get("sst", scenario_values.get("thetao", 27.0))),
        chl=float(scenario_values.get("chl", 0.8)),
        o2=float(scenario_values.get("o2", 5.0)),
        salinity=float(scenario_values.get("salinity", scenario_values.get("so", 35.0))),
    )
    return build_species_prediction_payload(species_scores, already_scaled=True)


def build_scenario_payload(sst, chl, o2, salinity):
    working_df = df.copy().sort_values("time")
    latest_row = working_df.iloc[-1]

    baseline_mehi = float(latest_row["Predicted_MEHI"])
    scenario_mehi = float(predict_mehi_from_raw_inputs(sst, chl, o2, salinity))
    mehi_change = scenario_mehi - baseline_mehi

    z_inputs = compute_z_inputs_from_raw(sst, chl, o2, salinity)
    ecosystem_indicators = compute_environmental_indicators_from_z(
        z_inputs["sst_z"],
        z_inputs["chl_z"],
        z_inputs["o2_z"],
        z_inputs["salinity_z"],
    )
    species_scores = compute_species_health_from_raw(sst, chl, o2, salinity)
    species_payload = build_species_prediction_payload(species_scores, already_scaled=True)

    if mehi_change > 0:
        change_direction = "increase"
    elif mehi_change < 0:
        change_direction = "decrease"
    else:
        change_direction = "flat"

    return {
        "baseline_mehi": round(baseline_mehi, 2),
        "scenario_mehi": round(scenario_mehi, 2),
        "mehi_change": round(mehi_change, 2),
        "change_direction": change_direction,
        "current_alert": str(latest_row["Ecosystem_Alert"]),
        "scenario_alert": infer_alert_level(scenario_mehi),
        "scenario_inputs": {
            "sst": round(float(sst), 2),
            "chl": round(float(chl), 3),
            "o2": round(float(o2), 2),
            "salinity": round(float(salinity), 2),
        },
        "env_inputs": {key: round(float(value), 3) for key, value in z_inputs.items()},
        "ecosystem_indicators": ecosystem_indicators,
        "ecosystem_prediction": species_payload,
    }


def predict_multi_species_risk():
    working_df = df.copy().sort_values("time")
    feature_cols = get_feature_columns(working_df)
    feature_window = working_df[feature_cols].tail(WINDOW_SIZE)
    latest_row = working_df.iloc[-1]
    X_input = feature_window.values.flatten().reshape(1, -1)
    _ = predict_output(X_input, horizon=1)

    species_scores = compute_species_health_from_raw(
        sst=float(latest_row.get("sst", latest_row.get("thetao", 27.0))),
        chl=float(latest_row.get("chl", 0.8)),
        o2=float(latest_row.get("o2", 5.0)),
        salinity=float(latest_row.get("salinity", latest_row.get("so", 35.0))),
    )

    return build_species_prediction_payload(species_scores, already_scaled=True)


def compute_species_current_vs_predicted():
    working_df = df.copy().sort_values("time")
    feature_cols = get_feature_columns(working_df)
    feature_window = working_df[feature_cols].tail(WINDOW_SIZE)
    latest_row = working_df.iloc[-1]

    current_scores = compute_species_health_from_raw(
        sst=float(latest_row.get("sst", latest_row.get("thetao", 27.0))),
        chl=float(latest_row.get("chl", 0.8)),
        o2=float(latest_row.get("o2", 5.0)),
        salinity=float(latest_row.get("salinity", latest_row.get("so", 35.0))),
    )
    current_payload = build_species_prediction_payload(current_scores, already_scaled=True)

    previous_row = working_df.iloc[-2] if len(working_df) > 1 else latest_row

    def project_next(current_value, previous_value):
        drift = (float(current_value) - float(previous_value)) * 0.6
        return float(current_value) + drift

    projected_sst = project_next(
        latest_row.get("sst", latest_row.get("thetao", 27.0)),
        previous_row.get("sst", previous_row.get("thetao", 27.0)),
    )
    projected_chl = project_next(latest_row.get("chl", 0.8), previous_row.get("chl", 0.8))
    projected_o2 = project_next(latest_row.get("o2", 5.0), previous_row.get("o2", 5.0))
    projected_salinity = project_next(
        latest_row.get("salinity", latest_row.get("so", 35.0)),
        previous_row.get("salinity", previous_row.get("so", 35.0)),
    )

    predicted_scores = compute_species_health_from_raw(
        sst=projected_sst,
        chl=projected_chl,
        o2=projected_o2,
        salinity=projected_salinity,
    )
    predicted_payload = build_species_prediction_payload(predicted_scores, already_scaled=True)

    def classify_magnitude(delta_value):
        abs_delta = abs(delta_value)
        if abs_delta >= 10:
            return "significantly"
        if abs_delta >= 3:
            return "moderately"
        return "slightly"

    comparison_rows = []
    increased_species = []
    decreased_species = []
    for idx, meta in enumerate(SPECIES_METADATA):
        current_score = round(float(current_payload["species_scores"][idx]), 1)
        predicted_score = round(float(predicted_payload["species_scores"][idx]), 1)
        delta = round(predicted_score - current_score, 1)

        if delta > 0:
            increased_species.append((str(meta["key"]).title(), delta))
        elif delta < 0:
            decreased_species.append((str(meta["key"]).title(), delta))

        comparison_rows.append(
            {
                "species": str(meta["key"]).title(),
                "current": current_score,
                "predicted": predicted_score,
                "change": delta,
                "change_abs": round(abs(delta), 1),
                "change_direction": "up" if delta > 0 else "down" if delta < 0 else "flat",
            }
        )

    if increased_species and decreased_species:
        increased_names = ", ".join([name for name, _ in increased_species])
        decreased_names = ", ".join([name for name, _ in decreased_species])

        strongest_gain = max(increased_species, key=lambda pair: pair[1])
        gain_magnitude = classify_magnitude(strongest_gain[1])
        insight = (
            f"{decreased_names} are projected to decline, while {increased_names} improve {gain_magnitude} "
            "under forecasted environmental conditions."
        )
    elif increased_species:
        increased_names = ", ".join([name for name, _ in increased_species])
        strongest_gain = max(increased_species, key=lambda pair: pair[1])
        gain_magnitude = classify_magnitude(strongest_gain[1])
        insight = (
            f"{increased_names} are projected to improve {gain_magnitude} under forecasted "
            "environmental conditions."
        )
    elif decreased_species:
        decreased_names = ", ".join([name for name, _ in decreased_species])
        strongest_drop = min(decreased_species, key=lambda pair: pair[1])
        drop_magnitude = classify_magnitude(strongest_drop[1])
        insight = (
            f"{decreased_names} are projected to decline {drop_magnitude} under forecasted "
            "environmental conditions."
        )
    else:
        insight = "Species scores remain broadly stable between current and predicted conditions."

    return {
        "current": current_payload,
        "predicted": predicted_payload,
        "rows": comparison_rows,
        "insight": insight,
    }


def compute_importance_for_horizon(months_ahead):
    working_df = df.copy().sort_values("time")
    feature_cols = get_feature_columns(working_df)
    feature_labels = get_feature_labels(feature_cols)

    model_for_horizon = model_3m if months_ahead == 3 and model_3m is not None else model_1m
    scaler_for_horizon = scaler_3m if months_ahead == 3 and scaler_3m is not None else scaler_1m

    feature_window = working_df[feature_cols].tail(WINDOW_SIZE).reset_index(drop=True)
    X_input = feature_window.values.flatten().reshape(1, -1)
    X_model = safe_transform(X_input, scaler_for_horizon)

    background_count = min(len(working_df), 60)
    background_windows = []
    for idx in range(WINDOW_SIZE, background_count + 1):
        historical_window = working_df[feature_cols].iloc[idx - WINDOW_SIZE:idx].values.flatten()
        background_windows.append(historical_window)

    if not background_windows:
        background_windows = [X_input.reshape(-1)]

    X_background = np.asarray(background_windows)
    X_background_model = safe_transform(X_background, scaler_for_horizon)

    shap_values_1d = None
    shap_source = "linear-fallback"

    if shap is not None:
        try:
            # Use appropriate SHAP explainer based on model type
            if hasattr(model_for_horizon, "predict") and hasattr(model_for_horizon, "estimators_"):
                # Tree-based models (RandomForest, GradientBoosting, etc.)
                explainer = shap.TreeExplainer(model_for_horizon)
            else:
                # Linear models
                explainer = shap.LinearExplainer(model_for_horizon, X_background_model, feature_perturbation="interventional")
            
            shap_output = explainer.shap_values(X_model)
            shap_array = np.asarray(shap_output)

            if shap_array.ndim == 3:
                safe_idx = min(TARGET_OUTPUT_INDEX, shap_array.shape[0] - 1)
                shap_values_1d = shap_array[safe_idx, 0, :]
            elif shap_array.ndim == 2:
                shap_values_1d = shap_array[0, :]
            else:
                shap_values_1d = shap_array.reshape(-1)

            if shap_values_1d is not None and len(shap_values_1d) > 0:
                shap_source = "shap"
            else:
                logger.warning("SHAP computation returned empty values, using linear fallback")
        except Exception as exc:
            logger.warning("SHAP explainer failed for horizon %d months, falling back to linear contributions: %s", months_ahead, exc)

    if shap_values_1d is None:
        centered_input = np.asarray(X_model).reshape(-1) - np.asarray(X_background_model).mean(axis=0)

        if hasattr(model_for_horizon, "coef_"):
            coefficients = np.asarray(model_for_horizon.coef_, dtype=float)
            if coefficients.ndim > 1:
                safe_index = min(TARGET_OUTPUT_INDEX, coefficients.shape[0] - 1)
                coefficients = coefficients[safe_index]
            if coefficients.shape[0] != X_model.shape[1]:
                coefficients = np.resize(coefficients, X_model.shape[1])
            shap_values_1d = coefficients * centered_input
        elif hasattr(model_for_horizon, "feature_importances_"):
            importances = np.asarray(model_for_horizon.feature_importances_, dtype=float)
            if importances.shape[0] != X_model.shape[1]:
                importances = np.resize(importances, X_model.shape[1])

            # Keep sign from standardized deviation, use tree importances for magnitude.
            shap_values_1d = np.sign(centered_input) * (np.abs(centered_input) * importances)
        else:
            fallback_weights = np.ones(X_model.shape[1], dtype=float) / max(X_model.shape[1], 1)
            shap_values_1d = np.sign(centered_input) * (np.abs(centered_input) * fallback_weights)

    abs_values = np.abs(np.asarray(shap_values_1d, dtype=float))
    total_abs = float(abs_values.sum())
    if total_abs > 0:
        percent_contribution = (abs_values / total_abs) * 100.0
    else:
        percent_contribution = np.zeros_like(abs_values)

    return {
        "feature_labels": feature_labels,
        "importance_values": [round(float(value), 4) for value in abs_values.tolist()],
        "importance_percent": [round(float(value), 2) for value in percent_contribution.tolist()],
        "signed_shap_values": [round(float(value), 4) for value in np.asarray(shap_values_1d).tolist()],
        "shap_source": shap_source,
    }


def _infer_env_key_from_feature_label(feature_label):
    lower = str(feature_label).lower()
    if "o2" in lower:
        return "o2"
    if "chl" in lower:
        return "chl"
    if "thetao" in lower or "sst" in lower:
        return "sst"
    if "salinity" in lower or "so" in lower:
        return "salinity"
    return "unknown"


def _extract_lag_description(feature_label):
    label = str(feature_label)
    if "(t-2)" in label:
        return "2 months ago"
    if "(t-1)" in label:
        return "previous month"
    if "(t)" in label:
        return "current month"
    return "recent period"


def get_feature_display_metadata(feature_label):
    env_key = _infer_env_key_from_feature_label(feature_label)
    lag_desc = _extract_lag_description(feature_label)

    if env_key == "chl":
        friendly_name = f"Chlorophyll ({lag_desc})"
        tooltip = f"Chlorophyll concentration ({lag_desc})"
    elif env_key == "o2":
        friendly_name = f"Dissolved oxygen ({lag_desc})"
        tooltip = f"Dissolved oxygen level ({lag_desc})"
    elif env_key == "sst":
        friendly_name = f"Sea surface temperature ({lag_desc})"
        tooltip = f"Sea surface temperature ({lag_desc})"
    elif env_key == "salinity":
        friendly_name = f"Salinity ({lag_desc})"
        tooltip = f"Salinity concentration ({lag_desc})"
    else:
        friendly_name = str(feature_label)
        tooltip = f"Model feature value ({lag_desc})"

    return {
        "friendly_name": friendly_name,
        "tooltip": tooltip,
        "env_key": env_key,
    }


def build_prediction_impact_summary(top_features, env_inputs):
    impact_by_feature = []

    o2_z = float(env_inputs.get("o2_z", 0.0))
    sst_z = float(env_inputs.get("sst_z", 0.0))
    chl_z = float(env_inputs.get("chl_z", 0.0))

    for item in top_features[:3]:
        feature_name = item.get("name", "")
        metadata = get_feature_display_metadata(feature_name)
        env_key = metadata["env_key"]
        pct = float(item.get("avg_percent", 0.0))
        avg_signed = float(item.get("avg_signed", 0.0))

        # Positive SHAP pushes MEHI up (lower risk), negative SHAP pushes MEHI down (higher risk).
        if avg_signed > 0.05:
            impact_direction = "decrease"
            impact_arrow = "↓ Risk"
        elif avg_signed < -0.05:
            impact_direction = "increase"
            impact_arrow = "↑ Risk"
        else:
            impact_direction = "neutral"
            impact_arrow = "→ Neutral"

        if env_key == "o2":
            if o2_z < -0.5:
                narrative = "Low oxygen amplifies hypoxia pressure, raising fish and turtle risk."
            else:
                narrative = "Stable oxygen supports fish and turtle survival and helps reduce risk."
        elif env_key == "sst":
            if sst_z > 0.7:
                narrative = "Warm temperature anomalies elevate thermal stress and can increase turtle and fish risk."
            else:
                narrative = "Temperature contribution is currently near baseline, so thermal stress impact is limited."
        elif env_key == "chl":
            if chl_z > 0.8:
                narrative = "High chlorophyll increases bloom probability, which can elevate plant stress and indirectly affect fish and turtle survival."
            else:
                narrative = "Chlorophyll supports productivity without severe bloom stress, helping keep species risk moderate."
        elif env_key == "salinity":
            narrative = "Salinity contribution reflects osmotic stress pressure that can shift fish and turtle resilience."
        else:
            narrative = "This feature contributes to MEHI variation and indirectly shifts species risk levels."

        impact_by_feature.append(
            {
                "name": feature_name,
                "friendly_name": metadata["friendly_name"],
                "tooltip": metadata["tooltip"],
                "contribution": round(pct, 2),
                "impact_arrow": impact_arrow,
                "impact_direction": impact_direction,
                "impact_text": narrative,
            }
        )

    if not impact_by_feature:
        summary = "Insufficient explainability signals to map feature impact to species risk."
    else:
        primary = impact_by_feature[0]
        summary = (
            f"Primary driver: {primary['friendly_name']} ({primary['contribution']}% contribution). "
            "Its current behavior is the strongest factor behind near-term fish and turtle risk movement."
        )

    return impact_by_feature, summary


def run_forecast(months_ahead):
    working_df = df.copy()
    feature_cols = get_feature_columns(working_df)

    feature_window = working_df[feature_cols].tail(WINDOW_SIZE).reset_index(drop=True)
    predicted_mehi = float(working_df.iloc[-1]["Predicted_MEHI"])

    if months_ahead == 3 and model_3m is not None and model_3m is not model_1m:
        X_input = feature_window.values.flatten().reshape(1, -1)
        direct_output = predict_output(X_input, horizon=3)
        return extract_scalar_prediction(direct_output, TARGET_OUTPUT_INDEX)

    for _ in range(months_ahead):
        X_input = feature_window.values.flatten().reshape(1, -1)
        model_output = predict_output(X_input, horizon=1)
        predicted_mehi = extract_scalar_prediction(model_output, TARGET_OUTPUT_INDEX)

        next_row = feature_window.iloc[-1].copy()
        feature_window = pd.concat(
            [feature_window.iloc[1:], pd.DataFrame([next_row])],
            ignore_index=True,
        )

    return predicted_mehi


def get_default_context(error_message=None):
    return {
        "mehi": 0.0,
        "alert": "UNKNOWN",
        "alert_class": "alert-warning",
        "climate_trend": "Stable",
        "trend_labels": [],
        "trend_values": [],
        "trend_insight": "No trend insight available.",
        "critical_months_total": 0,
        "avg_critical_per_year": 0,
        "early_5_mean": 0,
        "recent_5_mean": 0,
        "climate_shift_delta": 0,
        "climate_shift_direction": "Stable",
        "anomaly_count": 0,
        "event_year_labels": [],
        "event_year_values": [],
        "shap_feature_labels": [],
        "shap_importance_1m": [],
        "shap_importance_3m": [],
        "shap_source_label": "Unavailable",
        "top_features": [],
        "explain_text": "Explainability information is currently unavailable.",
        "top_driver_badge": "Top Driver: Unavailable",
        "non_technical_summary": "No non-technical summary available.",
        "impact_summary_text": "No prediction impact summary available.",
        "impact_on_prediction": [],
        "env_inputs": {"sst_z": 0, "chl_z": 0, "o2_z": 0, "salinity_z": 0},
        "env_actuals": {
            "sst": {"value": 0.0, "unit": "deg C", "z": 0.0},
            "chl": {"value": 0.0, "unit": "mg/m3", "z": 0.0},
            "o2": {"value": 0.0, "unit": "mg/L", "z": 0.0},
            "salinity": {"value": 0.0, "unit": "PSU", "z": 0.0},
        },
        "ecosystem_indicators": {
            "bloom_risk": 0,
            "bloom_status": "LOW",
            "hypoxia_risk": 0,
            "hypoxia_status": "LOW",
            "water_quality_index": 0,
            "wqi_status": "POOR",
        },
        "wqi_trend_labels": [],
        "wqi_trend_values": [],
        "ecosystem_prediction": None,
        "ecosystem_current": None,
        "species_comparison_rows": [],
        "species_change_insight": "No comparison insight available.",
        "species_labels": [],
        "species_scores": [],
        "forecast": None,
        "error_message": error_message,
        "feature_columns_used": [],
        "active_page": "dashboard",
        "latest_update": "N/A",
        "latest_data_label": "N/A",
        "record_count": 0,
        "trend_actual_values": [],
        "trend_forecast_values": [],
        "forecast_note": "Future values are model-generated.",
        "live_parameters": {
            "sst": {"label": "SST", "icon": "🌡", "value": 0, "unit": "deg C", "arrow": "→", "direction": "flat", "delta": 0},
            "chlorophyll": {"label": "Chlorophyll", "icon": "🟢", "value": 0, "unit": "mg/m3", "arrow": "→", "direction": "flat", "delta": 0},
            "oxygen": {"label": "Oxygen", "icon": "💧", "value": 0, "unit": "mg/L", "arrow": "→", "direction": "flat", "delta": 0},
            "salinity": {"label": "Salinity", "icon": "🌊", "value": 0, "unit": "PSU", "arrow": "→", "direction": "flat", "delta": 0},
        },
        "live_risk_alerts": [],
        "notifications": [],
        "notification_count": 0,
        "live_log_rows": [],
    }


def get_dashboard_context(prediction_mode=None):
    working_df = df.copy()
    working_df["time"] = pd.to_datetime(working_df["time"], errors="coerce")
    working_df = working_df.dropna(subset=["time"]).sort_values("time")
    working_df["year"] = working_df["time"].dt.year

    if working_df.empty:
        raise ValueError("No valid data available for dashboard context.")

    if "data_source" not in working_df.columns:
        working_df["data_source"] = "actual"
    else:
        working_df["data_source"] = working_df["data_source"].fillna("actual").str.lower()

    feature_cols = get_feature_columns(working_df)
    latest = working_df.iloc[-1]
    trend_df = working_df.tail(12)
    env_inputs = get_environment_input_snapshot(working_df)
    env_actuals = {
        "sst": {
            "value": round(float(latest.get("sst", latest.get("thetao", 0.0))), 2),
            "unit": "deg C",
            "z": round(float(env_inputs["sst_z"]), 2),
        },
        "chl": {
            "value": round(float(latest.get("chl", 0.0)), 3),
            "unit": "mg/m3",
            "z": round(float(env_inputs["chl_z"]), 2),
        },
        "o2": {
            "value": round(float(latest.get("o2", 0.0)), 2),
            "unit": "mg/L",
            "z": round(float(env_inputs["o2_z"]), 2),
        },
        "salinity": {
            "value": round(float(latest.get("salinity", 0.0)), 2),
            "unit": "PSU",
            "z": round(float(env_inputs["salinity_z"]), 2),
        },
    }
    ecosystem_indicators = compute_environmental_indicators_from_z(
        env_inputs["sst_z"],
        env_inputs["chl_z"],
        env_inputs["o2_z"],
        env_inputs["salinity_z"],
    )
    live_parameters = build_live_parameter_cards(working_df)
    wqi_trend_labels, wqi_trend_values = compute_wqi_trend(working_df)
    live_risk_alerts = build_live_risk_alerts(str(latest["Ecosystem_Alert"]), ecosystem_indicators)
    notifications = build_notification_feed(str(latest["Ecosystem_Alert"]), ecosystem_indicators, live_parameters)
    live_log_rows = build_live_log_rows(working_df)

    chart_df = working_df[working_df["time"] >= pd.Timestamp("2023-01-01")].copy()
    if chart_df.empty:
        chart_df = working_df.tail(24).copy()

    trend_labels = chart_df["time"].dt.strftime("%b %Y").tolist()
    trend_actual_values = []
    trend_forecast_values = []
    for _, row in chart_df.iterrows():
        mehi_value = round(float(row["Predicted_MEHI"]), 2)
        if str(row.get("data_source", "actual")) == "forecast":
            trend_actual_values.append(None)
            trend_forecast_values.append(mehi_value)
        else:
            trend_actual_values.append(mehi_value)
            trend_forecast_values.append(None)

    window_size = min(12, len(working_df))
    early_mehi_mean = working_df["Predicted_MEHI"].head(window_size).mean()
    recent_mehi_mean = working_df["Predicted_MEHI"].tail(window_size).mean()

    if recent_mehi_mean > early_mehi_mean:
        climate_trend = "Improving"
    elif recent_mehi_mean < early_mehi_mean:
        climate_trend = "Declining"
    else:
        climate_trend = "Stable"

    last_quarter_mean = trend_df["Predicted_MEHI"].tail(3).mean()
    prev_quarter_mean = trend_df["Predicted_MEHI"].tail(6).head(3).mean()

    if last_quarter_mean < prev_quarter_mean:
        trend_insight = "MEHI shows increased ecosystem stress in the last quarter."
    elif last_quarter_mean > prev_quarter_mean:
        trend_insight = "MEHI indicates improving ecosystem conditions in the last quarter."
    else:
        trend_insight = "MEHI remains broadly stable over the last quarter."

    alert_class_map = {
        "STABLE": "alert-stable",
        "WARNING": "alert-warning",
        "CRITICAL": "alert-critical",
    }

    explain_1m = compute_importance_for_horizon(1)
    explain_3m = compute_importance_for_horizon(3)

    importance_1m = np.asarray(explain_1m["importance_percent"])
    importance_3m = np.asarray(explain_3m["importance_percent"])

    top_features = []
    if importance_1m.size > 0 and importance_3m.size > 0:
        combined_importance = (importance_1m + importance_3m) / 2
        signed_1m = np.asarray(explain_1m.get("signed_shap_values", [0.0] * len(importance_1m)))
        signed_3m = np.asarray(explain_3m.get("signed_shap_values", [0.0] * len(importance_3m)))
        combined_signed = (signed_1m + signed_3m) / 2
        top_indices = np.argsort(combined_importance)[::-1][:5]

        for idx in top_indices:
            delta = float(importance_3m[idx] - importance_1m[idx])
            metadata = get_feature_display_metadata(explain_1m["feature_labels"][idx])
            if delta > 0:
                comparison = "Higher in 3M"
            elif delta < 0:
                comparison = "Higher in 1M"
            else:
                comparison = "Equal"

            top_features.append(
                {
                    "name": explain_1m["feature_labels"][idx],
                    "friendly_name": metadata["friendly_name"],
                    "tooltip": metadata["tooltip"],
                    "importance_1m": f"{round(float(importance_1m[idx]), 1)}%",
                    "importance_3m": f"{round(float(importance_3m[idx]), 1)}%",
                    "avg_percent": round(float(combined_importance[idx]), 2),
                    "avg_signed": round(float(combined_signed[idx]), 4),
                    "comparison": comparison,
                }
            )

    leading_feature = top_features[0]["name"] if top_features else "MEHI drivers"
    explain_text = (
        f"SHAP analysis indicates {leading_feature} is the strongest contributor to forecast behavior. "
        "Contributions are normalized as percentage share of total absolute SHAP impact."
    )
    impact_on_prediction, impact_summary_text = build_prediction_impact_summary(top_features, env_inputs)

    if top_features:
        top_driver_badge = f"Top Driver: {top_features[0]['friendly_name']}"
        non_technical_summary = (
            f"Ecosystem prediction is primarily driven by {top_features[0]['friendly_name'].lower()}, "
            "with moderate influence from salinity and temperature variations."
        )
    else:
        top_driver_badge = "Top Driver: Unavailable"
        non_technical_summary = "Ecosystem prediction summary is unavailable for the current data state."

    if explain_1m.get("shap_source") == "shap" and explain_3m.get("shap_source") == "shap":
        shap_source_label = "Advanced AI Explanation"
    elif explain_1m.get("shap_source") == "shap" or explain_3m.get("shap_source") == "shap":
        shap_source_label = "Hybrid Explanation"
    else:
        shap_source_label = "Statistical Analysis"

    critical_months_total = int((working_df["Ecosystem_Alert"] == "CRITICAL").sum())
    total_years = int(working_df["year"].nunique())
    avg_critical_per_year = round(critical_months_total / total_years if total_years else 0, 2)

    unique_years = sorted(working_df["year"].dropna().unique().tolist())
    early_years = unique_years[:5]
    recent_years = unique_years[-5:]

    early_5_mean = (
        working_df[working_df["year"].isin(early_years)]["Predicted_MEHI"].mean()
        if early_years
        else 0
    )
    recent_5_mean = (
        working_df[working_df["year"].isin(recent_years)]["Predicted_MEHI"].mean()
        if recent_years
        else 0
    )
    climate_shift_delta = float(recent_5_mean - early_5_mean)

    if climate_shift_delta > 0:
        climate_shift_direction = "Improving"
    elif climate_shift_delta < 0:
        climate_shift_direction = "Declining"
    else:
        climate_shift_direction = "Stable"

    anomaly_mask = pd.Series(False, index=working_df.index)
    for z_col in ["sst_z", "chl_z", "o2_z"]:
        if z_col in working_df.columns:
            anomaly_mask = anomaly_mask | (working_df[z_col].abs() >= 1.0)

    anomaly_count = int(anomaly_mask.sum())

    alert_event_mask = working_df["Ecosystem_Alert"].isin(["WARNING", "CRITICAL"])
    event_mask = alert_event_mask | anomaly_mask
    events_per_year = (
        working_df[event_mask]
        .groupby("year")
        .size()
        .reindex(unique_years, fill_value=0)
    )

    context = {
        "mehi": round(float(latest["Predicted_MEHI"]), 2),
        "current_month": pd.to_datetime(latest["time"]).strftime("%B %Y"),
        "alert": str(latest["Ecosystem_Alert"]),
        "alert_class": alert_class_map.get(str(latest["Ecosystem_Alert"]), "alert-warning"),
        "climate_trend": climate_trend,
        "trend_labels": trend_labels,
        "trend_values": [round(float(value), 2) for value in chart_df["Predicted_MEHI"].tolist()],
        "trend_actual_values": trend_actual_values,
        "trend_forecast_values": trend_forecast_values,
        "trend_insight": trend_insight,
        "critical_months_total": critical_months_total,
        "avg_critical_per_year": avg_critical_per_year,
        "early_5_mean": round(float(early_5_mean), 2),
        "recent_5_mean": round(float(recent_5_mean), 2),
        "climate_shift_delta": round(climate_shift_delta, 2),
        "climate_shift_direction": climate_shift_direction,
        "anomaly_count": anomaly_count,
        "event_year_labels": [str(year) for year in events_per_year.index.tolist()],
        "event_year_values": [int(value) for value in events_per_year.values.tolist()],
        "shap_feature_labels": explain_1m["feature_labels"],
        "shap_importance_1m": [round(float(value), 3) for value in importance_1m.tolist()],
        "shap_importance_3m": [round(float(value), 3) for value in importance_3m.tolist()],
        "shap_source_label": shap_source_label,
        "top_features": top_features,
        "explain_text": explain_text,
        "top_driver_badge": top_driver_badge,
        "non_technical_summary": non_technical_summary,
        "impact_summary_text": impact_summary_text,
        "impact_on_prediction": impact_on_prediction,
        "env_inputs": env_inputs,
        "env_actuals": env_actuals,
        "ecosystem_indicators": ecosystem_indicators,
        "wqi_trend_labels": wqi_trend_labels,
        "wqi_trend_values": wqi_trend_values,
        "ecosystem_prediction": None,
        "ecosystem_current": None,
        "species_comparison_rows": [],
        "species_change_insight": "No comparison insight available.",
        "species_labels": [],
        "species_scores": [],
        "forecast": None,
        "error_message": None,
        "feature_columns_used": feature_cols,
        "latest_update": latest["time"].strftime("%Y-%m-%d"),
        "latest_data_label": (
            f"{int(latest['time'].year)} (Forecasted)"
            if str(latest.get("data_source", "actual")) == "forecast"
            else str(int(latest["time"].year))
        ),
        "record_count": int(len(working_df)),
        "forecast_note": "Future values are model-generated from the trained model (not API/live feed).",
        "live_parameters": live_parameters,
        "live_risk_alerts": live_risk_alerts,
        "notifications": notifications,
        "notification_count": len(notifications),
        "live_log_rows": live_log_rows,
    }

    if prediction_mode in (1, 3):
        predicted_mehi = run_forecast(prediction_mode)
        current_mehi = float(latest["Predicted_MEHI"])
        change = predicted_mehi - current_mehi

        context["forecast"] = {
            "months": prediction_mode,
            "predicted_mehi": round(predicted_mehi, 2),
            "predicted_alert": infer_alert_level(predicted_mehi),
            "change": round(change, 2),
            "change_direction": "increase" if change >= 0 else "decrease",
        }

    if prediction_mode == "ecosystem":
        species_context = compute_species_current_vs_predicted()
        context["ecosystem_current"] = species_context["current"]
        context["ecosystem_prediction"] = species_context["predicted"]
        context["species_comparison_rows"] = species_context["rows"]
        context["species_change_insight"] = species_context["insight"]
        context["species_labels"] = context["ecosystem_prediction"]["species_labels"]
        context["species_scores"] = context["ecosystem_prediction"]["species_scores"]

    return context


def build_export_report_rows(context):
    notifications = context.get("notifications", [])
    top_features = context.get("top_features", [])
    live_alerts = context.get("live_risk_alerts", [])

    rows = [
        ["Section", "Metric", "Value", "Details"],
        ["Report", "Generated At", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "Marine Risk System export"],
        ["Dataset", "Latest Update", context.get("latest_update", "N/A"), context.get("latest_data_label", "N/A")],
        ["Dataset", "Record Count", context.get("record_count", 0), "Rows available for analysis"],
        ["Forecast", "Current MEHI", context.get("mehi", 0), f"Alert: {context.get('alert', 'UNKNOWN')}"] ,
        ["Forecast", "Climate Trend", context.get("climate_trend", "Stable"), context.get("trend_insight", "")],
        ["Risk", "Bloom Status", context.get("ecosystem_indicators", {}).get("bloom_status", "LOW"), f"WQI: {context.get('ecosystem_indicators', {}).get('water_quality_index', 0)}"],
        ["Risk", "Hypoxia Status", context.get("ecosystem_indicators", {}).get("hypoxia_status", "LOW"), f"WQI Status: {context.get('ecosystem_indicators', {}).get('wqi_status', 'POOR')}"] ,
        ["Risk", "Notification Count", len(notifications), "; ".join([item.get("title", "") for item in notifications])],
        ["Risk", "Active Alerts", len(live_alerts), "; ".join([f"{item.get('title', '')}: {item.get('state', '')}" for item in live_alerts])],
        ["Explainability", "SHAP Source", context.get("shap_source_label", "Unavailable"), context.get("top_driver_badge", "Top Driver: Unavailable")],
    ]

    for feature in top_features[:5]:
        rows.append(
            [
                "Explainability",
                feature.get("friendly_name", feature.get("name", "Feature")),
                feature.get("avg_percent", "N/A"),
                f"1M: {feature.get('importance_1m', 'N/A')} | 3M: {feature.get('importance_3m', 'N/A')} | {feature.get('comparison', '')}",
            ]
        )

    if notifications:
        for item in notifications:
            rows.append(
                [
                    "Notification",
                    item.get("title", "Alert"),
                    item.get("level", "UNKNOWN"),
                    item.get("message", ""),
                ]
            )

    return rows


def export_dashboard_report_csv():
    context = get_dashboard_context()
    rows = build_export_report_rows(context)

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)

    filename = f"marine_risk_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def render_page(template_name, active_page, prediction_mode=None):
    try:
        context = get_dashboard_context(prediction_mode)
        context["active_page"] = active_page
        return render_template(template_name, **context)
    except Exception as exc:
        logger.exception("Page render error (%s): %s", template_name, exc)
        fallback_context = get_default_context(
            "System recovered from a backend issue. Check model files and dataset columns."
        )
        fallback_context["active_page"] = active_page
        return render_template(template_name, **fallback_context), 200


# -------------------------
# Load Data + Models + Scalers
# -------------------------

raw_df = pd.read_csv(DATA_PATH)
# FIX: Convert oxygen to mg/L if values are too high
if "o2" in raw_df.columns and raw_df["o2"].mean() > 20:
    raw_df["o2"] = raw_df["o2"] / 31.25
df = clean_dataset(raw_df)

model_1m = load_pickle_from_candidates(
    [FINAL_MODEL_1M_PATH, LEGACY_MODEL_1M_PATH],
    "1-month model",
    required=True,
)

model_3m = load_pickle_from_candidates(
    [FINAL_MODEL_3M_PATH, LEGACY_MODEL_3M_PATH],
    "3-month model",
    required=False,
)
if model_3m is None:
    model_3m = model_1m

scaler_1m = load_pickle_from_candidates(
    [FINAL_SCALER_1M_PATH, LEGACY_SCALER_1M_PATH],
    "1-month scaler",
    required=False,
)

scaler_3m = load_pickle_from_candidates(
    [FINAL_SCALER_3M_PATH, LEGACY_SCALER_3M_PATH],
    "3-month scaler",
    required=False,
)
if scaler_3m is None:
    scaler_3m = scaler_1m

save_artifact_if_possible(model_1m, FINAL_MODEL_1M_PATH, "1-month model")
save_artifact_if_possible(model_3m, FINAL_MODEL_3M_PATH, "3-month model")
save_artifact_if_possible(scaler_1m, FINAL_SCALER_1M_PATH, "1-month scaler")
save_artifact_if_possible(scaler_3m, FINAL_SCALER_3M_PATH, "3-month scaler")


# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return render_page("dashboard.html", "dashboard")


@app.route("/forecast")
def forecast_page():
    return render_page("forecast.html", "forecast")


@app.route("/analytics")
def analytics_page():
    return render_page("analytics.html", "analytics")


@app.route("/explainability")
def explainability_page():
    return render_page("explainability.html", "explainability")


@app.route("/reports/latest.csv")
def download_latest_report_csv():
    return export_dashboard_report_csv()


@app.post("/predict")
def predict():
    months = request.form.get("months", "1")

    try:
        prediction_months = int(months)
    except ValueError:
        prediction_months = 1

    if prediction_months not in (1, 3):
        prediction_months = 1

    return render_page("forecast.html", "forecast", prediction_months)


@app.post("/predict_ecosystem")
def predict_ecosystem():
    return render_page("forecast.html", "forecast", "ecosystem")


@app.post("/api/simulate")
def simulate_scenario():
    try:
        payload = request.get_json(silent=True) or {}
        working_df = df.copy().sort_values("time")
        latest_row = working_df.iloc[-1]

        def resolve_value(raw_key, z_key, column_name, fallback_value):
            raw_value = payload.get(raw_key)
            if raw_value not in (None, ""):
                return _safe_float(raw_value, raw_key)

            z_value = payload.get(z_key)
            if z_value not in (None, ""):
                return z_to_raw(column_name, _safe_float(z_value, z_key), working_df)

            return float(fallback_value)

        sst_value = resolve_value("sst", "sst_z", "sst", latest_row.get("sst", latest_row.get("thetao", 0.0)))
        chl_value = resolve_value("chl", "chl_z", "chl", latest_row.get("chl", 0.0))
        o2_value = resolve_value("o2", "o2_z", "o2", latest_row.get("o2", 0.0))
        salinity_value = resolve_value("salinity", "salinity_z", "salinity", latest_row.get("salinity", latest_row.get("so", 0.0)))

        result = build_scenario_payload(sst_value, chl_value, o2_value, salinity_value)

        species_scores = result["ecosystem_prediction"].get("species_scores", [0.0, 0.0, 0.0])
        if len(species_scores) < 3:
            species_scores = list(species_scores) + [0.0] * (3 - len(species_scores))

        response = {
            "baseline_mehi": result["baseline_mehi"],
            "scenario_mehi": result["scenario_mehi"],
            "mehi_change": result["mehi_change"],
            "change_direction": result["change_direction"],
            "current_alert": result["current_alert"],
            "scenario_alert": result["scenario_alert"],
            "turtle_risk": round(float(species_scores[0]), 1),
            "fish_risk": round(float(species_scores[1]), 1),
            "plant_risk": round(float(species_scores[2]), 1),
            "bloom_risk": result["ecosystem_indicators"]["bloom_risk"],
            "hypoxia_risk": result["ecosystem_indicators"]["hypoxia_risk"],
            "water_quality_index": result["ecosystem_indicators"]["water_quality_index"],
            "wqi_status": result["ecosystem_indicators"]["wqi_status"],
            "scenario_inputs": result["scenario_inputs"],
            "ecosystem_prediction": result["ecosystem_prediction"],
        }

        return jsonify({"ok": True, "result": result, "response": response})
    except Exception as exc:
        logger.exception("Scenario simulation error: %s", exc)
        return jsonify({"ok": False, "error": "Simulation failed for the provided scenario."}), 400


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return error

    logger.exception("Unhandled server error: %s", error)
    fallback_context = get_default_context(
        "Unexpected server error handled safely. Please retry prediction."
    )
    return render_template("dashboard.html", **fallback_context), 500


if __name__ == "__main__":
    app.run(debug=True)
