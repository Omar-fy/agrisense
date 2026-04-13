"""
AgriSense Commons — Core Pipeline
===================================
TeleAgriCulture Brief · ARTD2143 · Winchester School of Art · 2025/26

Transforms raw TeleAgriCulture sensor data into intelligent agronomic insights.

Pipeline stages:
  1. fetch_sensor_data()      — paginated TeleAgriCulture API fetch
  2. fetch_latest_reading()   — single latest reading for alert polling
  3. fetch_owm_current()      — current outdoor conditions (OpenWeatherMap)
  4. fetch_owm_forecast()     — 5-day / 3-hourly forecast (OpenWeatherMap)
  5. clean()                  — drop nulls, physical bounds, dedup
  6. calibrate_soil_moisture()— convert raw ADC to 0-100% if needed
  7. featurise()              — lagged feature engineering (from Schmiede notebook)
  8. build_classifier()       — shallow Decision Tree on agronomic thresholds
  9. classify_windows()       — apply classifier to full time series
  10. detect_anomalies()      — rolling z-score anomaly detection
  11. build_peer_profile()    — serialisable stats profile for benchmarking
  12. compare_to_peers()      — cross-board percentile comparison
  13. compute_deviation()     — board mean vs OWM outdoor baseline
  14. compute_indoor_outdoor_delta() — learned delta for forecast projection
  15. project_forecast()      — 5-day indoor condition projection
  16. generate_summary()      — sensor-adaptive plain-language summary
  17. run_pipeline()          — full orchestrator (insights endpoint)
  18. run_forecast_pipeline() — forecast orchestrator (forecast endpoint)

Default board: Kit 1001 (Schmiede festival, Salzburg 2019)
Sensor naming: ftTemp, ftHumid, ftSoilMoist, ftMoisture, ftTDS, ftLight, ftPressure
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

TELEAGRI_BASE    = "https://kits.teleagriculture.org/api/kits"
OWM_CURRENT_URL  = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

DEFAULT_KIT_ID = "1001"
DEFAULT_SENSOR = "ftTemp"
DEFAULT_LAT    = 47.7981   # Salzburg — Schmiede board location
DEFAULT_LON    = 13.0456

ANOMALY_Z_THRESHOLD = 2.5
ANOMALY_WINDOW      = 24   # rolling window in number of readings

SENSOR_UNITS = {
    "ftTemp":      "°C",
    "ftHumid":     "%",
    "ftSoilMoist": "%",
    "ftMoisture":  "%",
    "ftLight":     " lux",
    "ftTDS":       " ppm",
    "ftSoilTemp":  "°C",
    "ftPressure":  " hPa",
    "ftADC":       "",
}

# Physical validity bounds per sensor — anything outside is a hardware error
SENSOR_BOUNDS = {
    "ftTemp":      (-40.0,  85.0),
    "ftHumid":     (0.0,   100.0),
    "ftSoilMoist": (0.0,   100.0),
    "ftMoisture":  (0.0,   100.0),
    "ftLight":     (0.0,   100000.0),
    "ftPressure":  (870.0, 1085.0),
    "ftTDS":       (0.0,   3000.0),
    "ftSoilTemp":  (-10.0,  85.0),
    "ftADC":       (0.0,   4095.0),
}

# Agronomic state thresholds — derived from standard growing guidelines.
# These produce rule-based labels used to train the Decision Tree.
THRESHOLDS = {
    "ftTemp": [
        ("cold_stress",  lambda v: v < 5),
        ("cool",         lambda v: 5  <= v < 15),
        ("optimal",      lambda v: 15 <= v < 28),
        ("heat_stress",  lambda v: v >= 28),
    ],
    "ftHumid": [
        ("drought_risk",  lambda v: v < 30),
        ("moderate",      lambda v: 30 <= v < 50),
        ("optimal",       lambda v: 50 <= v < 80),
        ("humidity_risk", lambda v: v >= 80),
    ],
    "ftSoilMoist": [
        ("drought_risk", lambda v: v < 20),
        ("moderate",     lambda v: 20 <= v < 40),
        ("optimal",      lambda v: 40 <= v < 70),
        ("waterlogged",  lambda v: v >= 80),
    ],
    "ftMoisture": [
        ("drought_risk", lambda v: v < 20),
        ("moderate",     lambda v: 20 <= v < 40),
        ("optimal",      lambda v: 40 <= v < 70),
        ("waterlogged",  lambda v: v >= 80),
    ],
    "ftTDS": [
        ("nutrient_low",  lambda v: v < 200),
        ("optimal",       lambda v: 200 <= v < 800),
        ("nutrient_high", lambda v: v >= 800),
    ],
}

# States that should trigger an alert
ALERT_STATES = {
    "cold_stress", "heat_stress", "drought_risk",
    "humidity_risk", "waterlogged", "nutrient_high", "nutrient_low"
}

# Plain-language templates per sensor and state
TEMPLATES = {
    "ftTemp": {
        "optimal":      "Temperature is in the ideal growing range. Conditions are favourable.",
        "heat_stress":  "Temperature is above optimal. Consider ventilation or shading.",
        "cold_stress":  "Temperature is critically low. Frost risk is elevated.",
        "cool":         "Temperature is below optimal but manageable. Expect slower growth.",
        "unknown":      "Temperature data available but conditions unclear.",
    },
    "ftHumid": {
        "optimal":      "Humidity is in a healthy range for most crops.",
        "drought_risk": "Humidity is very low. Transpiration stress is likely.",
        "humidity_risk":"High humidity detected. Monitor for fungal disease risk.",
        "moderate":     "Humidity is moderate. Keep an eye on trends.",
        "unknown":      "Humidity data available but conditions unclear.",
    },
    "ftSoilMoist": {
        "optimal":      "Soil moisture is well-balanced. No irrigation needed.",
        "drought_risk": "Soil moisture is critically low. Irrigation recommended soon.",
        "waterlogged":  "Soil is waterlogged. Check drainage to prevent root rot.",
        "moderate":     "Soil moisture is moderate. Monitor over the next day.",
        "unknown":      "Soil moisture data available but conditions unclear.",
    },
    "ftMoisture": {
        "optimal":      "Soil moisture is well-balanced. No irrigation needed.",
        "drought_risk": "Soil moisture is critically low. Irrigation recommended soon.",
        "waterlogged":  "Soil is waterlogged. Check drainage to prevent root rot.",
        "moderate":     "Soil moisture is moderate. Monitor over the next day.",
        "unknown":      "Soil moisture data available but conditions unclear.",
    },
}

FORECAST_RECOMMENDATIONS = {
    "optimal":      "Conditions look favourable. No intervention needed.",
    "heat_stress":  "Heat stress likely. Plan ventilation or shading.",
    "cold_stress":  "Frost risk. Consider insulation or heating overnight.",
    "cool":         "Cool conditions expected. Growth may slow.",
    "drought_risk": "Low humidity forecast. Monitor soil moisture closely.",
    "humidity_risk":"High humidity expected. Increase airflow to reduce disease risk.",
    "moderate":     "Moderate conditions. Keep monitoring.",
    "unknown":      "Forecast available but conditions unclear.",
}

ALERT_DESCRIPTIONS = {
    "cold_stress":   "Temperature has dropped critically low for most crops.",
    "heat_stress":   "Temperature has spiked above safe growing threshold.",
    "drought_risk":  "Moisture critically low — plants may be under water stress.",
    "humidity_risk": "Humidity elevated — fungal disease risk is high.",
    "waterlogged":   "Waterlogging detected — root rot risk.",
    "nutrient_high": "Nutrient concentration dangerously high.",
    "nutrient_low":  "Nutrient levels critically low.",
}


# ─────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_sensor_data(kit_id: str, sensor_name: str) -> pd.DataFrame:
    """
    Fetch the complete paginated measurement history for one sensor on one kit.

    Uses cursor-based pagination from the TeleAgriCulture REST API.
    Handles both link-based (links.next) and cursor-based (meta.next_cursor)
    pagination automatically.

    Args:
        kit_id:      TeleAgriCulture Kit ID, e.g. "1001"
        sensor_name: Sensor identifier, e.g. "ftTemp"

    Returns:
        DataFrame with columns: created_at (datetime), value (float)
        Empty DataFrame if no data found.

    Raises:
        requests.HTTPError: if the API returns a non-2xx response
        ValueError: if the response structure is unexpected
    """
    all_measurements = []
    url = f"{TELEAGRI_BASE}/{kit_id}/{sensor_name}/measurements"

    while url:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        if not isinstance(data.get("data"), list):
            raise ValueError(
                f"Unexpected API response: kit={kit_id}, sensor={sensor_name}. "
                f"Got: {list(data.keys())}"
            )

        all_measurements.extend(data["data"])

        links = data.get("links", {})
        meta  = data.get("meta", {})

        if links.get("next"):
            url = links["next"]
        elif meta.get("next_cursor"):
            url = (f"{TELEAGRI_BASE}/{kit_id}/{sensor_name}/measurements"
                   f"?page[cursor]={meta['next_cursor']}")
        else:
            url = None

    if not all_measurements:
        return pd.DataFrame(columns=["created_at", "value"])

    df = pd.DataFrame(all_measurements)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["value"]      = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values("created_at").reset_index(drop=True)


def fetch_latest_reading(kit_id: str, sensor_name: str) -> Optional[dict]:
    """
    Fetch only the most recent reading for a sensor.

    Used by the alert polling loop to avoid refetching full history.
    The TeleAgriCulture API returns newest readings first by default.

    Returns:
        dict with keys: created_at (datetime), value (float)
        None on any failure (network error, empty response, etc.)
    """
    try:
        r = requests.get(
            f"{TELEAGRI_BASE}/{kit_id}/{sensor_name}/measurements",
            timeout=10
        )
        r.raise_for_status()
        items = r.json().get("data", [])
        if not items:
            return None
        latest = items[0]
        return {
            "created_at": pd.to_datetime(latest["created_at"]),
            "value":      float(latest["value"]),
        }
    except Exception:
        return None


def fetch_owm_current(lat: float, lon: float, api_key: str) -> dict:
    """
    Fetch current outdoor conditions from OpenWeatherMap.

    Uses the /weather endpoint (free tier compatible).

    Returns:
        dict with keys: temp_c, feels_like, humidity, description, location
        Empty dict on any failure — caller must check for presence of keys.
    """
    try:
        r = requests.get(OWM_CURRENT_URL, params={
            "lat": lat, "lon": lon,
            "appid": api_key, "units": "metric"
        }, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "temp_c":      round(d["main"]["temp"], 1),
            "feels_like":  round(d["main"]["feels_like"], 1),
            "humidity":    d["main"]["humidity"],
            "description": d["weather"][0]["description"],
            "location":    d.get("name", "unknown"),
        }
    except Exception:
        return {}


def fetch_owm_forecast(lat: float, lon: float, api_key: str) -> pd.DataFrame:
    """
    Fetch the 5-day / 3-hourly forecast from OpenWeatherMap (40 slots).

    Uses the /forecast endpoint (free tier compatible).

    Returns:
        DataFrame with columns: timestamp, temp_c, feels_like, humidity,
                                 description, dt_txt, date
        Empty DataFrame on failure.
    """
    try:
        r = requests.get(OWM_FORECAST_URL, params={
            "lat": lat, "lon": lon,
            "appid": api_key, "units": "metric", "cnt": 40
        }, timeout=10)
        r.raise_for_status()

        rows = []
        for item in r.json().get("list", []):
            rows.append({
                "timestamp":   pd.to_datetime(item["dt"], unit="s", utc=True),
                "temp_c":      item["main"]["temp"],
                "feels_like":  item["main"]["feels_like"],
                "humidity":    item["main"]["humidity"],
                "description": item["weather"][0]["description"],
                "dt_txt":      item.get("dt_txt", ""),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = df["timestamp"].dt.date
        return df

    except Exception:
        return pd.DataFrame(
            columns=["timestamp", "temp_c", "feels_like",
                     "humidity", "description", "dt_txt", "date"]
        )


# ─────────────────────────────────────────────────────────────
# 2. CLEANING AND CALIBRATION
# ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, sensor_name: str = "ftTemp") -> pd.DataFrame:
    """
    Clean raw sensor data:
      - Drop rows with null created_at or value
      - Remove physically impossible values using SENSOR_BOUNDS
      - Remove duplicate timestamps (keep last reading)
      - Sort chronologically and reset index

    Never modifies input in place — always returns a copy.
    """
    df = df.copy().dropna(subset=["created_at", "value"])
    lo, hi = SENSOR_BOUNDS.get(sensor_name, (-1e9, 1e9))
    df = df[(df["value"] >= lo) & (df["value"] <= hi)]
    df = df.drop_duplicates(subset=["created_at"], keep="last")
    return df.sort_values("created_at").reset_index(drop=True)


def calibrate_soil_moisture(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """
    Convert raw ADC readings from capacitive soil moisture sensor v1.0
    to a 0-100% volumetric moisture scale.

    Capacitive soil moisture v1.0 typical calibration values:
      ~3200 ADC counts = dry air (0%)
      ~1500 ADC counts = saturated soil (100%)

    If the sensor is already reporting values in 0-100 range (e.g. ftSoilMoist)
    this function is a no-op. Only activates for ftADC or out-of-range raw values.
    """
    if sensor_name not in ("ftADC",) and df["value"].max() <= 100:
        return df   # Already calibrated

    df = df.copy()
    ADC_DRY = 3200.0
    ADC_WET  = 1500.0
    df["value"] = (
        (ADC_DRY - df["value"]) / (ADC_DRY - ADC_WET) * 100
    ).clip(0, 100)
    return df


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def featurise(
    df: pd.DataFrame, look_back: int = 3
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a supervised ML dataset from the time series using lagged features.

    Directly mirrors create_supervised_dataset() from the original
    TeleAgriCulture Schmiede notebook, extended with rolling statistics
    for richer feature representation.

    For each timestep t, creates features:
      value_lag_1, value_lag_2, ..., value_lag_n  (previous n readings)
      rolling_mean                                  (mean of lag window)
      rolling_std                                   (std of lag window)
      rolling_range                                 (max - min of lag window)

    Target y: the value at timestep t (next-step prediction).

    Args:
        df:        cleaned DataFrame with 'value' column
        look_back: number of lagged timesteps to use as features (default 3)

    Returns:
        X (pd.DataFrame): feature matrix, shape (n - look_back, look_back + 3)
        y (pd.Series):    target values, length (n - look_back)
    """
    values = df["value"].values
    X_rows, y_vals = [], []

    for i in range(len(values) - look_back):
        X_rows.append(values[i : i + look_back])
        y_vals.append(values[i + look_back])

    cols = [f"value_lag_{i+1}" for i in range(look_back)]
    X    = pd.DataFrame(X_rows, columns=cols)
    y    = pd.Series(y_vals, name="target_value")

    X["rolling_mean"]  = X[cols].mean(axis=1)
    X["rolling_std"]   = X[cols].std(axis=1).fillna(0)
    X["rolling_range"] = X[cols].max(axis=1) - X[cols].min(axis=1)

    return X, y


# ─────────────────────────────────────────────────────────────
# 4. CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def label_value(value: float, sensor_name: str) -> str:
    """
    Rule-based agronomic state labeller.

    Assigns a state string based on THRESHOLDS for the given sensor.
    Used to generate training labels for the Decision Tree.
    Returns 'unknown' if no threshold matches.
    """
    for state, condition in THRESHOLDS.get(sensor_name, []):
        if condition(value):
            return state
    return "unknown"


def build_classifier(
    df: pd.DataFrame, sensor_name: str
) -> tuple[DecisionTreeClassifier, LabelEncoder, list]:
    """
    Train a shallow (max_depth=4) interpretable Decision Tree classifier.

    Process:
      1. Generate rule-based labels using THRESHOLDS (label_value)
      2. Encode labels with LabelEncoder
      3. Train DecisionTreeClassifier on featurised data

    max_depth=4 means the tree has at most 15 decision nodes, making
    every classification rule human-readable via export_text().

    Args:
        df:          cleaned DataFrame
        sensor_name: used to select correct thresholds

    Returns:
        clf:           trained DecisionTreeClassifier
        le:            fitted LabelEncoder (needed to decode predictions)
        feature_names: list of feature column names (for explainability)
    """
    X, _ = featurise(df)
    labels = (
        df["value"]
        .iloc[len(df) - len(X):]
        .apply(lambda v: label_value(v, sensor_name))
        .values
    )
    le  = LabelEncoder().fit(labels)
    clf = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    ).fit(X, le.transform(labels))

    return clf, le, list(X.columns)


def classify_windows(
    df: pd.DataFrame,
    sensor_name: str,
    clf: DecisionTreeClassifier,
    le: LabelEncoder,
) -> pd.DataFrame:
    """
    Apply the trained classifier to produce a 'state' column on the DataFrame.

    The first look_back rows (where we lack full lag history) get None.
    All subsequent rows get a predicted agronomic state string.
    """
    df   = df.copy()
    X, _ = featurise(df)
    pad  = len(df) - len(X)
    df["state"] = [None] * pad + list(le.inverse_transform(clf.predict(X)))
    return df


def get_current_state(df: pd.DataFrame) -> str:
    """Return the most recently classified agronomic state."""
    states = df["state"].dropna()
    return str(states.iloc[-1]) if len(states) else "unknown"


def get_state_distribution(df: pd.DataFrame) -> dict:
    """
    Return percentage of time spent in each agronomic state.
    e.g. {"optimal": 68.2, "heat_stress": 12.4, "cool": 19.4}
    """
    states = df["state"].dropna()
    if not len(states):
        return {}
    return (states.value_counts(normalize=True) * 100).round(1).to_dict()


def get_decision_rules(
    clf: DecisionTreeClassifier, feature_names: list
) -> str:
    """
    Return human-readable decision tree rules as a formatted string.
    Displayed in the dashboard's 'Decision tree rules' expandable panel.
    """
    return export_text(clf, feature_names=feature_names, max_depth=4)


# ─────────────────────────────────────────────────────────────
# 5. ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
    window: int = ANOMALY_WINDOW,
) -> pd.DataFrame:
    """
    Flag statistically anomalous readings using a rolling z-score.

    A reading is flagged when:
        |value - rolling_mean| / rolling_std > z_threshold

    z_threshold=2.5 is tuned to avoid false positives from normal
    diurnal cycles (day/night temperature swings etc.).

    Adds columns: rolling_mean, rolling_std, z_score, is_anomaly
    """
    df = df.copy()
    df["rolling_mean"] = df["value"].rolling(window=window, min_periods=3).mean()
    df["rolling_std"]  = df["value"].rolling(window=window, min_periods=3).std()
    df["z_score"]      = (
        (df["value"] - df["rolling_mean"]) / df["rolling_std"]
    ).abs()
    df["is_anomaly"]   = df["z_score"] > z_threshold
    return df


def build_alert(
    reading: dict,
    sensor_name: str,
    z_score: float,
    board_mean: float,
    kit_id: str,
) -> dict:
    """
    Construct a structured alert dict from an anomalous or stress-state reading.

    Returns a JSON-serialisable dict consumed by the /alerts endpoint
    and rendered in the frontend AlertsPanel.
    """
    value = reading["value"]
    state = label_value(value, sensor_name)
    unit  = SENSOR_UNITS.get(sensor_name, "")
    ts    = reading["created_at"]

    return {
        "kit_id":       kit_id,
        "sensor_name":  sensor_name,
        "timestamp":    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        "value":        value,
        "unit":         unit,
        "state":        state,
        "z_score":      round(float(z_score), 2),
        "delta":        round(value - board_mean, 2),
        "severity":     "high" if abs(z_score) > 3.5 else "medium",
        "description":  ALERT_DESCRIPTIONS.get(
            state, f"Unusual {sensor_name} reading: {value}{unit}"
        ),
        "triggered_by": (
            f"z-score {round(float(z_score), 2)} > threshold {ANOMALY_Z_THRESHOLD}"
        ),
    }


# ─────────────────────────────────────────────────────────────
# 6. PEER BENCHMARKING
# ─────────────────────────────────────────────────────────────

def build_peer_profile(
    kit_id: str, sensor_name: str, df: pd.DataFrame
) -> dict:
    """
    Build a serialisable statistical profile for a board+sensor combination.

    Stored in FastAPI's in-memory _peers dict (keyed by "kit_id:sensor_name").
    Used by compare_to_peers() for cross-board benchmarking.

    Returns empty dict if DataFrame is empty.
    """
    if df.empty:
        return {}
    v = df["value"].dropna()
    return {
        "kit_id":      kit_id,
        "sensor_name": sensor_name,
        "mean":        round(float(v.mean()), 3),
        "std":         round(float(v.std()), 3),
        "p25":         round(float(v.quantile(0.25)), 3),
        "p75":         round(float(v.quantile(0.75)), 3),
        "count":       int(len(v)),
        "date_from":   str(df["created_at"].min()),
        "date_to":     str(df["created_at"].max()),
        "state_dist":  get_state_distribution(df),
    }


def compare_to_peers(
    board_profile: dict, peers: list[dict]
) -> dict:
    """
    Compare one board's profile against a list of peer board profiles.

    Peers are other boards with the same sensor_name, filtered in main.py
    before being passed here.

    Returns:
        available:  bool — False if no peers or no profile
        peer_count: int
        peer_mean:  float
        delta:      float — board mean minus peer mean
        percentile: float — where this board sits (0-100)
        note:       str — plain-language comparison sentence
    """
    if not peers or not board_profile:
        return {"available": False}

    peer_means = [p["mean"] for p in peers if "mean" in p]
    if not peer_means:
        return {"available": False}

    peer_mean  = round(float(np.mean(peer_means)), 2)
    board_mean = board_profile.get("mean", 0)
    delta      = round(board_mean - peer_mean, 2)
    percentile = round(
        float(np.mean([board_mean > p for p in peer_means])) * 100, 1
    )
    n = len(peer_means)

    if abs(delta) < 0.5:
        note = f"closely matching the average of {n} similar boards"
    elif delta > 0:
        note = f"{abs(delta):.1f}° above the average of {n} similar boards (top {100-percentile:.0f}%)"
    else:
        note = f"{abs(delta):.1f}° below the average of {n} similar boards (bottom {percentile:.0f}%)"

    return {
        "available":  True,
        "peer_count": n,
        "peer_mean":  peer_mean,
        "delta":      delta,
        "percentile": percentile,
        "note":       note,
    }


# ─────────────────────────────────────────────────────────────
# 7. DEVIATION AND SUMMARISATION
# ─────────────────────────────────────────────────────────────

def compute_deviation(df: pd.DataFrame, owm: dict) -> dict:
    """
    Compare the board's sensor mean against the OWM outdoor baseline.

    Returns:
        available:    bool
        board_mean:   float
        owm_temp:     float
        delta:        float (positive = warmer than outdoor)
        note:         str plain-language description
        owm_location: str
        owm_desc:     str weather description
    """
    if not owm or df.empty:
        return {"available": False}
    owm_temp = owm.get("temp_c")
    if owm_temp is None:
        return {"available": False}

    board_mean = round(float(df["value"].mean()), 2)
    delta      = round(board_mean - owm_temp, 2)
    note = (
        "closely matching outdoor ambient" if abs(delta) < 1.0 else
        f"{abs(delta):.1f}° warmer than outdoor ambient" if delta > 0 else
        f"{abs(delta):.1f}° cooler than outdoor ambient"
    )
    return {
        "available":    True,
        "board_mean":   board_mean,
        "owm_temp":     owm_temp,
        "delta":        delta,
        "note":         note,
        "owm_location": owm.get("location", ""),
        "owm_desc":     owm.get("description", ""),
    }


def generate_summary(
    sensor_name: str,
    current_state: str,
    deviation: dict,
    peer_comparison: dict,
    state_dist: dict,
) -> str:
    """
    Compose a 2-4 sentence plain-language insight summary.

    Built from three parts:
      1. Current condition sentence (sensor+state specific template)
      2. OWM deviation sentence (if available)
      3. Peer comparison sentence (if available)
      4. Historical dominant state note (if different from current and >50%)

    Designed to be readable by non-technical community growers.
    """
    parts = [
        TEMPLATES.get(sensor_name, {}).get(
            current_state, "Sensor data processed successfully."
        )
    ]
    if deviation.get("available"):
        parts.append(f"Your readings are {deviation['note']}.")
    if peer_comparison.get("available"):
        parts.append(
            f"Compared to similar setups, your board is {peer_comparison['note']}."
        )
    if state_dist:
        dominant = max(state_dist, key=state_dist.get)
        pct = state_dist[dominant]
        if dominant != current_state and pct > 50:
            parts.append(
                f"Over the full recorded period, conditions were "
                f"'{dominant.replace('_', ' ')}' {pct:.0f}% of the time."
            )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────
# 8. FORECAST PROJECTION
# ─────────────────────────────────────────────────────────────

def compute_indoor_outdoor_delta(
    df: pd.DataFrame, owm_current: dict
) -> Optional[float]:
    """
    Compute the board's learned indoor/outdoor temperature delta.

    This is the key insight enabling forecast projection:
    if a polytunnel board consistently runs +6°C warmer than outdoor,
    we apply that delta to every OWM forecast slot to project indoor
    conditions without needing indoor forecast data.

    Returns None if OWM data is unavailable.
    """
    if not owm_current or df.empty:
        return None
    owm_temp = owm_current.get("temp_c")
    if owm_temp is None:
        return None
    return round(float(df["value"].mean()) - owm_temp, 2)


def project_forecast(
    forecast_df: pd.DataFrame,
    delta: Optional[float],
    sensor_name: str,
) -> list[dict]:
    """
    Project indoor conditions from the 5-day OWM forecast using the learned delta.

    Groups 3-hourly forecast slots by calendar day.
    For each day computes projected indoor high/low/mean by applying delta.
    Classifies each day's projected indoor mean into an agronomic state.

    Returns a list of daily forecast dicts, one per day in the forecast window.
    """
    if forecast_df.empty:
        return []

    days = []
    for date, group in forecast_df.groupby("date"):
        out_mean = float(group["temp_c"].mean())
        out_high = float(group["temp_c"].max())
        out_low  = float(group["temp_c"].min())
        d        = delta or 0.0

        in_mean  = round(out_mean + d, 1)
        in_high  = round(out_high + d, 1)
        in_low   = round(out_low  + d, 1)
        state    = label_value(in_mean, sensor_name)

        days.append({
            "date":            str(date),
            "outdoor_mean":    round(out_mean, 1),
            "outdoor_high":    round(out_high, 1),
            "outdoor_low":     round(out_low, 1),
            "indoor_mean":     in_mean,
            "indoor_high":     in_high,
            "indoor_low":      in_low,
            "delta_applied":   delta,
            "projected_state": state,
            "alert":           state in ALERT_STATES,
            "recommendation":  FORECAST_RECOMMENDATIONS.get(state, "Monitor conditions."),
            "humidity_mean":   round(float(group["humidity"].mean()), 1),
            "description":     group["description"].mode().iloc[0] if len(group) else "",
        })

    return days


# ─────────────────────────────────────────────────────────────
# 9. ORCHESTRATORS
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    kit_id:        str            = DEFAULT_KIT_ID,
    sensor_name:   str            = DEFAULT_SENSOR,
    lat:           Optional[float] = DEFAULT_LAT,
    lon:           Optional[float] = DEFAULT_LON,
    owm_api_key:   Optional[str]   = None,
    peer_profiles: Optional[list]  = None,
    look_back:     int             = 3,
) -> dict:
    """
    Full AgriSense insights pipeline for one board + sensor.

    Stages: Fetch -> Clean -> Calibrate -> Featurise ->
            Classify -> Anomaly detect -> Compare -> Summarise

    Called by: GET /kit/{kit_id}/insights in main.py

    Returns a JSON-serialisable dict containing all insight data
    needed to populate the frontend Insights and Alerts tabs.
    """
    # Fetch
    raw_df = fetch_sensor_data(kit_id, sensor_name)
    if raw_df.empty:
        return {
            "kit_id": kit_id, "sensor_name": sensor_name,
            "error": "No data returned from API",
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Clean + calibrate
    clean_df = calibrate_soil_moisture(clean(raw_df, sensor_name), sensor_name)

    # Classify
    clf, le, feature_names = build_classifier(clean_df, sensor_name)
    classified_df = classify_windows(clean_df, sensor_name, clf, le)

    # Anomaly detection
    anomaly_df = detect_anomalies(classified_df)
    recent_anomalies = [
        {
            "created_at": str(row["created_at"]),
            "value":      row["value"],
            "z_score":    round(float(row["z_score"]), 2),
            "state":      row.get("state"),
        }
        for _, row in anomaly_df[anomaly_df["is_anomaly"]].tail(10).iterrows()
    ]

    # OWM comparison
    owm = {}
    if lat and lon and owm_api_key:
        owm = fetch_owm_current(lat, lon, owm_api_key)

    # Peer + deviation + summary
    board_profile = build_peer_profile(kit_id, sensor_name, classified_df)
    deviation     = compute_deviation(classified_df, owm)
    peer_comp     = compare_to_peers(board_profile, peer_profiles or [])
    current_state = get_current_state(classified_df)
    state_dist    = get_state_distribution(classified_df)
    summary       = generate_summary(
        sensor_name, current_state, deviation, peer_comp, state_dist
    )

    return {
        "kit_id":             kit_id,
        "sensor_name":        sensor_name,
        "record_count":       len(classified_df),
        "date_from":          str(classified_df["created_at"].min()),
        "date_to":            str(classified_df["created_at"].max()),
        "current_state":      current_state,
        "state_distribution": state_dist,
        "board_profile":      board_profile,
        "deviation":          deviation,
        "peer_comparison":    peer_comp,
        "decision_rules":     get_decision_rules(clf, feature_names),
        "recent_anomalies":   recent_anomalies,
        "summary":            summary,
        "processed_at":       datetime.now(timezone.utc).isoformat(),
    }


def run_forecast_pipeline(
    kit_id:      str,
    sensor_name: str,
    lat:         float,
    lon:         float,
    owm_api_key: str,
) -> dict:
    """
    Forecast pipeline for one board + sensor.

    Stages: Fetch -> Clean -> Calibrate -> Compute delta ->
            Fetch forecast -> Project indoor conditions

    Called by: GET /kit/{kit_id}/forecast in main.py

    Uses the board's learned indoor/outdoor delta to project
    5-day indoor conditions from OWM outdoor forecast data.
    Returns day-by-day recommendations for the frontend Forecast tab.
    """
    raw_df = fetch_sensor_data(kit_id, sensor_name)
    if raw_df.empty:
        return {"error": "No board data available for forecast projection"}

    clean_df    = calibrate_soil_moisture(clean(raw_df, sensor_name), sensor_name)
    owm_current = fetch_owm_current(lat, lon, owm_api_key)
    forecast_df = fetch_owm_forecast(lat, lon, owm_api_key)
    delta       = compute_indoor_outdoor_delta(clean_df, owm_current)
    daily       = project_forecast(forecast_df, delta, sensor_name)
    alert_days  = [d for d in daily if d.get("alert")]

    return {
        "kit_id":               kit_id,
        "sensor_name":          sensor_name,
        "board_mean":           round(float(clean_df["value"].mean()), 2),
        "current_outdoor":      owm_current,
        "indoor_outdoor_delta": delta,
        "forecast":             daily,
        "alert_days":           len(alert_days),
        "forecast_summary": (
            f"{len(alert_days)} of the next {len(daily)} days "
            f"have projected stress conditions."
            if alert_days else
            f"All {len(daily)} forecast days look within normal range."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
