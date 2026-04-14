"""
AgriSense v2 — FastAPI Backend (Multi-Board Edition)
=============================================================
Run: uvicorn main:app --reload --port 8000

New endpoints over v1:
  GET  /boards                        List all registered boards from DB
  POST /boards/discover               Trigger board discovery (seeds or range)
  POST /boards/sync/{kit_id}          Fetch + store all readings for one board
  POST /boards/sync/all               Sync all registered boards
  GET  /boards/{kit_id}/readings      Query stored readings with filters
  GET  /db/summary                    Database stats

Existing endpoints (unchanged):
  GET  /kit/{kit_id}/insights         Full pipeline (uses DB readings if available)
  GET  /kit/{kit_id}/forecast         5-day forecast
  GET  /kit/{kit_id}/alerts           Alert queue
  POST /kit/{kit_id}/alerts/check     Check for new alerts
  DEL  /kit/{kit_id}/alerts           Clear alerts
  GET  /peers                         Peer profiles (now from DB)
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from collections import deque
from datetime import datetime, timezone
import os, sys, pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (
    run_pipeline, run_forecast_pipeline,
    fetch_sensor_data, clean, calibrate_soil_moisture,
    fetch_latest_reading, build_alert,
    build_peer_profile, label_value,
    build_classifier, classify_windows, detect_anomalies,
    get_current_state, get_state_distribution, get_decision_rules,
    compute_deviation, compare_to_peers, generate_summary,
    fetch_owm_current,
    ALERT_STATES, ANOMALY_Z_THRESHOLD,
)
from database import (
    init_db, upsert_board, get_all_boards, get_board,
    mark_board_fetched, save_readings, get_readings,
    get_reading_count, save_profile, get_all_profiles,
    get_profile, get_db_summary,
)
from discovery import (
    discover_boards_from_seeds, discover_boards_by_range,
    SEED_KIT_IDS,
)
from upload import (
    init_upload_tables, queue_readings_for_upload,
    flush_upload_queue, run_board_handshake,
    is_board_reachable, get_upload_queue_summary,
)

app = FastAPI(
    title="AgriSense Commons API",
    description="Multi-board discovery, storage, and intelligent insights",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000","http://localhost:5500","http://127.0.0.1:5500","http://localhost:8080","http://127.0.0.1:8080"],
    allow_origin_regex="http://localhost:.*",
    allow_methods=["*"],
    allow_headers=["*"],
)

_alerts:    dict[str, deque] = {}
_baselines: dict[str, float] = {}

OWM_KEY    = os.getenv("OWM_API_KEY", "")
MAX_ALERTS = 50


@app.on_event("startup")
def startup():
    init_db()
    init_upload_tables()


def _key(kit_id: str, sensor_name: str) -> str:
    return f"{kit_id}:{sensor_name}"


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "AgriSense Commons API", "version": "2.1.0"}


# ── Board discovery ────────────────────────────────────────────────────────────

class DiscoverRequest(BaseModel):
    mode:     str                   = "seeds"
    seeds:    Optional[list[str]]   = None
    start_id: Optional[int]         = 1000
    end_id:   Optional[int]         = 1020


@app.get("/boards")
def list_boards():
    """Return all boards registered in the database."""
    boards = get_all_boards()
    return {"count": len(boards), "boards": boards}


@app.post("/boards/discover")
def trigger_discovery(body: DiscoverRequest, background_tasks: BackgroundTasks):
    """
    Discover public TeleAgriCulture boards in the background.

    mode='seeds'  — probe a known list of Kit IDs (fast, ~30 seconds)
    mode='range'  — scan a numeric range of Kit IDs (slower, ~2 min per 50 boards)

    Poll GET /boards to see results as they come in.
    """
    if body.mode == "seeds":
        seeds = body.seeds or SEED_KIT_IDS
        background_tasks.add_task(discover_boards_from_seeds, seeds)
        return {"status": "started", "mode": "seeds", "probing": seeds}

    if body.mode == "range":
        start = body.start_id or 1000
        end   = body.end_id   or 1020
        if end - start > 100:
            raise HTTPException(status_code=400,
                                detail="Range too large — maximum 100 Kit IDs per scan.")
        background_tasks.add_task(discover_boards_by_range, start, end)
        return {"status": "started", "mode": "range",
                "scanning": f"Kit {start} to {end}"}

    raise HTTPException(status_code=400, detail="mode must be 'seeds' or 'range'")


# ── Board sync ─────────────────────────────────────────────────────────────────

def _sync_board(kit_id: str, sensor_name: str) -> dict:
    """Fetch all readings from the API and store new ones in the database."""
    try:
        raw_df = fetch_sensor_data(kit_id, sensor_name)
        if raw_df.empty:
            return {"kit_id": kit_id, "sensor_name": sensor_name,
                    "status": "no_data", "new_readings": 0}

        clean_df  = calibrate_soil_moisture(clean(raw_df, sensor_name), sensor_name)
        new_count = save_readings(kit_id, sensor_name, clean_df)

        upsert_board(kit_id, sensor_name)
        mark_board_fetched(kit_id, sensor_name, len(clean_df))

        profile = build_peer_profile(kit_id, sensor_name, clean_df)
        if profile:
            save_profile(kit_id, sensor_name, profile)
            _baselines[_key(kit_id, sensor_name)] = profile.get("mean", 0)

        return {
            "kit_id":       kit_id,
            "sensor_name":  sensor_name,
            "status":       "ok",
            "total_stored": get_reading_count(kit_id, sensor_name),
            "new_readings": new_count,
            "synced_at":    datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"kit_id": kit_id, "sensor_name": sensor_name,
                "status": "error", "detail": str(e)}


@app.post("/boards/sync/{kit_id}")
def sync_board(kit_id: str, sensor_name: str = Query(default="ftTemp")):
    """
    Fetch and store all readings for one board+sensor.
    Takes 10-30 seconds for boards with many readings.
    """
    upsert_board(kit_id, sensor_name)
    result = _sync_board(kit_id, sensor_name)
    if result["status"] == "error":
        raise HTTPException(status_code=502, detail=result["detail"])
    return result


@app.post("/boards/sync/all")
def sync_all_boards(background_tasks: BackgroundTasks):
    """Sync all registered boards in the background."""
    boards = get_all_boards()
    if not boards:
        return {"message": "No boards registered. Run POST /boards/discover first."}

    def _run():
        for b in boards:
            _sync_board(b["kit_id"], b["sensor_name"])

    background_tasks.add_task(_run)
    return {"status": "started", "boards": len(boards)}


# ── Readings query ─────────────────────────────────────────────────────────────

@app.get("/boards/{kit_id}/readings")
def get_board_readings(
    kit_id:      str,
    sensor_name: str            = Query(default="ftTemp"),
    limit:       Optional[int]  = Query(default=None),
    date_from:   Optional[str]  = Query(default=None),
    date_to:     Optional[str]  = Query(default=None),
):
    """
    Return stored readings from the database.
    Filters: limit, date_from, date_to (ISO date strings).

    Example: /boards/1001/readings?sensor_name=ftTemp&date_from=2019-08-01&limit=100
    """
    if not get_board(kit_id, sensor_name):
        raise HTTPException(
            status_code=404,
            detail=f"No stored readings for {kit_id}/{sensor_name}. "
                   f"Call POST /boards/sync/{kit_id}?sensor_name={sensor_name} first."
        )
    readings = get_readings(kit_id, sensor_name,
                            limit=limit, date_from=date_from, date_to=date_to)
    return {"kit_id": kit_id, "sensor_name": sensor_name,
            "count": len(readings), "readings": readings}


# ── Insights ───────────────────────────────────────────────────────────────────

@app.get("/kit/{kit_id}/insights")
def get_insights(
    kit_id:      str,
    sensor_name: str            = Query(default="ftTemp"),
    lat:         Optional[float] = Query(default=47.7981),
    lon:         Optional[float] = Query(default=13.0456),
    use_db:      bool            = Query(default=True),
):
    """
    Full AgriSense pipeline. Uses stored DB readings when available (faster).
    Falls back to live API fetch if no readings are stored.
    Peer profiles are loaded from the database across all registered boards.
    """
    k = _key(kit_id, sensor_name)
    peer_profiles = [p for p in get_all_profiles(sensor_name) if p.get("kit_id") != kit_id]

    if use_db and get_reading_count(kit_id, sensor_name) > 0:
        stored = get_readings(kit_id, sensor_name)
        df = pd.DataFrame(stored)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["value"]      = df["value"].astype(float)

        clf, le, fnames = build_classifier(df, sensor_name)
        classified_df   = classify_windows(df, sensor_name, clf, le)
        anomaly_df      = detect_anomalies(classified_df)

        recent_anomalies = [
            {"created_at": str(r["created_at"]), "value": r["value"],
             "z_score": round(float(r["z_score"]), 2), "state": r.get("state")}
            for _, r in anomaly_df[anomaly_df["is_anomaly"]].tail(10).iterrows()
        ]

        owm           = fetch_owm_current(lat, lon, OWM_KEY) if (lat and lon and OWM_KEY) else {}
        board_profile = build_peer_profile(kit_id, sensor_name, classified_df)
        deviation     = compute_deviation(classified_df, owm)
        peer_comp     = compare_to_peers(board_profile, peer_profiles)
        current_state = get_current_state(classified_df)
        state_dist    = get_state_distribution(classified_df)
        summary       = generate_summary(sensor_name, current_state,
                                         deviation, peer_comp, state_dist)

        if board_profile:
            save_profile(kit_id, sensor_name, board_profile)
            _baselines[k] = board_profile.get("mean", 0)

        return {
            "kit_id":             kit_id,
            "sensor_name":        sensor_name,
            "source":             "database",
            "record_count":       len(classified_df),
            "date_from":          str(classified_df["created_at"].min()),
            "date_to":            str(classified_df["created_at"].max()),
            "current_state":      current_state,
            "state_distribution": state_dist,
            "board_profile":      board_profile,
            "deviation":          deviation,
            "peer_comparison":    peer_comp,
            "decision_rules":     get_decision_rules(clf, fnames),
            "recent_anomalies":   recent_anomalies,
            "summary":            summary,
            "processed_at":       datetime.now(timezone.utc).isoformat(),
        }

    try:
        result = run_pipeline(
            kit_id=kit_id, sensor_name=sensor_name,
            lat=lat, lon=lon,
            owm_api_key=OWM_KEY or None,
            peer_profiles=peer_profiles or None,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    result["source"] = "api"
    if result.get("board_profile"):
        save_profile(kit_id, sensor_name, result["board_profile"])
        _baselines[k] = result["board_profile"].get("mean", 0)

    return result


# ── Forecast ───────────────────────────────────────────────────────────────────

@app.get("/kit/{kit_id}/forecast")
def get_forecast(
    kit_id:      str,
    sensor_name: str            = Query(default="ftTemp"),
    lat:         Optional[float] = Query(default=47.7981),
    lon:         Optional[float] = Query(default=13.0456),
):
    if not OWM_KEY:
        raise HTTPException(status_code=400, detail="OWM_API_KEY not set in .env")
    try:
        result = run_forecast_pipeline(
            kit_id=kit_id, sensor_name=sensor_name,
            lat=lat, lon=lon, owm_api_key=OWM_KEY,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Alerts ─────────────────────────────────────────────────────────────────────

@app.get("/kit/{kit_id}/alerts")
def get_alerts(kit_id: str, sensor_name: str = Query(default="ftTemp")):
    k = _key(kit_id, sensor_name)
    return {"kit_id": kit_id, "sensor_name": sensor_name,
            "count": len(_alerts.get(k, [])),
            "alerts": list(_alerts.get(k, deque()))}


@app.post("/kit/{kit_id}/alerts/check")
def check_alerts(kit_id: str, sensor_name: str = Query(default="ftTemp")):
    k          = _key(kit_id, sensor_name)
    board_mean = _baselines.get(k)

    if board_mean is None:
        profile = get_profile(kit_id, sensor_name)
        if profile:
            board_mean = profile.get("mean")
            _baselines[k] = board_mean

    if board_mean is None:
        return {"message": "No baseline. Call /kit/{kit_id}/insights first."}

    reading = fetch_latest_reading(kit_id, sensor_name)
    if not reading:
        return {"message": "Could not fetch latest reading."}

    value     = reading["value"]
    state     = label_value(value, sensor_name)
    profile   = get_profile(kit_id, sensor_name) or {}
    board_std = profile.get("std", 1.0) or 1.0
    z_approx  = abs(value - board_mean) / max(board_std, 0.1)

    if state in ALERT_STATES or z_approx > ANOMALY_Z_THRESHOLD:
        alert = build_alert(reading, sensor_name, z_approx, board_mean, kit_id)
        if k not in _alerts:
            _alerts[k] = deque(maxlen=MAX_ALERTS)
        _alerts[k].appendleft(alert)
        return {"new_alert": True, "alert": alert}

    return {"new_alert": False, "latest_value": value, "state": state,
            "z_score": round(z_approx, 2),
            "checked_at": datetime.now(timezone.utc).isoformat()}


@app.delete("/kit/{kit_id}/alerts")
def clear_alerts(kit_id: str, sensor_name: str = Query(default="ftTemp")):
    _alerts.pop(_key(kit_id, sensor_name), None)
    return {"cleared": True}


# ── DB stats ───────────────────────────────────────────────────────────────────

@app.get("/db/summary")
def db_summary():
    return get_db_summary()


@app.get("/peers")
def list_peers(sensor_name: str = Query(default="ftTemp")):
    profiles = get_all_profiles(sensor_name)
    return {"count": len(profiles), "sensor_name": sensor_name, "profiles": profiles}


# ─────────────────────────────────────────────────────────────
# UPLOAD / HANDSHAKE ENDPOINTS
# ─────────────────────────────────────────────────────────────

class HandshakeRequest(BaseModel):
    kit_id:    str
    api_token: Optional[str] = None


class FlushRequest(BaseModel):
    kit_id:      str
    sensor_name: str = "ftTemp"
    api_token:   str


@app.get("/upload/board/status")
def board_ap_status():
    """
    Check whether the TeleAgriCulture board's WiFi AP is reachable.
    Only returns True when your device is connected to the board's WiFi
    (SSID: TeleAgriCulture Board, password: enter123).
    """
    reachable = is_board_reachable()
    return {
        "reachable": reachable,
        "board_ip":  "192.168.4.1",
        "message": (
            "Board AP is reachable. You can run the handshake."
            if reachable else
            "Board not reachable. Connect to the board's WiFi AP first."
        ),
    }


@app.post("/upload/handshake")
def board_handshake(body: HandshakeRequest):
    """
    Run the WiFi handshake workflow.

    How to use:
      1. Connect your phone or laptop to the board's WiFi AP
         (SSID: TeleAgriCulture Board, password: enter123)
      2. Hit this endpoint with your Kit ID (and optionally your API token)
      3. AgriSense pulls the current reading from the board at 192.168.4.1,
         caches it locally, and — if a token is provided and internet is
         available — uploads it to kits.teleagriculture.org immediately
      4. If no internet yet, the reading stays queued and can be flushed
         later via POST /upload/flush once back online

    This is the 'cache and upload when next online' pattern described
    in the project feedback.
    """
    result = run_board_handshake(
        kit_id=body.kit_id,
        api_token=body.api_token,
    )
    return result


@app.post("/upload/flush")
def flush_to_teleagriculture(body: FlushRequest):
    """
    Upload all locally cached readings for a board to kits.teleagriculture.org.

    Use this when you are back online after a period of local-only collection.
    Requires the board's API token from kits.teleagriculture.org.

    The upload queue persists across server restarts in SQLite, so readings
    collected days ago will still be here waiting to upload.
    """
    result = flush_upload_queue(
        kit_id=body.kit_id,
        sensor_name=body.sensor_name,
        api_token=body.api_token,
    )
    return result


@app.post("/upload/queue/{kit_id}")
def queue_stored_readings(
    kit_id:      str,
    sensor_name: str = Query(default="ftTemp"),
):
    """
    Queue all locally stored readings for upload to TeleAgriCulture.
    Use this to queue readings that were synced via /boards/sync
    (pulled from API into local DB) back up to the platform.

    Then call POST /upload/flush with your API token to send them.
    """
    from database import get_readings
    readings = get_readings(kit_id, sensor_name)
    if not readings:
        raise HTTPException(
            status_code=404,
            detail=f"No stored readings for {kit_id}/{sensor_name}. "
                   f"Call POST /boards/sync/{kit_id} first."
        )
    queued = queue_readings_for_upload(kit_id, sensor_name, readings)
    return {
        "kit_id":      kit_id,
        "sensor_name": sensor_name,
        "queued":      queued,
        "message":     f"{queued} readings added to upload queue. "
                       f"Call POST /upload/flush with your API token to send them.",
    }


@app.get("/upload/queue/{kit_id}")
def get_queue_status(
    kit_id:      str,
    sensor_name: str = Query(default="ftTemp"),
):
    """Return the current state of the upload queue for a board+sensor."""
    summary = get_upload_queue_summary(kit_id, sensor_name)
    return {
        "kit_id":      kit_id,
        "sensor_name": sensor_name,
        **summary,
    }
