"""
AgriSense — Cache and Upload Layer
============================================
Addresses the feedback: boards need to upload data to the TeleAgriCulture
SQL/PHP server for wider scientific use, not just store locally.

The feedback specifically suggested a WiFi handshake approach:
  - The board already creates its own WiFi AP during setup (SSID: TeleAgriCulture Board)
  - That same WiFi channel can be used to pull cached readings off the board
  - AgriSense can then relay those readings to kits.teleagriculture.org

This module handles three things:

1. LOCAL BOARD CACHE PULL
   When a phone/laptop connects to the board's WiFi AP (192.168.4.1),
   it can pull locally stored readings (if the board has SD card logging enabled).
   AgriSense polls the board's local dashboard endpoint and caches what it finds.

2. TELEAGRICULTURE UPLOAD RELAY
   AgriSense acts as an upload relay: readings cached locally (either from
   the board cache pull or from our SQLite DB) are uploaded to the
   TeleAgriCulture platform using the board's API token.
   This solves the offline gap — board collects data → phone picks it up
   over local WiFi → phone relays it to the cloud when back online.

3. UPLOAD QUEUE
   An in-memory + SQLite queue tracks which readings have been uploaded
   and which are pending, so nothing gets sent twice.

TeleAgriCulture upload endpoint (from API docs and board firmware):
  POST https://kits.teleagriculture.org/api/kits/{kit_id}/measurements
  Headers: Authorization: Bearer {api_token}
  Body:    { sensor_name, value, created_at }

Board local AP endpoints (when connected to board WiFi 192.168.4.1):
  GET  /                    — board dashboard HTML
  GET  /data                — live sensor JSON (if available)
  The board does NOT expose a structured readings history endpoint locally —
  local data is shown in the AP dashboard UI only, not as a JSON API.
  SD card data is accessible via the board's local file browser if enabled.
"""

import requests
import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager

BOARD_LOCAL_IP    = "http://192.168.4.1"
TELEAGRI_BASE     = "https://kits.teleagriculture.org/api/kits"
DB_PATH           = "agrisense.db"   # shared with database.py


# ─────────────────────────────────────────────────────────────
# UPLOAD QUEUE — tracked in SQLite
# ─────────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_upload_tables():
    """
    Create the upload tracking table if it doesn't exist.
    Called at startup alongside database.init_db().
    """
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS upload_queue (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                kit_id      TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                value       REAL NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                attempted_at TEXT,
                error       TEXT,
                UNIQUE(kit_id, sensor_name, created_at)
            );

            CREATE INDEX IF NOT EXISTS idx_upload_queue_status
                ON upload_queue(kit_id, sensor_name, status);
        """)


def queue_readings_for_upload(kit_id: str, sensor_name: str, readings: list[dict]) -> int:
    """
    Add readings to the upload queue. Skips duplicates.
    readings: list of dicts with keys created_at and value.
    Returns number of new items queued.
    """
    if not readings:
        return 0

    rows = [
        (kit_id, sensor_name, str(r["created_at"]), float(r["value"]))
        for r in readings
    ]

    with _db() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM upload_queue WHERE kit_id=? AND sensor_name=? AND status='pending'",
            (kit_id, sensor_name)
        ).fetchone()[0]

        conn.executemany("""
            INSERT OR IGNORE INTO upload_queue (kit_id, sensor_name, created_at, value, status)
            VALUES (?, ?, ?, ?, 'pending')
        """, rows)

        after = conn.execute(
            "SELECT COUNT(*) FROM upload_queue WHERE kit_id=? AND sensor_name=? AND status='pending'",
            (kit_id, sensor_name)
        ).fetchone()[0]

    return after - before


def get_pending_uploads(kit_id: str, sensor_name: str, limit: int = 100) -> list[dict]:
    """Return pending upload queue items for a board, oldest first."""
    with _db() as conn:
        rows = conn.execute("""
            SELECT id, created_at, value
            FROM upload_queue
            WHERE kit_id=? AND sensor_name=? AND status='pending'
            ORDER BY created_at ASC
            LIMIT ?
        """, (kit_id, sensor_name, limit)).fetchall()
    return [dict(r) for r in rows]


def mark_uploaded(item_ids: list[int]):
    """Mark upload queue items as successfully uploaded."""
    if not item_ids:
        return
    placeholders = ",".join("?" * len(item_ids))
    with _db() as conn:
        conn.execute(f"""
            UPDATE upload_queue SET status='uploaded', attempted_at=?
            WHERE id IN ({placeholders})
        """, [datetime.now(timezone.utc).isoformat()] + item_ids)


def mark_upload_failed(item_ids: list[int], error: str):
    """Mark upload queue items as failed with an error message."""
    if not item_ids:
        return
    placeholders = ",".join("?" * len(item_ids))
    with _db() as conn:
        conn.execute(f"""
            UPDATE upload_queue SET status='failed', attempted_at=?, error=?
            WHERE id IN ({placeholders})
        """, [datetime.now(timezone.utc).isoformat(), error] + item_ids)


def get_upload_queue_summary(kit_id: str, sensor_name: str) -> dict:
    """Return counts of pending/uploaded/failed items for a board."""
    with _db() as conn:
        rows = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM upload_queue
            WHERE kit_id=? AND sensor_name=?
            GROUP BY status
        """, (kit_id, sensor_name)).fetchall()
    summary = {r["status"]: r["count"] for r in rows}
    return {
        "pending":  summary.get("pending", 0),
        "uploaded": summary.get("uploaded", 0),
        "failed":   summary.get("failed", 0),
    }


# ─────────────────────────────────────────────────────────────
# BOARD LOCAL WIFI PULL
# ─────────────────────────────────────────────────────────────

def pull_from_board_ap(timeout: int = 5) -> Optional[dict]:
    """
    Attempt to pull live sensor data from the board's local WiFi AP.

    The TeleAgriCulture Board V2.1 serves a live dashboard at 192.168.4.1
    when in AP mode (SSID: TeleAgriCulture Board / Teleagriculture DB).
    This attempts to pull the current sensor reading from that local endpoint.

    IMPORTANT: This only works when your device is connected to the board's
    WiFi AP — i.e. the same WiFi used during initial setup. It will fail
    (timeout) when connected to normal internet WiFi. This is by design:
    the handshake happens intentionally when you walk up to the board.

    Returns dict with sensor readings if successful, None otherwise.
    """
    try:
        r = requests.get(f"{BOARD_LOCAL_IP}/data", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    # Some board firmware versions serve data at the root
    try:
        r = requests.get(BOARD_LOCAL_IP, timeout=timeout)
        if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
            return r.json()
    except Exception:
        pass

    return None


def is_board_reachable() -> bool:
    """
    Check if the board's local AP is reachable.
    Returns True if device is connected to the board's WiFi network.
    """
    try:
        r = requests.get(BOARD_LOCAL_IP, timeout=3)
        return r.status_code in (200, 301, 302)
    except Exception:
        return False


def parse_board_ap_reading(raw: dict, kit_id: str) -> list[dict]:
    """
    Parse whatever the board's local endpoint returns into a standard
    list of {sensor_name, value, created_at, kit_id} dicts.

    The board firmware returns different structures depending on version —
    this handles the most common patterns seen in TeleAgriCulture V2.1.
    """
    now = datetime.now(timezone.utc).isoformat()
    readings = []

    # Pattern 1: {"ftTemp": 18.5, "ftHumid": 62.1, ...}
    sensor_keys = {"ftTemp", "ftHumid", "ftSoilMoist", "ftMoisture",
                   "ftLight", "ftTDS", "ftSoilTemp", "ftPressure", "ftADC"}

    for key in sensor_keys:
        if key in raw and raw[key] is not None:
            try:
                readings.append({
                    "kit_id":      kit_id,
                    "sensor_name": key,
                    "value":       float(raw[key]),
                    "created_at":  raw.get("timestamp", now),
                })
            except (ValueError, TypeError):
                pass

    # Pattern 2: {"sensors": [{"name": "ftTemp", "value": 18.5}, ...]}
    if not readings and "sensors" in raw:
        for s in raw["sensors"]:
            try:
                readings.append({
                    "kit_id":      kit_id,
                    "sensor_name": s["name"],
                    "value":       float(s["value"]),
                    "created_at":  s.get("timestamp", now),
                })
            except (KeyError, ValueError, TypeError):
                pass

    return readings


# ─────────────────────────────────────────────────────────────
# TELEAGRICULTURE PLATFORM UPLOAD
# ─────────────────────────────────────────────────────────────

def upload_reading_to_teleagri(
    kit_id: str,
    sensor_name: str,
    value: float,
    created_at: str,
    api_token: str,
) -> bool:
    """
    Upload a single reading to the TeleAgriCulture platform.

    Uses the authenticated POST endpoint that the board firmware itself uses.
    Requires the board's API token from kits.teleagriculture.org.

    Args:
        kit_id:      board Kit ID
        sensor_name: sensor identifier (e.g. "ftTemp")
        value:       numeric sensor reading
        created_at:  ISO timestamp string
        api_token:   TeleAgriCulture API token for this board

    Returns True on success, False on failure.
    """
    try:
        url = f"{TELEAGRI_BASE}/{kit_id}/{sensor_name}/measurements"
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type":  "application/json",
            },
            json={
                "value":      value,
                "created_at": created_at,
            },
            timeout=15,
        )
        return r.status_code in (200, 201)
    except Exception:
        return False


def flush_upload_queue(
    kit_id: str,
    sensor_name: str,
    api_token: str,
    batch_size: int = 50,
) -> dict:
    """
    Upload all pending readings for a board to the TeleAgriCulture platform.

    Works in batches of batch_size to avoid overwhelming the API.
    Marks each reading as uploaded or failed in the local queue.

    Returns a result dict with counts of uploaded and failed items.

    This is the key function for the offline-then-sync workflow:
      1. Board collects readings (on-device or SD card)
      2. AgriSense pulls them locally (via AP WiFi or from its own DB)
      3. This function uploads them all to kits.teleagriculture.org
      4. Data is now in the TeleAgriCulture SQL database for wider science use
    """
    pending = get_pending_uploads(kit_id, sensor_name, limit=batch_size)
    if not pending:
        return {"uploaded": 0, "failed": 0, "message": "Nothing pending"}

    uploaded_ids = []
    failed_ids   = []
    last_error   = ""

    for item in pending:
        success = upload_reading_to_teleagri(
            kit_id=kit_id,
            sensor_name=sensor_name,
            value=item["value"],
            created_at=item["created_at"],
            api_token=api_token,
        )
        if success:
            uploaded_ids.append(item["id"])
        else:
            failed_ids.append(item["id"])
            last_error = "Upload rejected by TeleAgriCulture API"

    if uploaded_ids:
        mark_uploaded(uploaded_ids)
    if failed_ids:
        mark_upload_failed(failed_ids, last_error)

    return {
        "uploaded":   len(uploaded_ids),
        "failed":     len(failed_ids),
        "remaining":  get_upload_queue_summary(kit_id, sensor_name)["pending"],
        "message":    f"Uploaded {len(uploaded_ids)}, failed {len(failed_ids)}",
    }


# ─────────────────────────────────────────────────────────────
# COMBINED HANDSHAKE WORKFLOW
# ─────────────────────────────────────────────────────────────

def run_board_handshake(
    kit_id: str,
    api_token: Optional[str] = None,
) -> dict:
    """
    Run the full WiFi handshake workflow described in the project feedback.

    This is the intended flow:
      1. User connects their phone/laptop to the board's WiFi AP
         (SSID: TeleAgriCulture Board, password: enter123)
      2. AgriSense detects the board is reachable at 192.168.4.1
      3. Pulls the current sensor reading(s) from the board
      4. Caches them in the local SQLite database
      5. Queues them for upload to kits.teleagriculture.org
      6. If api_token is provided AND internet is available, flushes the queue now
         (the board is connected to the phone's hotspot / internet)
      7. Returns a summary of what was collected and uploaded

    If internet is not available at step 6, the queue persists in SQLite
    and can be flushed later by calling flush_upload_queue() or hitting
    POST /upload/flush in the API.

    Args:
        kit_id:    the board's Kit ID from kits.teleagriculture.org
        api_token: TeleAgriCulture API token (optional — skips upload if absent)

    Returns a summary dict.
    """
    result = {
        "board_reachable": False,
        "readings_pulled": 0,
        "readings_queued": 0,
        "uploaded":        0,
        "failed":          0,
        "message":         "",
    }

    # Step 1: Check board is reachable
    if not is_board_reachable():
        result["message"] = (
            "Board not reachable at 192.168.4.1. "
            "Make sure you are connected to the board's WiFi AP "
            "(SSID: TeleAgriCulture Board, password: enter123)."
        )
        return result

    result["board_reachable"] = True

    # Step 2: Pull readings from board
    raw = pull_from_board_ap()
    if not raw:
        result["message"] = (
            "Board is reachable but returned no sensor data. "
            "The board may not expose a /data endpoint in this firmware version. "
            "SD card data can be accessed via the board dashboard at 192.168.4.1."
        )
        return result

    # Step 3: Parse readings
    readings = parse_board_ap_reading(raw, kit_id)
    result["readings_pulled"] = len(readings)

    if not readings:
        result["message"] = "Board responded but no recognisable sensor values found."
        return result

    # Step 4: Cache in local DB and queue for upload
    from database import save_readings
    import pandas as pd

    for sensor_group in _group_by_sensor(readings):
        sensor_name = sensor_group[0]["sensor_name"]
        df = pd.DataFrame([
            {"created_at": r["created_at"], "value": r["value"]}
            for r in sensor_group
        ])
        df["created_at"] = pd.to_datetime(df["created_at"])
        save_readings(kit_id, sensor_name, df)

        queued = queue_readings_for_upload(kit_id, sensor_name, sensor_group)
        result["readings_queued"] += queued

    # Step 5: Flush queue if token provided
    if api_token:
        sensor_names = list({r["sensor_name"] for r in readings})
        for sensor_name in sensor_names:
            flush_result = flush_upload_queue(kit_id, sensor_name, api_token)
            result["uploaded"] += flush_result["uploaded"]
            result["failed"]   += flush_result["failed"]
        result["message"] = (
            f"Handshake complete. {result['readings_pulled']} readings pulled, "
            f"{result['uploaded']} uploaded to TeleAgriCulture."
        )
    else:
        result["message"] = (
            f"Handshake complete. {result['readings_pulled']} readings pulled and cached. "
            f"Provide an API token to upload to TeleAgriCulture, or call POST /upload/flush later."
        )

    return result


def _group_by_sensor(readings: list[dict]) -> list[list[dict]]:
    """Group a flat list of readings by sensor_name."""
    groups: dict[str, list] = {}
    for r in readings:
        groups.setdefault(r["sensor_name"], []).append(r)
    return list(groups.values())
