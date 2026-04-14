"""
AgriSense — Board Discovery
======================================
Discovers all public TeleAgriCulture boards by probing the API.

The TeleAgriCulture API does not have a public /kits index endpoint
that lists all boards without auth. Instead we use two strategies:

Strategy 1 — Range scan:
  Probe Kit IDs sequentially (e.g. 1000-1100). If a board responds
  with data, it is public and active. This is the most reliable approach
  since community boards typically use sequential IDs.

Strategy 2 — Known seeds:
  Start from a list of known active Kit IDs (e.g. Kit 1001 from Schmiede)
  and expand via any cross-references found in API responses.

Both strategies probe a standard set of sensor names per board.
Results are stored in the database via database.upsert_board().
"""

import requests
import time
from typing import Optional
from database import upsert_board, get_all_boards

TELEAGRI_BASE = "https://kits.teleagriculture.org/api/kits"

# Sensor names to probe per board — most common on TeleAgriCulture boards
PROBE_SENSORS = [
    "ftTemp",
    "ftHumid",
    "ftSoilMoist",
    "ftMoisture",
    "ftLight",
    "ftTDS",
    "ftSoilTemp",
    "ftPressure",
    "ftADC",
]

# Known active boards as starting seeds
SEED_KIT_IDS = ["1001", "1002", "1003", "1004", "1005"]


def probe_board(kit_id: str, sensor_name: str, timeout: int = 8) -> bool:
    """
    Check if a board+sensor combination has any data.
    Returns True if the API returns at least one reading, False otherwise.
    Does not raise — all errors return False.
    """
    try:
        url = f"{TELEAGRI_BASE}/{kit_id}/{sensor_name}/measurements"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False
        data = r.json()
        return isinstance(data.get("data"), list) and len(data["data"]) > 0
    except Exception:
        return False


def discover_sensors_for_kit(kit_id: str) -> list[str]:
    """
    Probe all PROBE_SENSORS for a given kit_id.
    Returns list of sensor names that have data.
    """
    found = []
    for sensor in PROBE_SENSORS:
        if probe_board(kit_id, sensor):
            found.append(sensor)
        time.sleep(0.3)   # be polite to the API
    return found


def discover_boards_by_range(
    start_id: int = 1000,
    end_id: int = 1050,
    on_found=None,
) -> list[dict]:
    """
    Scan a range of Kit IDs to find active public boards.

    Args:
        start_id:  first Kit ID to probe (inclusive)
        end_id:    last Kit ID to probe (inclusive)
        on_found:  optional callback called with (kit_id, sensors) when a board is found

    Returns list of dicts: [{kit_id, sensors}]

    Note: a range of 50 boards takes roughly 2-3 minutes due to API rate limiting.
    For larger scans, run this as a background task.
    """
    found_boards = []

    for kit_id_int in range(start_id, end_id + 1):
        kit_id = str(kit_id_int)

        # First check if the board exists at all with a single fast probe
        if not probe_board(kit_id, "ftTemp"):
            # Try one more sensor before giving up on this ID
            if not probe_board(kit_id, "ftHumid"):
                time.sleep(0.2)
                continue

        # Board exists — probe all sensors
        sensors = discover_sensors_for_kit(kit_id)

        if sensors:
            entry = {"kit_id": kit_id, "sensors": sensors}
            found_boards.append(entry)

            # Register in database
            for sensor in sensors:
                upsert_board(kit_id, sensor)

            if on_found:
                on_found(kit_id, sensors)

    return found_boards


def discover_boards_from_seeds(seeds: Optional[list[str]] = None) -> list[dict]:
    """
    Probe a list of known seed Kit IDs to discover which sensors they have.
    Faster than range scan when you already know some Kit IDs.

    Args:
        seeds: list of Kit ID strings. Defaults to SEED_KIT_IDS.

    Returns list of dicts: [{kit_id, sensors}]
    """
    seeds = seeds or SEED_KIT_IDS
    found_boards = []

    for kit_id in seeds:
        sensors = discover_sensors_for_kit(kit_id)
        if sensors:
            entry = {"kit_id": kit_id, "sensors": sensors}
            found_boards.append(entry)
            for sensor in sensors:
                upsert_board(kit_id, sensor)

    return found_boards


def get_registered_boards() -> list[dict]:
    """Return all boards already stored in the database."""
    return get_all_boards()
