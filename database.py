"""
AgriSense Commons — Database Layer
=====================================
SQLite storage for multi-board sensor readings and profiles.

Tables:
  boards        — registered boards with metadata
  readings      — time-series sensor readings per board
  board_profiles — statistical snapshots per board+sensor (for peer benchmarking)

Uses Python's built-in sqlite3 — no extra dependencies needed.
Database file: agrisense.db (created automatically on first run)
"""

import sqlite3
import json
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager

DB_PATH = "agrisense.db"


# ─────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """Context manager for SQLite connections. Always closes cleanly."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────

def init_db():
    """
    Create all tables if they do not exist.
    Called once at FastAPI startup.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS boards (
                kit_id          TEXT NOT NULL,
                sensor_name     TEXT NOT NULL,
                discovered_at   TEXT NOT NULL,
                last_fetched_at TEXT,
                record_count    INTEGER DEFAULT 0,
                lat             REAL,
                lon             REAL,
                location_name   TEXT,
                active          INTEGER DEFAULT 1,
                PRIMARY KEY (kit_id, sensor_name)
            );

            CREATE TABLE IF NOT EXISTS readings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                kit_id      TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                value       REAL NOT NULL,
                UNIQUE(kit_id, sensor_name, created_at)
            );

            CREATE INDEX IF NOT EXISTS idx_readings_kit
                ON readings(kit_id, sensor_name, created_at DESC);

            CREATE TABLE IF NOT EXISTS board_profiles (
                kit_id      TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                PRIMARY KEY (kit_id, sensor_name)
            );
        """)


# ─────────────────────────────────────────────────────────────
# BOARDS
# ─────────────────────────────────────────────────────────────

def upsert_board(
    kit_id: str,
    sensor_name: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    location_name: Optional[str] = None,
):
    """Register a board+sensor combination. Safe to call repeatedly."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO boards (kit_id, sensor_name, discovered_at, lat, lon, location_name)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(kit_id, sensor_name) DO UPDATE SET
                lat           = COALESCE(excluded.lat, lat),
                lon           = COALESCE(excluded.lon, lon),
                location_name = COALESCE(excluded.location_name, location_name)
        """, (kit_id, sensor_name, now, lat, lon, location_name))


def get_all_boards() -> list[dict]:
    """Return all registered boards as a list of dicts."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM boards WHERE active = 1 ORDER BY kit_id, sensor_name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_board(kit_id: str, sensor_name: str) -> Optional[dict]:
    """Return a single board record or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM boards WHERE kit_id = ? AND sensor_name = ?",
            (kit_id, sensor_name)
        ).fetchone()
    return dict(row) if row else None


def mark_board_fetched(kit_id: str, sensor_name: str, record_count: int):
    """Update last_fetched_at and record_count after a successful sync."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            UPDATE boards
            SET last_fetched_at = ?, record_count = ?
            WHERE kit_id = ? AND sensor_name = ?
        """, (now, record_count, kit_id, sensor_name))


# ─────────────────────────────────────────────────────────────
# READINGS
# ─────────────────────────────────────────────────────────────

def save_readings(kit_id: str, sensor_name: str, df) -> int:
    """
    Bulk-insert readings from a DataFrame into the readings table.
    Skips duplicates using INSERT OR IGNORE (unique constraint on
    kit_id + sensor_name + created_at).

    Returns the number of new rows inserted.
    """
    if df.empty:
        return 0

    rows = [
        (kit_id, sensor_name, str(row["created_at"]), float(row["value"]))
        for _, row in df.iterrows()
    ]

    with get_db() as conn:
        before = conn.execute("SELECT COUNT(*) FROM readings WHERE kit_id = ? AND sensor_name = ?",
                              (kit_id, sensor_name)).fetchone()[0]
        conn.executemany(
            "INSERT OR IGNORE INTO readings (kit_id, sensor_name, created_at, value) VALUES (?, ?, ?, ?)",
            rows
        )
        after = conn.execute("SELECT COUNT(*) FROM readings WHERE kit_id = ? AND sensor_name = ?",
                             (kit_id, sensor_name)).fetchone()[0]

    return after - before


def get_readings(
    kit_id: str,
    sensor_name: str,
    limit: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve stored readings for a board+sensor.
    Optional filters: date_from, date_to (ISO strings), limit.
    Returns list of dicts with created_at and value.
    """
    query = "SELECT created_at, value FROM readings WHERE kit_id = ? AND sensor_name = ?"
    params: list = [kit_id, sensor_name]

    if date_from:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= ?"
        params.append(date_to)

    query += " ORDER BY created_at ASC"

    if limit:
        query += f" LIMIT {int(limit)}"

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(r) for r in rows]


def get_reading_count(kit_id: str, sensor_name: str) -> int:
    """Return total number of stored readings for a board+sensor."""
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM readings WHERE kit_id = ? AND sensor_name = ?",
            (kit_id, sensor_name)
        ).fetchone()[0]


def get_latest_stored_reading(kit_id: str, sensor_name: str) -> Optional[dict]:
    """Return the most recent stored reading, or None if none exist."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT created_at, value FROM readings WHERE kit_id = ? AND sensor_name = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (kit_id, sensor_name)
        ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────
# BOARD PROFILES (peer benchmarking)
# ─────────────────────────────────────────────────────────────

def save_profile(kit_id: str, sensor_name: str, profile: dict):
    """Store or update a board's statistical profile."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO board_profiles (kit_id, sensor_name, updated_at, profile_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(kit_id, sensor_name) DO UPDATE SET
                updated_at   = excluded.updated_at,
                profile_json = excluded.profile_json
        """, (kit_id, sensor_name, now, json.dumps(profile)))


def get_all_profiles(sensor_name: str) -> list[dict]:
    """Return all stored profiles for a given sensor type."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT kit_id, profile_json FROM board_profiles WHERE sensor_name = ?",
            (sensor_name,)
        ).fetchall()
    return [json.loads(r["profile_json"]) for r in rows]


def get_profile(kit_id: str, sensor_name: str) -> Optional[dict]:
    """Return a single board's stored profile or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT profile_json FROM board_profiles WHERE kit_id = ? AND sensor_name = ?",
            (kit_id, sensor_name)
        ).fetchone()
    return json.loads(row["profile_json"]) if row else None


# ─────────────────────────────────────────────────────────────
# SUMMARY / STATS
# ─────────────────────────────────────────────────────────────

def get_db_summary() -> dict:
    """Return high-level stats about what is stored in the database."""
    with get_db() as conn:
        board_count   = conn.execute("SELECT COUNT(*) FROM boards WHERE active = 1").fetchone()[0]
        reading_count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        profile_count = conn.execute("SELECT COUNT(*) FROM board_profiles").fetchone()[0]
        oldest = conn.execute("SELECT MIN(created_at) FROM readings").fetchone()[0]
        newest = conn.execute("SELECT MAX(created_at) FROM readings").fetchone()[0]

    return {
        "boards":         board_count,
        "total_readings": reading_count,
        "profiles":       profile_count,
        "oldest_reading": oldest,
        "newest_reading": newest,
    }
