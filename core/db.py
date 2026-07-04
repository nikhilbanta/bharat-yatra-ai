"""
SQLite persistence layer.

Identity model: by explicit product decision, users are identified purely by
a self-chosen name/email string — there is no password. This is intentionally
simple for a demo, and the UI must make that tradeoff visible to the user
(see app.py disclaimer). No sensitive data beyond the optional profile fields
below is ever collected or stored.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    identifier TEXT PRIMARY KEY,
    name TEXT,
    age_band TEXT,
    gender TEXT,
    location TEXT,
    companions TEXT,
    interests TEXT,
    budget TEXT,
    language_pref TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    query_input TEXT,
    ai_output TEXT,
    created_at TEXT,
    FOREIGN KEY (identifier) REFERENCES profiles (identifier)
);
"""


@contextmanager
def get_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_profile(identifier: str, profile: dict, db_path: str = DB_PATH) -> None:
    """Insert or update a user's profile. `identifier` is a normalized name/email."""
    identifier = identifier.strip().lower()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO profiles (identifier, name, age_band, gender, location,
                                   companions, interests, budget, language_pref, updated_at)
            VALUES (:identifier, :name, :age_band, :gender, :location,
                    :companions, :interests, :budget, :language_pref, :updated_at)
            ON CONFLICT(identifier) DO UPDATE SET
                name=excluded.name,
                age_band=excluded.age_band,
                gender=excluded.gender,
                location=excluded.location,
                companions=excluded.companions,
                interests=excluded.interests,
                budget=excluded.budget,
                language_pref=excluded.language_pref,
                updated_at=excluded.updated_at
            """,
            {
                "identifier": identifier,
                "name": profile.get("name", ""),
                "age_band": profile.get("age_band", ""),
                "gender": profile.get("gender", ""),
                "location": profile.get("location", ""),
                "companions": profile.get("companions", ""),
                "interests": json.dumps(profile.get("interests", [])),
                "budget": profile.get("budget", ""),
                "language_pref": profile.get("language_pref", "English"),
                "updated_at": _now(),
            },
        )


def get_profile(identifier: str, db_path: str = DB_PATH) -> dict | None:
    identifier = identifier.strip().lower()
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM profiles WHERE identifier = ?", (identifier,)
        ).fetchone()
        if row is None:
            return None
        profile = dict(row)
        profile["interests"] = json.loads(profile.get("interests") or "[]")
        return profile


def add_history_entry(
    identifier: str, feature_type: str, query_input: str, ai_output: str, db_path: str = DB_PATH
) -> None:
    identifier = identifier.strip().lower()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO history (identifier, feature_type, query_input, ai_output, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (identifier, feature_type, query_input, ai_output, _now()),
        )


def get_history(identifier: str, limit: int = 20, db_path: str = DB_PATH) -> list[dict]:
    identifier = identifier.strip().lower()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT feature_type, query_input, ai_output, created_at
            FROM history WHERE identifier = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (identifier, limit),
        ).fetchall()
        return [dict(r) for r in rows]
