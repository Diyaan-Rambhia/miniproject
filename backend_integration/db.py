"""
Database Access Layer using Standard Library sqlite3
"""

import os
import json
import sqlite3
from typing import Generator
from backend_integration.config import DB_FILE


def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            threat_score REAL NOT NULL,
            transformer_confidence REAL NOT NULL,
            vae_anomaly_score REAL NOT NULL,
            dga_probability REAL NOT NULL,
            predicted_class TEXT NOT NULL,
            raw_sequence TEXT
        )
    """)
    try:
        cursor.execute("ALTER TABLE events ADD COLUMN transformer_predicted_class TEXT")
    except sqlite3.OperationalError:
        pass
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS robustness_results (
            run_id TEXT PRIMARY KEY,
            timestamp REAL,
            model_name TEXT,
            clean_acc REAL,
            fgsm_acc REAL,
            pgd_acc REAL
        )
    """)
    conn.commit()
    conn.close()
    print(f"   [OK] SQLite Database initialized at: {DB_FILE}")


def insert_robustness_result(run_id: str, model_name: str, clean_acc: float, fgsm_acc: float, pgd_acc: float, timestamp: float | None = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO robustness_results (run_id, timestamp, model_name, clean_acc, fgsm_acc, pgd_acc)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, timestamp if timestamp is not None else __import__("time").time(), model_name, clean_acc, fgsm_acc, pgd_acc),
    )
    conn.commit()
    conn.close()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()
