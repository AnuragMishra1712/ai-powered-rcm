import sqlite3
import pandas as pd
import datetime
import os

DB_PATH = "metrics_log.db"


def init_db():
    """Initialize DB if not present."""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name TEXT,
            confidence REAL,
            runtime REAL,
            created_at TEXT
        );
        """)
        conn.commit()
        conn.close()


def log_metric(module_name: str, confidence: float, runtime: float):
    """Log module execution summary."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO metrics (module_name, confidence, runtime, created_at) VALUES (?, ?, ?, ?)",
        (module_name, confidence, runtime, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_metrics_summary(limit: int = 100):
    """Fetch recent metrics for dashboard display."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM metrics ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return df
