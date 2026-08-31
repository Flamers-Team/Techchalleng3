"""Backend SQLite + Loguru para auditoria."""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from loguru import logger

DB_PATH = Path("audit.db")


def init_db(db_path: Path = DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            session_id TEXT NOT NULL,
            user_id TEXT,
            agent TEXT,
            model TEXT,
            input_hash TEXT,
            output_hash TEXT,
            input_preview TEXT,
            output_preview TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            latency_ms INTEGER,
            cost_usd REAL,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON events(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON events(user_id)")
    conn.commit()
    conn.close()
    logger.info(f"DB inicializado: {db_path}")


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_event(event, db_path: Path = DB_PATH):
    import hashlib
    data = event.to_dict()

    input_hash = hashlib.sha256(str(data.get("input_preview", "")).encode()).hexdigest()[:16]
    output_hash = hashlib.sha256(str(data.get("output_preview", "")).encode()).hexdigest()[:16]

    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO events (
                timestamp, event_type, session_id, user_id,
                agent, model, input_hash, output_hash,
                input_preview, output_preview,
                tokens_in, tokens_out, latency_ms, cost_usd, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("timestamp", datetime.utcnow().isoformat()),
            data.get("event_type", "unknown"),
            data.get("session_id", ""),
            data.get("user_id", "anonymous"),
            data.get("agent", ""),
            data.get("model", ""),
            input_hash, output_hash,
            str(data.get("input_preview", ""))[:500],
            str(data.get("output_preview", ""))[:500],
            data.get("tokens_in", 0),
            data.get("tokens_out", 0),
            data.get("latency_ms", 0),
            data.get("estimated_cost_usd", 0.0),
            json.dumps(data.get("metadata", {})),
        ))
    logger.info(f"[{event.event_type}] session={event.session_id}")