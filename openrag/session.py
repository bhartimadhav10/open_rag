from __future__ import annotations
import sqlite3
import uuid
import time
from pathlib import Path
from .config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);
"""


class SessionStore:
    def __init__(self, db_path: Path | None = None):
        p = Path(db_path or settings.session_db)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(p), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def create(self) -> str:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        self.conn.execute(
            "INSERT INTO sessions(session_id, created_at, updated_at) VALUES(?,?,?)",
            (sid, now, now),
        )
        self.conn.commit()
        return sid

    def ensure(self, session_id: str | None) -> str:
        if not session_id:
            return self.create()
        row = self.conn.execute(
            "SELECT session_id FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            now = time.time()
            self.conn.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at) VALUES(?,?,?)",
                (session_id, now, now),
            )
            self.conn.commit()
        return session_id

    def append_turn(self, session_id: str, role: str, content: str):
        now = time.time()
        self.conn.execute(
            "INSERT INTO turns(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (session_id, role, content, now),
        )
        self.conn.execute(
            "UPDATE sessions SET updated_at=? WHERE session_id=?",
            (now, session_id),
        )
        self.conn.commit()

    def history(self, session_id: str, max_turns: int | None = None) -> list[dict]:
        limit = (max_turns if max_turns is not None else settings.max_history_turns) * 2
        rows = self.conn.execute(
            "SELECT role, content FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        rows.reverse()
        return [{"role": r, "content": c} for r, c in rows]

    def reset(self, session_id: str):
        self.conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        self.conn.commit()

    def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            """SELECT s.session_id, s.created_at, s.updated_at,
                      (SELECT COUNT(*) FROM turns t WHERE t.session_id=s.session_id) as turn_count
               FROM sessions s ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "updated_at": r[2], "turns": r[3]}
            for r in rows
        ]
