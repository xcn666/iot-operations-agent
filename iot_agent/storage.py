"""SQLite persistence for sessions, steps and pending write actions."""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .models import AgentState, AgentStatus


class AgentStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_actions (
                    token TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    executed_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
                );
                """
            )

    def save(self, state: AgentState) -> None:
        payload = json.dumps(state.to_dict(), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_sessions(session_id, task, status, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    task=excluded.task,
                    status=excluded.status,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    state.session_id,
                    state.task,
                    state.status.value,
                    payload,
                    state.created_at,
                    state.updated_at,
                ),
            )

    def load(self, session_id: str) -> AgentState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return AgentState.from_dict(json.loads(row["state_json"]))

    def create_pending(self, session_id: str, tool_name: str, args: dict[str, Any], created_at: str) -> str:
        token = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pending_actions(token, session_id, tool_name, args_json, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (token, session_id, tool_name, json.dumps(args, ensure_ascii=False), created_at),
            )
        return token

    def get_pending(self, token: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pending_actions WHERE token = ?",
                (token,),
            ).fetchone()
        if not row:
            return None
        return {
            "token": row["token"],
            "session_id": row["session_id"],
            "tool_name": row["tool_name"],
            "args": json.loads(row["args_json"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "executed_at": row["executed_at"],
        }

    def mark_pending(self, token: str, status: str, executed_at: str) -> None:
        if status not in {"executed", "rejected", "failed"}:
            raise ValueError("invalid pending action status")
        with self._connect() as connection:
            connection.execute(
                "UPDATE pending_actions SET status = ?, executed_at = ? WHERE token = ? AND status = 'pending'",
                (status, executed_at, token),
            )

    def session_status(self, session_id: str) -> AgentStatus | None:
        state = self.load(session_id)
        return state.status if state else None
