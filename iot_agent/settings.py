"""Runtime settings for the IoT Operations Agent."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


@dataclass(frozen=True)
class Settings:
    rag_index_path: Path
    agent_db_path: Path
    notify_config_path: Path
    rag_top_k: int = 3
    max_steps: int = 8
    max_tool_retries: int = 2
    tool_timeout_seconds: int = 15
    monitor_api_url: str = "http://127.0.0.1:5002"
    monitor_api_token: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        return cls(
            rag_index_path=_expand(os.getenv("IOT_RAG_INDEX", str(project_root / "rag_index.json"))),
            agent_db_path=_expand(os.getenv("IOT_AGENT_DB", str(project_root / "data" / "agent.db"))),
            notify_config_path=_expand(os.getenv("IOT_NOTIFY_CONFIG", "~/iot-lab/notify_config.json")),
            monitor_api_url=os.getenv("IOT_MONITOR_API_URL", "http://127.0.0.1:5002").rstrip("/"),
            monitor_api_token=os.getenv("IOT_MONITOR_API_TOKEN", ""),
        )
