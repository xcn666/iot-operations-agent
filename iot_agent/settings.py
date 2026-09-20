"""Runtime settings for the IoT Operations Agent."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


@dataclass(frozen=True)
class Settings:
    sensor_db_path: Path
    rag_index_path: Path
    agent_db_path: Path
    docs_path: Path
    notify_config_path: Path
    rag_top_k: int = 3
    max_steps: int = 8
    max_tool_retries: int = 2
    tool_timeout_seconds: int = 15

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        return cls(
            sensor_db_path=_expand(os.getenv("IOT_DB_PATH", "~/iot-lab/logs/sensor.db")),
            rag_index_path=_expand(os.getenv("IOT_RAG_INDEX", str(project_root / "rag_index.json"))),
            agent_db_path=_expand(os.getenv("IOT_AGENT_DB", str(project_root / "data" / "agent.db"))),
            docs_path=_expand(os.getenv("IOT_DOCS_PATH", "~/iot-lab/README.md")),
            notify_config_path=_expand(os.getenv("IOT_NOTIFY_CONFIG", "~/iot-lab/notify_config.json")),
        )
