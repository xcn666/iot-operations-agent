"""Domain models for the multi-step Agent runtime."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AgentStatus(str, Enum):
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    NEEDS_INPUT = "needs_input"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS = "max_steps"


class DecisionKind(str, Enum):
    CALL_TOOL = "call_tool"
    FINISH = "finish"
    ASK_USER = "ask_user"


@dataclass
class AgentStep:
    step_no: int
    kind: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_no": self.step_no,
            "kind": self.kind,
            "name": self.name,
            "args": self.args,
            "result": self.result,
            "elapsed_ms": self.elapsed_ms,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentStep":
        return cls(**value)


@dataclass
class AgentState:
    session_id: str
    task: str
    status: AgentStatus = AgentStatus.RUNNING
    steps: list[AgentStep] = field(default_factory=list)
    answer: str = ""
    error: str = ""
    pending_action: dict[str, Any] | None = None
    plan_queue: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def new(cls, task: str, session_id: str | None = None) -> "AgentState":
        return cls(session_id=session_id or uuid.uuid4().hex, task=task.strip())

    @property
    def step_count(self) -> int:
        return len([step for step in self.steps if step.kind != "decision"])

    def add_step(
        self,
        kind: str,
        name: str,
        args: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        elapsed_ms: int = 0,
    ) -> AgentStep:
        step = AgentStep(
            step_no=len(self.steps) + 1,
            kind=kind,
            name=name,
            args=args or {},
            result=result or {},
            elapsed_ms=elapsed_ms,
        )
        self.steps.append(step)
        self.updated_at = utc_now()
        return step

    def tool_observations(self, limit: int = 12) -> list[dict[str, Any]]:
        observations = []
        for step in self.steps:
            if step.kind != "tool":
                continue
            observations.append(
                {
                    "tool": step.name,
                    "args": step.args,
                    "ok": step.result.get("ok", False),
                    "data": step.result.get("data"),
                    "error": step.result.get("error"),
                }
            )
        return observations[-limit:]

    def duplicate_tool_calls(self, tool_name: str, args: dict[str, Any]) -> int:
        count = 0
        for step in self.steps:
            if step.kind == "tool" and step.name == tool_name and step.args == args:
                count += 1
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task": self.task,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "answer": self.answer,
            "error": self.error,
            "pending_action": self.pending_action,
            "plan_queue": self.plan_queue,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentState":
        value = dict(value)
        value["status"] = AgentStatus(value.get("status", AgentStatus.RUNNING.value))
        value["steps"] = [AgentStep.from_dict(item) for item in value.get("steps", [])]
        value.setdefault("plan_queue", [])
        value.setdefault("pending_action", None)
        return cls(**value)
