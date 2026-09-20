"""Agent loop, guardrails and confirmation workflow."""
from __future__ import annotations

import re
from typing import Any

from .models import AgentState, AgentStatus, DecisionKind
from .planner import Decision, Planner
from .storage import AgentStore
from .tools import ToolRegistry, validate_arguments
from .models import utc_now


class Verifier:
    def validate(self, decision: Decision, state: AgentState, registry: ToolRegistry) -> str | None:
        if decision.kind == DecisionKind.CALL_TOOL:
            if not registry.get(decision.tool_name):
                return f"unknown tool: {decision.tool_name}"
            if state.duplicate_tool_calls(decision.tool_name, decision.args) >= 1:
                return f"duplicate tool call stopped: {decision.tool_name}"
        if decision.kind == DecisionKind.FINISH and not decision.content.strip():
            return "finish answer is empty"
        if decision.kind == DecisionKind.FINISH:
            executed = {step.name for step in state.steps if step.kind == "tool"}
            required = set()
            task = state.task
            if "趋势" in task or "变化" in task or "异常次数" in task:
                required.add("analyze_trend")
            if any(marker in task for marker in ("怎么实现", "如何实现", "什么数据库", "用了什么", "使用了什么", "机制", "原理", "检索文档", "说明")):
                required.add("search_iot_docs")
            if any(marker in task for marker in ("多少条", "几条", "数据量", "总数", "平均", "均值", "最高", "最大", "最低", "最小")) and "文档" not in task:
                required.add("get_sensor_stats")
            if "最近" in task and any(marker in task for marker in ("读数", "记录")) and "告警" not in task:
                required.add("get_sensor_readings")
            if "告警" in task and any(marker in task for marker in ("最近", "查看", "哪些", "列出")) and "规则" not in task:
                required.add("list_alerts")
            if "创建" in task and "告警规则" in task:
                required.add("create_alert_rule")
            if "发送通知" in task or "推送通知" in task:
                required.add("send_notification")
            missing = sorted(required - executed)
            if missing:
                return "task coverage: missing " + ", ".join(missing)
        if decision.kind == DecisionKind.ASK_USER and not decision.content.strip():
            return "ask_user question is empty"
        return None


class AgentRuntime:
    def __init__(
        self,
        store: AgentStore,
        registry: ToolRegistry,
        planner: Planner,
        max_steps: int = 8,
        verifier: Verifier | None = None,
    ):
        self.store = store
        self.registry = registry
        self.planner = planner
        self.max_steps = max_steps
        self.verifier = verifier or Verifier()

    @staticmethod
    def _response(state: AgentState) -> dict[str, Any]:
        return {
            "session_id": state.session_id,
            "status": state.status.value,
            "answer": state.answer,
            "error": state.error,
            "pending_action": state.pending_action,
            "trace": [step.to_dict() for step in state.steps],
        }

    def chat(self, task: str, session_id: str | None = None) -> dict[str, Any]:
        task = (task or "").strip()
        if not task:
            return {"status": "failed", "answer": "请输入问题", "trace": []}
        state = self.store.load(session_id) if session_id else None
        if state and state.status == AgentStatus.AWAITING_CONFIRMATION:
            return self._response(state)
        if not state:
            state = AgentState.new(task, session_id=session_id)
        elif state.status in {AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.NEEDS_INPUT}:
            state = AgentState.new(task)
        else:
            state.task = task
        self.store.save(state)
        return self._run_loop(state)

    def confirm(self, token: str) -> dict[str, Any]:
        pending = self.store.get_pending(token)
        if not pending:
            return {"status": "failed", "error": "confirmation token not found", "trace": []}
        if pending["status"] != "pending":
            return {"status": "failed", "error": f"action already {pending['status']}", "trace": []}
        state = self.store.load(pending["session_id"])
        if not state:
            return {"status": "failed", "error": "session not found", "trace": []}

        result = self.registry.execute(pending["tool_name"], pending["args"], confirmed=True)
        state.add_step(
            kind="tool",
            name=pending["tool_name"],
            args=pending["args"],
            result=result.to_dict(),
            elapsed_ms=result.elapsed_ms,
        )
        state.pending_action = None
        state.status = AgentStatus.RUNNING if result.ok else AgentStatus.RUNNING
        if not result.ok:
            state.error = result.error or "confirmed tool failed"
        self.store.mark_pending(token, "executed" if result.ok else "failed", utc_now())
        if result.ok:
            state.answer = self.planner.finalize(state)
            state.status = AgentStatus.COMPLETED
            self.store.save(state)
            return self._response(state)
        self.store.save(state)
        return self._run_loop(state)

    def reject(self, token: str) -> dict[str, Any]:
        pending = self.store.get_pending(token)
        if not pending:
            return {"status": "failed", "error": "confirmation token not found", "trace": []}
        if pending["status"] != "pending":
            return {"status": "failed", "error": f"action already {pending['status']}", "trace": []}
        state = self.store.load(pending["session_id"])
        if not state:
            return {"status": "failed", "error": "session not found", "trace": []}
        state.add_step(
            kind="confirmation",
            name="rejected",
            args={"tool": pending["tool_name"], "args": pending["args"]},
        )
        state.pending_action = None
        state.answer = "已取消该写操作。"
        state.status = AgentStatus.COMPLETED
        self.store.mark_pending(token, "rejected", utc_now())
        self.store.save(state)
        return self._response(state)

    @staticmethod
    def _task_has_explicit_time_range(task: str) -> bool:
        if re.search(r"\d{4}[-/年]\d{1,2}", task):
            return True
        if re.search(r"\d+\s*(小时|天|周|个月|月|年)前?", task):
            return True
        return any(word in task for word in ("今天", "昨天", "前天", "本周", "上周", "本月", "上月", "今年", "去年"))

    @classmethod
    def _sanitize_tool_args(cls, task: str, args: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(args)
        if not cls._task_has_explicit_time_range(task):
            cleaned.pop("start", None)
            cleaned.pop("end", None)
        return cleaned

    def _run_loop(self, state: AgentState) -> dict[str, Any]:
        while state.step_count < self.max_steps:
            decision = self.planner.decide(state)
            if decision.kind == DecisionKind.CALL_TOOL:
                decision.args = self._sanitize_tool_args(state.task, decision.args)
            state.add_step(
                kind="decision",
                name=decision.kind.value,
                args={
                    "tool": decision.tool_name,
                    "planner": decision.source,
                },
            )
            validation_error = self.verifier.validate(decision, state, self.registry)
            if validation_error and validation_error.startswith("duplicate tool call"):
                previous_guardrails = sum(
                    1
                    for step in state.steps
                    if step.kind == "guardrail"
                    and step.args.get("tool") == decision.tool_name
                    and step.args.get("called_with") == decision.args
                )
                state.add_step(
                    kind="guardrail",
                    name="duplicate_tool_call",
                    args={"tool": decision.tool_name, "called_with": decision.args},
                    result={"ok": False, "error": validation_error},
                )
                if previous_guardrails >= 1:
                    state.answer = self.planner.finalize(state)
                    state.status = AgentStatus.COMPLETED
                    self.store.save(state)
                    return self._response(state)
                self.store.save(state)
                continue
            if validation_error and validation_error.startswith("task coverage:"):
                previous_coverage_errors = sum(
                    1 for step in state.steps
                    if step.kind == "guardrail" and step.name == "task_coverage"
                    and step.result.get("error") == validation_error
                )
                state.add_step(
                    kind="guardrail",
                    name="task_coverage",
                    args={},
                    result={"ok": False, "error": validation_error},
                )
                if previous_coverage_errors >= 1:
                    state.error = validation_error
                    state.status = AgentStatus.FAILED
                    self.store.save(state)
                    return self._response(state)
                self.store.save(state)
                continue
            if validation_error:
                state.error = validation_error
                state.status = AgentStatus.FAILED
                self.store.save(state)
                return self._response(state)

            if decision.kind == DecisionKind.ASK_USER:
                state.answer = decision.content
                state.status = AgentStatus.NEEDS_INPUT
                self.store.save(state)
                return self._response(state)

            if decision.kind == DecisionKind.FINISH:
                state.answer = decision.content
                state.status = AgentStatus.COMPLETED
                self.store.save(state)
                return self._response(state)

            spec = self.registry.get(decision.tool_name)
            if not spec:
                state.error = f"unknown tool: {decision.tool_name}"
                state.status = AgentStatus.FAILED
                self.store.save(state)
                return self._response(state)

            try:
                validate_arguments(spec.parameters, decision.args)
            except Exception as exc:
                result = {"ok": False, "error": f"invalid arguments: {exc}", "data": None, "summary": "", "elapsed_ms": 0}
                state.add_step("tool", decision.tool_name, decision.args, result)
                self.store.save(state)
                continue

            if spec.requires_confirmation:
                token = self.store.create_pending(state.session_id, spec.name, decision.args, utc_now())
                state.pending_action = {
                    "token": token,
                    "tool": spec.name,
                    "args": decision.args,
                    "permission": spec.permission,
                    "description": spec.description,
                }
                state.answer = "该操作会修改系统状态，请确认后继续。"
                state.status = AgentStatus.AWAITING_CONFIRMATION
                self.store.save(state)
                return self._response(state)

            result = self.registry.execute(spec.name, decision.args)
            state.add_step(
                kind="tool",
                name=spec.name,
                args=decision.args,
                result=result.to_dict(),
                elapsed_ms=result.elapsed_ms,
            )
            if not result.ok:
                state.error = result.error or "tool failed"
            self.store.save(state)
            if result.ok and decision.terminal:
                state.answer = self.planner.finalize(state)
                state.status = AgentStatus.COMPLETED
                self.store.save(state)
                return self._response(state)

        state.status = AgentStatus.MAX_STEPS
        state.answer = self.planner.finalize(state)
        self.store.save(state)
        return self._response(state)
