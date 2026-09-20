"""Planner for one-step-at-a-time tool selection."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .llm import LLMClient, LLMError, parse_json_lenient
from .models import AgentState, DecisionKind
from .tools import ToolRegistry


@dataclass
class Decision:
    kind: DecisionKind
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    source: str = "unknown"
    terminal: bool = False


class Planner:
    def __init__(self, llm: LLMClient, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry

    def _system_prompt(self) -> str:
        now = datetime.now().isoformat(timespec="minutes")
        tools = self.registry.names()
        return (
            "You are an IoT Operations Agent. Solve the user task step by step. "
            "Call exactly one tool per turn, inspect the result, then decide the next action. "
            "Never invent sensor values or documentation. Use get_sensor_stats for row count and simple max/min/average. Use analyze_trend for trends or threshold anomaly counts. Use finish only after the task is satisfied. "
            "Use ask_user when required information is missing. "
            "Write tools require explicit confirmation and the runtime will pause automatically. "
            f"Current local time: {now}. Available tools: {', '.join(tools)}. "
            "For multi-intent tasks, complete every requested subtask before finish. Answer in Chinese unless the user asks otherwise."
        )

    def _context(self, state: AgentState) -> str:
        return json.dumps(
            {
                "task": state.task,
                "observations": state.tool_observations(),
                "guardrails": [
                    {"tool": step.args.get("tool"), "error": step.result.get("error")}
                    for step in state.steps
                    if step.kind == "guardrail"
                ][-3:],
                "executed_steps": state.step_count,
            },
            ensure_ascii=False,
        )

    def _control_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Finish the task with a grounded final answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": "Ask the user for missing information required to continue.",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        return parse_json_lenient(str(raw))

    def _decision_from_tool_call(self, call: dict[str, Any]) -> Decision:
        function = call.get("function", {})
        name = function.get("name", "")
        args = self._parse_args(function.get("arguments", {}))
        if name == "finish":
            return Decision(kind=DecisionKind.FINISH, content=args.get("answer", ""), source="native")
        if name == "ask_user":
            return Decision(kind=DecisionKind.ASK_USER, content=args.get("question", ""), source="native")
        if not self.registry.get(name):
            return Decision(kind=DecisionKind.ASK_USER, content=f"工具不存在：{name}", source="native")
        return Decision(kind=DecisionKind.CALL_TOOL, tool_name=name, args=args, source="native")

    def _json_prompt(self, state: AgentState, include_tools: bool) -> list[dict[str, str]]:
        tool_text = json.dumps(self.registry.schemas(), ensure_ascii=False) if include_tools else "[]"
        user = (
            "Choose the next action. Return exactly one JSON object and nothing else.\n"
            "Allowed forms:\n"
            '{"kind":"call_tool","tool":"TOOL_NAME","args":{}}\n'
            '{"kind":"finish","answer":"GROUNDED_ANSWER"}\n'
            '{"kind":"ask_user","question":"QUESTION"}\n\n'
            f"Tool schemas:\n{tool_text}\n\nState:\n{self._context(state)}"
        )
        return [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": user}]

    @staticmethod
    def _decompose_task(task: str) -> list[dict[str, Any]]:
        field = "hum" if "湿度" in task else "temp"
        threshold_match = re.search(r"(?:超过|高于|大于)\s*(\d+(?:\.\d+)?)", task)
        trend_args: dict[str, Any] = {"field": field}
        if threshold_match:
            trend_args["threshold"] = float(threshold_match.group(1))

        if ("先" in task and "再" in task and any(word in task for word in ("读数", "记录"))
                and ("趋势" in task or "变化" in task)):
            limit_match = re.search(r"(\d+)\s*条", task)
            return [
                {"tool": "get_sensor_readings", "args": {"limit": int(limit_match.group(1)) if limit_match else 3}},
                {"tool": "analyze_trend", "args": trend_args},
            ]
        if "先" in task and "再" in task and "告警" in task and any(word in task for word in ("文档", "检索", "说明", "机制")):
            return [
                {"tool": "list_alerts", "args": {"limit": 10}},
                {"tool": "search_iot_docs", "args": {"query": task}},
            ]
        if any(word in task for word in ("最高", "最大", "平均", "最低", "最小")) and any(
            word in task for word in ("文档", "检索", "说明", "采集")
        ):
            metric = "avg" if "平均" in task else ("min" if any(word in task for word in ("最低", "最小")) else "max")
            return [
                {"tool": "get_sensor_stats", "args": {"metric": metric, "field": field}},
                {"tool": "search_iot_docs", "args": {"query": task}},
            ]
        return []

    @staticmethod
    def _rule_based_decision(task: str) -> Decision | None:
        if any(marker in task for marker in ("先", "然后", "再", "同时")):
            return None
        if "并" in task and "趋势" not in task:
            return None
        if any(word in task for word in ("创建", "告警规则")):
            return None
        if any(phrase in task for phrase in ("这个设备", "该设备", "本设备")):
            return None
        if "发送通知" in task or "推送通知" in task:
            text = task.split("：", 1)[-1].split(":", 1)[-1].strip() or task
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                tool_name="send_notification",
                args={"text": text},
                source="rule",
                terminal=True,
            )
        field = "hum" if "湿度" in task else "temp"
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", task)
        date_args = {"start": date_match.group(0), "end": date_match.group(0)} if date_match else {}
        doc_markers = ("怎么实现", "如何实现", "实现细节", "什么数据库", "用了什么", "使用了什么", "机制", "原理")
        if any(marker in task for marker in doc_markers):
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                tool_name="search_iot_docs",
                args={"query": task},
                source="rule",
                terminal=True,
            )
        threshold_match = re.search(r"(?:超过|高于|大于)\s*(\d+(?:\.\d+)?)", task)
        if "趋势" in task or "变化" in task or threshold_match:
            args = {"field": field, **date_args}
            if threshold_match:
                args["threshold"] = float(threshold_match.group(1))
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                tool_name="analyze_trend",
                args=args,
                source="rule",
                terminal=True,
            )
        if any(word in task for word in ("多少条", "几条", "数据量", "总数")) and "告警" not in task:
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                tool_name="get_sensor_stats",
                args={"metric": "count", "field": field},
                source="rule",
                terminal=True,
            )
        if any(word in task for word in ("平均", "均值", "最高", "最大", "最低", "最小")):
            if "平均" in task or "均值" in task:
                metric = "avg"
            elif "最低" in task or "最小" in task:
                metric = "min"
            else:
                metric = "max"
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                tool_name="get_sensor_stats",
                args={"metric": metric, "field": field, **date_args},
                source="rule",
                terminal=True,
            )
        if "告警" in task and "规则" not in task and "发送" not in task and any(
            word in task for word in ("最近", "查看", "哪些", "列出")
        ):
            limit_match = re.search(r"(\d+)\s*条", task)
            args = {"limit": int(limit_match.group(1)) if limit_match else 10}
            return Decision(kind=DecisionKind.CALL_TOOL, tool_name="list_alerts", args=args, source="rule", terminal=True)
        if any(word in task for word in ("最近", "读数是", "读数有", "原始读数", "记录")) and "告警" not in task:
            limit_match = re.search(r"(\d+)\s*条", task)
            args = {"limit": int(limit_match.group(1)) if limit_match else 5}
            return Decision(kind=DecisionKind.CALL_TOOL, tool_name="get_sensor_readings", args=args, source="rule", terminal=True)
        return None

    def decide(self, state: AgentState) -> Decision:
        if state.step_count == 0 and any(phrase in state.task for phrase in ("这个设备", "该设备", "本设备")):
            return Decision(
                kind=DecisionKind.ASK_USER,
                content="请提供需要分析的设备 ID。",
                source="clarification",
            )
        if state.plan_queue:
            queued = state.plan_queue.pop(0)
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                tool_name=queued["tool"],
                args=queued.get("args", {}),
                source="decomposition",
                terminal=not state.plan_queue,
            )
        if state.step_count == 0:
            plan = self._decompose_task(state.task)
            if plan:
                first = plan.pop(0)
                state.plan_queue = plan
                return Decision(
                    kind=DecisionKind.CALL_TOOL,
                    tool_name=first["tool"],
                    args=first.get("args", {}),
                    source="decomposition",
                    terminal=not plan,
                )
            rule_decision = self._rule_based_decision(state.task)
            if rule_decision:
                return rule_decision
        native_tools = self.registry.schemas() + self._control_tools()
        try:
            message = self.llm.chat(
                [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._context(state)},
                ],
                tools=native_tools,
                temperature=0.1,
            )
            calls = message.get("tool_calls") or []
            if calls:
                return self._decision_from_tool_call(calls[0])
        except Exception:
            pass

        try:
            message = self.llm.chat(self._json_prompt(state, include_tools=True), temperature=0.1)
            payload = parse_json_lenient(message.get("content", ""))
            kind = payload.get("kind")
            if kind == DecisionKind.CALL_TOOL.value:
                return Decision(
                    kind=DecisionKind.CALL_TOOL,
                    tool_name=payload.get("tool", ""),
                    args=payload.get("args") or {},
                    source="json",
                )
            if kind == DecisionKind.FINISH.value:
                return Decision(kind=DecisionKind.FINISH, content=payload.get("answer", ""), source="json")
            if kind == DecisionKind.ASK_USER.value:
                return Decision(kind=DecisionKind.ASK_USER, content=payload.get("question", ""), source="json")
        except Exception:
            pass
        return Decision(
            kind=DecisionKind.ASK_USER,
            content="任务规划失败，请补充更明确的问题或稍后重试。",
            source="fallback",
        )

    def finalize(self, state: AgentState) -> str:
        observations = json.dumps(state.tool_observations(), ensure_ascii=False)
        try:
            message = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "Answer in concise Chinese. Use only the supplied tool observations. Do not invent facts.",
                    },
                    {
                        "role": "user",
                        "content": f"任务：{state.task}\n工具结果：{observations}",
                    },
                ],
                temperature=0.2,
            )
            content = (message.get("content") or "").strip()
            if content:
                return content
        except LLMError:
            pass
        summaries = [step.result.get("summary", "") for step in state.steps if step.kind == "tool"]
        summaries = [summary for summary in summaries if summary]
        return "已完成工具执行。" + (" " + "；".join(summaries) if summaries else "")
