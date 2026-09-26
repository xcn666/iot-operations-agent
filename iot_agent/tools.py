"""Typed tool contracts and implementations for the IoT Agent."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .settings import Settings


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    summary: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "summary": self.summary,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]
    permission: str = "read"
    requires_confirmation: bool = False


class ToolValidationError(ValueError):
    pass


def _expect_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> None:
    if not isinstance(args, dict):
        raise ToolValidationError("arguments must be an object")
    properties = schema.get("properties", {})
    for name, rule in properties.items():
        if name not in args and "default" in rule:
            args[name] = rule["default"]
    required = schema.get("required", [])
    for name in required:
        if name not in args or args[name] is None:
            raise ToolValidationError(f"missing required argument: {name}")
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(properties))
        if unknown:
            raise ToolValidationError(f"unknown arguments: {', '.join(unknown)}")
    for name, value in args.items():
        rule = properties.get(name)
        if not rule:
            continue
        expected = rule.get("type")
        if isinstance(expected, list):
            if not any(_expect_type(value, item) for item in expected):
                raise ToolValidationError(f"invalid type for {name}")
        elif expected and not _expect_type(value, expected):
            raise ToolValidationError(f"invalid type for {name}")
        if "enum" in rule and value not in rule["enum"]:
            raise ToolValidationError(f"invalid value for {name}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                raise ToolValidationError(f"{name} is below minimum")
            if "maximum" in rule and value > rule["maximum"]:
                raise ToolValidationError(f"{name} is above maximum")
        if isinstance(value, str):
            if "maxLength" in rule and len(value) > rule["maxLength"]:
                raise ToolValidationError(f"{name} is too long")
            if rule.get("format") == "date-time":
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ToolValidationError(f"{name} must be ISO datetime") from exc


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_temporal(value: str) -> tuple[str, str | None]:
    normalized = value.strip().replace("T", " ")
    if len(normalized) == 10:
        datetime.strptime(normalized, "%Y-%m-%d")
        return normalized, None
    parsed = datetime.fromisoformat(normalized)
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%H:%M:%S")


def _temporal_where(start: str | None, end: str | None) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start:
        day, clock = _parse_temporal(start)
        if clock is None:
            clauses.append("day >= ?")
            params.append(day)
        else:
            clauses.append("(day > ? OR (day = ? AND time >= ?))")
            params.extend([day, day, clock])
    if end:
        day, clock = _parse_temporal(end)
        if clock is None:
            clauses.append("day <= ?")
            params.append(day)
        else:
            clauses.append("(day < ? OR (day = ? AND time <= ?))")
            params.extend([day, day, clock])
    return clauses, params


class ToolRegistry:
    def __init__(self, timeout_seconds: int = 15):
        self._tools: dict[str, ToolSpec] = {}
        self.timeout_seconds = timeout_seconds

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        if spec.permission not in {"read", "write"}:
            raise ValueError(f"invalid permission for {spec.name}")
        if spec.permission == "write" and not spec.requires_confirmation:
            raise ValueError(f"write tool must require confirmation: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, args: dict[str, Any], confirmed: bool = False) -> ToolResult:
        spec = self.get(name)
        if not spec:
            return ToolResult(ok=False, error=f"unknown tool: {name}")
        if spec.requires_confirmation and not confirmed:
            return ToolResult(ok=False, error=f"confirmation required: {name}")
        started = time.perf_counter()
        try:
            validate_arguments(spec.parameters, args)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(spec.handler, args)
                try:
                    result = future.result(timeout=self.timeout_seconds)
                except FutureTimeout:
                    return ToolResult(ok=False, error=f"tool timed out: {name}")
            if not isinstance(result, ToolResult):
                result = ToolResult(ok=True, data=result)
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            return result
        except ToolValidationError as exc:
            return ToolResult(ok=False, error=f"invalid arguments: {exc}")
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )


class RagIndex:
    def __init__(self, index_path: Path, llm: Any, top_k: int = 3):
        self.index_path = index_path
        self.llm = llm
        self.top_k = top_k
        self.chunks: list[dict[str, Any]] = []
        self.loaded = False

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"RAG index not found: {self.index_path}")
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.chunks = payload.get("chunks", [])
        self.chunk_terms = [self._query_terms(chunk.get("text", "")) for chunk in self.chunks]
        document_frequency: dict[str, int] = {}
        for terms in self.chunk_terms:
            for term in terms:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        document_count = max(len(self.chunks), 1)
        self.keyword_idf = {
            term: math.log((document_count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.loaded = True

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return dot / (left_norm * right_norm + 1e-9)

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        stop_words = ("项目", "里的", "怎么", "如何", "实现", "什么", "使用了", "用了", "为什么", "这个", "该", "是", "的")
        terms = {token.lower() for token in re.findall(r"[A-Za-z0-9_]{2,}", query)}
        for run in re.findall(r"[\u4e00-\u9fff]+", query):
            cleaned = run
            for word in stop_words:
                cleaned = cleaned.replace(word, "")
            for size in (2, 3, 4):
                terms.update(cleaned[index:index + size] for index in range(max(0, len(cleaned) - size + 1)))
        return {term for term in terms if term}

    def _keyword_score(self, query_terms: set[str], text_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        matched = query_terms & text_terms
        if not matched:
            return 0.0
        total_weight = sum(self.keyword_idf.get(term, 1.0) for term in query_terms)
        matched_weight = sum(self.keyword_idf.get(term, 1.0) for term in matched)
        return matched_weight / total_weight if total_weight else 0.0

    def search(self, query: str, top_k: int | None = None) -> list[tuple[float, dict[str, Any]]]:
        if not self.loaded:
            self.load()
        if not self.chunks:
            return []
        query_vector = self.llm.embed(query)
        terms = self._query_terms(query)
        scored = []
        for index, chunk in enumerate(self.chunks):
            vector_score = self._cosine(query_vector, chunk["vector"])
            keyword_score = self._keyword_score(terms, self.chunk_terms[index])
            score = 0.65 * vector_score + 0.35 * keyword_score
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[: min(top_k or self.top_k, 10)]


def build_default_registry(settings: Settings, llm: Any) -> ToolRegistry:
    registry = ToolRegistry(timeout_seconds=settings.tool_timeout_seconds)
    rag = RagIndex(settings.rag_index_path, llm, top_k=settings.rag_top_k)

    def get_sensor_stats(args: dict[str, Any]) -> ToolResult:
        metric = args.get("metric", "max")
        field = args.get("field", "temp")
        device_id = args.get("device_id")
        start = args.get("start")
        end = args.get("end")
        function = {"max": "MAX", "min": "MIN", "avg": "AVG", "count": "COUNT"} [metric]
        expression = "COUNT(*)" if metric == "count" else f"ROUND({function}({field}), 2)"
        clauses, params = _temporal_where(start, end)
        if device_id:
            clauses.append("device = ?")
            params.append(device_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT {expression}, COUNT(*) FROM readings{where}"
        with _connect_readonly(settings.sensor_db_path) as connection:
            value, count = connection.execute(sql, params).fetchone()
        return ToolResult(
            ok=True,
            data={"metric": metric, "field": field, "value": value, "sample_count": count},
            summary=f"{field} {metric}={value}, samples={count}",
        )

    def get_sensor_readings(args: dict[str, Any]) -> ToolResult:
        limit = int(args.get("limit", 20))
        clauses, params = _temporal_where(args.get("start"), args.get("end"))
        if args.get("device_id"):
            clauses.append("device = ?")
            params.append(args["device_id"])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = "SELECT day, time, device, temp, hum FROM readings" + where + " ORDER BY day DESC, time DESC LIMIT ?"
        params.append(limit)
        with _connect_readonly(settings.sensor_db_path) as connection:
            rows = [dict(row) for row in connection.execute(sql, params)]
        return ToolResult(
            ok=True,
            data={"rows": rows, "returned": len(rows)},
            summary=f"returned {len(rows)} sensor readings",
        )

    def analyze_trend(args: dict[str, Any]) -> ToolResult:
        field = args.get("field", "temp")
        threshold = args.get("threshold")
        clauses, params = _temporal_where(args.get("start"), args.get("end"))
        if args.get("device_id"):
            clauses.append("device = ?")
            params.append(args["device_id"])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT day, time, {field} AS value FROM readings{where} ORDER BY day ASC, time ASC LIMIT 5000"
        with _connect_readonly(settings.sensor_db_path) as connection:
            rows = connection.execute(sql, params).fetchall()
        values = [float(row["value"]) for row in rows if row["value"] is not None]
        if not values:
            return ToolResult(ok=True, data={"sample_count": 0}, summary="no readings in range")
        count = len(values)
        mean = sum(values) / count
        slope = 0.0
        if count > 1:
            x_mean = (count - 1) / 2
            numerator = sum((index - x_mean) * (value - mean) for index, value in enumerate(values))
            denominator = sum((index - x_mean) ** 2 for index in range(count))
            slope = numerator / denominator if denominator else 0.0
        anomalies = [value for value in values if threshold is not None and value > threshold]
        first_row, last_row = rows[0], rows[-1]
        data = {
            "field": field,
            "sample_count": count,
            "minimum": round(min(values), 2),
            "maximum": round(max(values), 2),
            "average": round(mean, 2),
            "first": {"day": first_row["day"], "time": first_row["time"], "value": first_row["value"]},
            "last": {"day": last_row["day"], "time": last_row["time"], "value": last_row["value"]},
            "delta": round(values[-1] - values[0], 2),
            "slope_per_sample": round(slope, 4),
            "threshold": threshold,
            "anomaly_count": len(anomalies),
        }
        return ToolResult(ok=True, data=data, summary=f"analyzed {count} samples")

    def list_alerts(args: dict[str, Any]) -> ToolResult:
        limit = int(args.get("limit", 10))
        clauses: list[str] = []
        params: list[Any] = []
        if args.get("device_id"):
            clauses.append("device = ?")
            params.append(args["device_id"])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = "SELECT time, temp, threshold, device FROM alerts" + where + " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with _connect_readonly(settings.sensor_db_path) as connection:
            rows = [dict(row) for row in connection.execute(sql, params)]
        return ToolResult(ok=True, data={"alerts": rows, "returned": len(rows)}, summary=f"returned {len(rows)} alerts")

    def search_iot_docs(args: dict[str, Any]) -> ToolResult:
        query = args["query"].strip()
        top_k = int(args.get("top_k", settings.rag_top_k))
        results = rag.search(query, top_k=top_k)
        data = {
            "query": query,
            "sources": [
                {
                    "id": chunk.get("id"),
                    "score": round(score, 4),
                    "text": chunk.get("text", ""),
                }
                for score, chunk in results
            ],
        }
        return ToolResult(ok=True, data=data, summary=f"retrieved {len(results)} document chunks")

    def create_alert_rule(args: dict[str, Any]) -> ToolResult:
        settings.sensor_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(settings.sensor_db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device TEXT,
                    field TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    cooldown_sec INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor = connection.execute(
                """
                INSERT INTO alert_rules(device, field, operator, threshold, cooldown_sec, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    args.get("device_id"),
                    args["field"],
                    args["operator"],
                    args["threshold"],
                    args["cooldown_sec"],
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            rule_id = cursor.lastrowid
        return ToolResult(ok=True, data={"rule_id": rule_id}, summary=f"created alert rule #{rule_id}")

    def send_notification(args: dict[str, Any]) -> ToolResult:
        text = args["text"].strip()
        config = {}
        if settings.notify_config_path.exists():
            config = json.loads(settings.notify_config_path.read_text(encoding="utf-8"))
        url = str(config.get("webhook_url", "")).strip()
        if not url:
            return ToolResult(ok=False, error="webhook_url is not configured")
        payload = json.dumps(
            {"msgtype": "text", "text": {"content": text}, "content": text},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=8) as response:
            status = response.status
        return ToolResult(ok=True, data={"status": status}, summary=f"webhook status={status}")

    registry.register(
        ToolSpec(
            name="get_sensor_stats",
            description="Get an aggregate statistic. Use metric=count only to count rows, or max/min/avg for numeric statistics. Do not use for trend or threshold anomaly counts.",
            parameters={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["max", "min", "avg", "count"], "default": "max"},
                    "field": {"type": "string", "enum": ["temp", "hum"], "default": "temp"},
                    "device_id": {"type": ["string", "null"]},
                    "start": {"type": ["string", "null"], "description": "Only set when the user explicitly requests a time range."},
                    "end": {"type": ["string", "null"], "description": "Only set when the user explicitly requests a time range."},
                },
                "additionalProperties": False,
            },
            handler=get_sensor_stats,
        )
    )
    registry.register(
        ToolSpec(
            name="get_sensor_readings",
            description="Read raw sensor rows in a time range for examples or recent samples. Do not use this tool to count the full dataset.",
            parameters={
                "type": "object",
                "properties": {
                    "device_id": {"type": ["string", "null"]},
                    "start": {"type": ["string", "null"], "description": "Only set when the user explicitly requests a time range."},
                    "end": {"type": ["string", "null"], "description": "Only set when the user explicitly requests a time range."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                "additionalProperties": False,
            },
            handler=get_sensor_readings,
        )
    )
    registry.register(
        ToolSpec(
            name="analyze_trend",
            description="Analyze time trend, min/max/average, delta and threshold anomaly count. Use this for trend or how many readings exceed a threshold.",
            parameters={
                "type": "object",
                "properties": {
                    "device_id": {"type": ["string", "null"]},
                    "field": {"type": "string", "enum": ["temp", "hum"], "default": "temp"},
                    "start": {"type": ["string", "null"], "description": "Only set when the user explicitly requests a time range."},
                    "end": {"type": ["string", "null"], "description": "Only set when the user explicitly requests a time range."},
                    "threshold": {"type": ["number", "null"]},
                },
                "additionalProperties": False,
            },
            handler=analyze_trend,
        )
    )
    registry.register(
        ToolSpec(
            name="list_alerts",
            description="List recent sensor alerts, optionally for one device.",
            parameters={
                "type": "object",
                "properties": {
                    "device_id": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "additionalProperties": False,
            },
            handler=list_alerts,
        )
    )
    registry.register(
        ToolSpec(
            name="search_iot_docs",
            description="Search IoT project documentation with vector retrieval.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 500},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "default": settings.rag_top_k},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_iot_docs,
        )
    )
    registry.register(
        ToolSpec(
            name="create_alert_rule",
            description="Create a persistent alert rule. This is a write operation and requires confirmation.",
            parameters={
                "type": "object",
                "properties": {
                    "device_id": {"type": ["string", "null"]},
                    "field": {"type": "string", "enum": ["temp", "hum"]},
                    "operator": {"type": "string", "enum": ["gt", "gte", "lt", "lte"]},
                    "threshold": {"type": "number", "minimum": -100, "maximum": 200},
                    "cooldown_sec": {"type": "integer", "minimum": 10, "maximum": 86400, "default": 300},
                },
                "required": ["field", "operator", "threshold"],
                "additionalProperties": False,
            },
            handler=create_alert_rule,
            permission="write",
            requires_confirmation=True,
        )
    )
    registry.register(
        ToolSpec(
            name="send_notification",
            description="Send a notification to the configured webhook. Requires confirmation.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 1000},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=send_notification,
            permission="write",
            requires_confirmation=True,
        )
    )
    return registry
