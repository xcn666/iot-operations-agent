"""Flask entrypoints for the IoT Operations Agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from .llm import LLMClient
from .planner import Planner
from .runtime import AgentRuntime
from .settings import Settings
from .storage import AgentStore
from .tools import build_default_registry


def create_app(runtime: AgentRuntime | None = None) -> Flask:
    root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, template_folder=str(root / "templates"))

    if runtime is None:
        settings = Settings.from_env()
        llm = LLMClient.from_config()
        registry = build_default_registry(settings, llm)
        store = AgentStore(settings.agent_db_path)
        runtime = AgentRuntime(
            store=store,
            registry=registry,
            planner=Planner(llm, registry),
            max_steps=settings.max_steps,
        )

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/health")
    def health() -> Any:
        return jsonify(
            {
                "ok": True,
                "agent": "IoT Operations Agent",
                "tools": runtime.registry.names(),
                "max_steps": runtime.max_steps,
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def api_chat() -> Any:
        payload = request.get_json(silent=True) or {}
        task = str(payload.get("question") or payload.get("task") or "").strip()
        session_id = str(payload.get("session_id") or "").strip() or None
        try:
            return jsonify(runtime.chat(task, session_id=session_id))
        except Exception as exc:
            return jsonify({"status": "failed", "answer": "", "error": f"{type(exc).__name__}: {exc}", "trace": []}), 500

    @app.route("/api/confirm", methods=["POST"])
    def api_confirm() -> Any:
        payload = request.get_json(silent=True) or {}
        token = str(payload.get("token") or "").strip()
        if not token:
            return jsonify({"status": "failed", "error": "missing confirmation token", "trace": []}), 400
        try:
            return jsonify(runtime.confirm(token))
        except Exception as exc:
            return jsonify({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "trace": []}), 500

    @app.route("/api/reject", methods=["POST"])
    def api_reject() -> Any:
        payload = request.get_json(silent=True) or {}
        token = str(payload.get("token") or "").strip()
        if not token:
            return jsonify({"status": "failed", "error": "missing confirmation token", "trace": []}), 400
        try:
            return jsonify(runtime.reject(token))
        except Exception as exc:
            return jsonify({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "trace": []}), 500

    @app.route("/api/session/<session_id>")
    def api_session(session_id: str) -> Any:
        state = runtime.store.load(session_id)
        if not state:
            return jsonify({"status": "not_found", "error": "session not found", "trace": []}), 404
        return jsonify(runtime._response(state))

    return app
