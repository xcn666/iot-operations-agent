import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from iot_agent.models import AgentState, AgentStatus, DecisionKind
from iot_agent.planner import Decision, Planner
from iot_agent.runtime import AgentRuntime, Verifier
from iot_agent.settings import Settings
from iot_agent.storage import AgentStore
from iot_agent.tools import ToolRegistry, ToolResult, ToolSpec, build_default_registry, validate_arguments


class DummyLLM:
    def embed(self, text):
        return [1.0, 0.0]


class SequencePlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, state):
        if not self.decisions:
            return Decision(kind=DecisionKind.FINISH, content="done", source="test")
        return self.decisions.pop(0)

    def finalize(self, state):
        return "forced finish"


def make_settings(root: Path) -> Settings:
    return Settings(
        rag_index_path=root / "rag.json",
        agent_db_path=root / "agent.db",
        notify_config_path=root / "notify.json",
        monitor_api_url="http://127.0.0.1:5002",
        monitor_api_token="test-token",
    )


class RuleTests(unittest.TestCase):
    def test_count_routes_to_stats(self):
        decision = Planner._rule_based_decision("现在一共有多少条传感器读数？")
        self.assertEqual(decision.tool_name, "get_sensor_stats")
        self.assertEqual(decision.args["metric"], "count")

    def test_threshold_routes_to_trend(self):
        decision = Planner._rule_based_decision("统计温度超过30度的异常次数")
        self.assertEqual(decision.tool_name, "analyze_trend")
        self.assertEqual(decision.args["threshold"], 30.0)

    def test_explicit_date_is_preserved(self):
        decision = Planner._rule_based_decision("查询2026-09-15的最高温度")
        self.assertEqual(decision.args["start"], "2026-09-15")
        self.assertEqual(decision.args["end"], "2026-09-15")

    def test_missing_device_asks_for_id(self):
        planner = Planner(None, ToolRegistry())
        decision = planner.decide(AgentState.new("分析这个设备的温度趋势"))
        self.assertEqual(decision.kind, DecisionKind.ASK_USER)
        self.assertIn("设备 ID", decision.content)

    def test_multi_intent_skips_fast_path(self):
        decision = Planner._rule_based_decision("先查询最近3条读数，再分析温度趋势")
        self.assertIsNone(decision)

    def test_alert_rule_listing_uses_read_tool(self):
        decision = Planner._rule_based_decision("查看当前告警规则")
        self.assertEqual(decision.tool_name, "list_alert_rules")

    def test_rule_creation_phrase_is_left_for_agent_planning(self):
        decision = Planner._rule_based_decision("创建一个全局温度告警规则，阈值为35度")
        self.assertIsNone(decision)

    def test_rule_update_lists_rules_before_planning_write(self):
        planner = Planner(None, ToolRegistry())
        decision = planner.decide(AgentState.new("把告警规则1的温度阈值调整为35"))
        self.assertEqual(decision.tool_name, "list_alert_rules")
        self.assertFalse(decision.terminal)


class CoverageTests(unittest.TestCase):
    def test_finish_requires_all_task_capabilities(self):
        state = AgentState.new("先看最近3条温度读数，再分析温度趋势")
        decision = Decision(kind=DecisionKind.FINISH, content="done")
        verifier = Verifier()
        registry = ToolRegistry()
        self.assertIn("analyze_trend", verifier.validate(decision, state, registry))
        state.add_step("tool", "get_sensor_readings", {}, {"ok": True})
        self.assertIn("analyze_trend", verifier.validate(decision, state, registry))
        state.add_step("tool", "analyze_trend", {}, {"ok": True})
        self.assertIsNone(verifier.validate(decision, state, registry))


class ToolTests(unittest.TestCase):
    def test_json_schema_validation(self):
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["limit"],
            "additionalProperties": False,
        }
        validate_arguments(schema, {"limit": 3})
        with self.assertRaises(ValueError):
            validate_arguments(schema, {"limit": 0})
        with self.assertRaises(ValueError):
            validate_arguments(schema, {"limit": 1, "extra": True})

    def test_registry_blocks_unconfirmed_write(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="write_test",
                description="test",
                parameters={"type": "object", "properties": {}},
                handler=lambda args: ToolResult(ok=True, data={"called": True}),
                permission="write",
                requires_confirmation=True,
            )
        )
        blocked = registry.execute("write_test", {})
        self.assertFalse(blocked.ok)
        self.assertIn("confirmation required", blocked.error)
        allowed = registry.execute("write_test", {}, confirmed=True)
        self.assertTrue(allowed.ok)


class RuntimeGuardTests(unittest.TestCase):
    def test_hallucinated_time_filter_is_removed(self):
        args = AgentRuntime._sanitize_tool_args("先看最近3条温度读数", {"limit": 3, "start": "now-3h"})
        self.assertEqual(args, {"limit": 3})

    def test_explicit_date_filter_is_kept(self):
        args = AgentRuntime._sanitize_tool_args("查询2026-09-15的温度", {"start": "2026-09-15"})
        self.assertEqual(args["start"], "2026-09-15")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = make_settings(self.root)
        self.registry = build_default_registry(self.settings, DummyLLM())
        self.store = AgentStore(self.settings.agent_db_path)

    def tearDown(self):
        self.temp.cleanup()

    def runtime(self, decisions):
        return AgentRuntime(
            store=self.store,
            registry=self.registry,
            planner=SequencePlanner(decisions),
            max_steps=8,
        )

    def test_multistep_read_and_finish(self):
        runtime = self.runtime(
            [
                Decision(kind=DecisionKind.CALL_TOOL, tool_name="get_sensor_stats", args={"metric": "max", "field": "temp"}),
                Decision(kind=DecisionKind.FINISH, content="最高温度是 25.5 摄氏度。"),
            ]
        )
        with patch(
            "iot_agent.tools._monitor_request",
            return_value={"metric": "max", "field": "temp", "value": 25.5, "sample_count": 1},
        ) as api:
            response = runtime.chat("最高温度是多少")
        self.assertEqual(response["status"], AgentStatus.COMPLETED.value)
        self.assertEqual(response["answer"], "最高温度是 25.5 摄氏度。")
        self.assertTrue(any(step["kind"] == "tool" for step in response["trace"]))
        self.assertEqual(api.call_args.args[1:3], ("GET", "/api/agent/stats"))

    def test_write_requires_confirmation(self):
        runtime = self.runtime(
            [
                Decision(
                    kind=DecisionKind.CALL_TOOL,
                    tool_name="create_alert_rule",
                    args={"field": "temp", "operator": "gt", "threshold": 30, "cooldown_sec": 60},
                ),
                Decision(kind=DecisionKind.FINISH, content="规则已创建。"),
            ]
        )
        first = runtime.chat("温度超过30度时告警")
        self.assertEqual(first["status"], AgentStatus.AWAITING_CONFIRMATION.value)
        self.assertIn("token", first["pending_action"])

        created = {
            "id": 4,
            "device": None,
            "field": "temp",
            "operator": "gt",
            "threshold": 30.0,
            "cooldown_sec": 60,
            "enabled": 1,
            "created_at": "2026-11-02T10:00:00",
        }
        with patch(
            "iot_agent.tools._monitor_request",
            side_effect=[{"rule": created}, {"rules": [created]}],
        ) as api:
            confirmed = runtime.confirm(first["pending_action"]["token"])
        self.assertEqual(confirmed["status"], AgentStatus.COMPLETED.value)
        self.assertEqual(api.call_args_list[0].args[1:3], ("POST", "/api/agent/alert-rules"))
        self.assertEqual(api.call_args_list[1].args[1:3], ("GET", "/api/agent/alert-rules"))

    def test_write_can_be_rejected(self):
        runtime = self.runtime(
            [
                Decision(
                    kind=DecisionKind.CALL_TOOL,
                    tool_name="create_alert_rule",
                    args={"field": "temp", "operator": "gt", "threshold": 30},
                )
            ]
        )
        first = runtime.chat("温度超过30度时告警")
        token = first["pending_action"]["token"]
        with patch("iot_agent.tools._monitor_request") as api:
            rejected = runtime.reject(token)
        self.assertEqual(rejected["status"], AgentStatus.COMPLETED.value)
        self.assertEqual(rejected["answer"], "已取消该写操作。")
        api.assert_not_called()

    def test_rule_update_requires_confirmation_and_is_verified(self):
        old_rule = {
            "id": 1,
            "device": None,
            "field": "temp",
            "operator": "gt",
            "threshold": 30.0,
            "cooldown_sec": 10,
            "enabled": 1,
            "created_at": "2026-11-02T10:00:00",
        }
        runtime = self.runtime(
            [
                Decision(kind=DecisionKind.CALL_TOOL, tool_name="list_alert_rules", args={}),
                Decision(
                    kind=DecisionKind.CALL_TOOL,
                    tool_name="update_alert_rule",
                    args={"rule_id": 1, "threshold": 35},
                )
            ]
        )
        with patch(
            "iot_agent.tools._monitor_request",
            return_value={"rules": [old_rule]},
        ):
            first = runtime.chat("把告警规则1的温度阈值调整为35")
        self.assertEqual(first["status"], AgentStatus.AWAITING_CONFIRMATION.value)

        new_rule = {**old_rule, "threshold": 35.0}
        with patch(
            "iot_agent.tools._monitor_request",
            side_effect=[
                {"rules": [old_rule]},
                {"rule": new_rule},
                {"rules": [new_rule]},
            ],
        ) as api:
            confirmed = runtime.confirm(first["pending_action"]["token"])
        self.assertEqual(confirmed["status"], AgentStatus.COMPLETED.value)
        self.assertEqual(api.call_args_list[1].args[1], "PUT")
        self.assertEqual(api.call_args_list[1].args[2], "/api/agent/alert-rules/1")

    def test_duplicate_tool_loop_is_stopped(self):
        duplicate = Decision(kind=DecisionKind.CALL_TOOL, tool_name="list_alerts", args={"limit": 1})
        runtime = self.runtime([duplicate, duplicate, duplicate])
        response = runtime.chat("重复调用测试")
        self.assertEqual(response["status"], AgentStatus.COMPLETED.value)
        self.assertTrue(any(step["kind"] == "guardrail" for step in response["trace"]))


if __name__ == "__main__":
    unittest.main()
