# 05_agent_query.py - AI Agent: 用自然语言查询你的 IoT 传感器数据库
#
# 核心机制（Function Calling / 工具调用）:
#   用户提问 → LLM 决定调用哪个工具(输出JSON) → Python 执行工具 → 结果回给 LLM → LLM 生成回答
#   注意: LLM 只负责"决策"和"组织语言", 真正查数据库的是 Python 代码
import ast
import json
import os
import sqlite3
from datetime import datetime, timedelta

import requests

try:
    from config import API_KEY, API_URL, MODEL
except ImportError:
    print("请先: cp config.example.py config.py 并填入 API Key")
    raise SystemExit(1)

DB = os.path.expanduser("~/iot-lab/logs/sensor.db")
TODAY = datetime.now().strftime("%Y-%m-%d")
YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


# ============ 1. 定义"工具"：真正干活的是这些 Python 函数 ============
def get_stats(metric="max", field="temp", day=None):
    """查询统计值。metric: max/min/avg/count；field: temp/hum；day: YYYY-MM-DD"""
    conn = sqlite3.connect(DB)
    try:
        if metric == "count":
            expr = "COUNT(*)"
        else:
            fn = {"max": "MAX", "min": "MIN", "avg": "AVG"}.get(metric, "MAX")
            expr = "ROUND({}({}), 2)".format(fn, "temp" if field == "temp" else "hum")
        sql = "SELECT {} FROM readings".format(expr)
        params = ()
        if day:
            sql += " WHERE day = ?"
            params = (day,)
        value = conn.execute(sql, params).fetchone()[0]
        return {"metric": metric, "field": field, "day": day or "全部", "value": value}
    finally:
        conn.close()


def list_alerts(limit=5):
    """查询最近的告警记录"""
    conn = sqlite3.connect(DB)
    try:
        rows = conn.execute(
            "SELECT time, temp, threshold, device FROM alerts ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [{"time": r[0], "temp": r[1], "threshold": r[2], "device": r[3]} for r in rows]
    finally:
        conn.close()


TOOLS = {
    "get_stats": get_stats,
    "list_alerts": list_alerts,
}

TOOL_DOC = """可用工具：
1. get_stats(metric, field, day)
   - metric: "max"(最大) / "min"(最小) / "avg"(平均) / "count"(条数)
   - field : "temp"(温度) / "hum"(湿度)
   - day   : 日期字符串，格式 "YYYY-MM-DD"；不传则统计全部数据
   例：查某天最高温度 -> {"metric": "max", "field": "temp", "day": "2026-09-13"}

2. list_alerts(limit)
   - limit: 返回最近几条告警，默认 5
   例：查最近告警 -> {"limit": 5}"""


def strip_json(text):
    """剥离 markdown 代码块, 返回纯文本"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()



def parse_json_lenient(text):
    """尽量解析模型输出: 先标准 JSON, 再尝试从文本中提取 JSON, 最后尝试 Python 字面量(单引号)"""
    t = strip_json(text)
    # 1) 直接标准解析
    try:
        return json.loads(t)
    except Exception:
        pass
    # 2) 提取第一个 {...} 再解析
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        cand = t[start:end + 1]
        try:
            return json.loads(cand)
        except Exception:
            pass
        # 3) 模型可能用了单引号 -> 用 Python 字面量解析
        try:
            return ast.literal_eval(cand)
        except Exception:
            pass
    raise ValueError("无法解析模型输出: " + repr(t[:200]))


def call_llm(messages, temperature=0.1):
    resp = requests.post(
        API_URL,
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "temperature": temperature},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError("API 错误 {}: {}".format(resp.status_code, resp.text[:200]))
    return resp.json()["choices"][0]["message"]["content"]


# ============ 2. Agent 主流程 ============
def run_agent(question):
    print("\n" + "-" * 52)
    print("① 用户提问:", question)

    # 步骤1: 让 LLM 决定调用哪个工具（只输出 JSON）
    system1 = """你是一个物联网数据分析助手。今天是 {today}，昨天是 {yesterday}。

{tools}

请判断用户问题需要调用哪个工具，并且**只输出 JSON**，不要解释、不要 markdown 代码块、**字符串必须用双引号**：
{{"tool": "工具名", "args": {{参数}}}}

如果问题不需要查询数据（例如打招呼），输出：
{{"tool": null, "answer": "你的直接回答"}}""".format(today=TODAY, yesterday=YESTERDAY, tools=TOOL_DOC)

    raw = call_llm([
        {"role": "system", "content": system1},
        {"role": "user", "content": question},
    ])
    try:
        plan = parse_json_lenient(raw)
    except Exception as e:
        print('[解析失败] 模型原始输出:', repr(raw[:300]))
        raise
    print("② LLM 决定调用工具:", json.dumps(plan, ensure_ascii=False))

    if not plan.get("tool"):
        return plan.get("answer", "(没有回答)")

    # 步骤2: Python 真正执行工具
    tool_name = plan["tool"]
    args = plan.get("args", {}) or {}
    if tool_name not in TOOLS:
        return "模型请求了不存在的工具: " + str(tool_name)
    result = TOOLS[tool_name](**args)
    print("③ Python 执行 {}({})  ->  {}".format(tool_name, json.dumps(args, ensure_ascii=False),
                                                 json.dumps(result, ensure_ascii=False)))

    # 步骤3: 把执行结果交回 LLM，生成自然语言回答
    system2 = """你是物联网数据分析助手。请根据工具返回的数据，用简洁自然的中文回答用户问题。
只依据给定数据回答，不要编造。如果数据为 null，请说明没有查到数据。"""
    final = call_llm([
        {"role": "system", "content": system2},
        {"role": "user", "content": "用户问题：{}\n工具返回的数据：{}".format(
            question, json.dumps(result, ensure_ascii=False))},
    ], temperature=0.3)
    print("④ LLM 生成回答")
    return final.strip()


if __name__ == "__main__":
    print("=" * 52)
    print("IoT 数据 AI 助手（输入 exit 退出）")
    print("试试问：昨天最高温度是多少？ / 最近有哪些告警？")
    print("=" * 52)
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if q.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            break
        if not q:
            continue
        try:
            print("\nAI:", run_agent(q))
        except Exception as e:
            print("[出错]", type(e).__name__, e)
