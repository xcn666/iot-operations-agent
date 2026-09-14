# 08_ai_web.py - 网页版 AI 助手：RAG + Function Calling 二合一
#
# 架构:
#   用户提问 -> LLM 判断意图(action) -> 调工具查数据库 / RAG 检索文档 / 直接回答 -> LLM 组织语言
import ast
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta

import requests
from flask import Flask, jsonify, render_template, request

try:
    from config import API_KEY, API_URL, MODEL
except ImportError:
    print("请先配置 config.py"); raise SystemExit(1)

EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBED_MODEL = "embedding-3"
DB = os.path.expanduser("~/iot-lab/logs/sensor.db")
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_index.json")
HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

app = Flask(__name__)

# ---------- 载入 RAG 索引 ----------
RAG_CHUNKS = []
if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, encoding="utf-8") as f:
        RAG_CHUNKS = json.load(f)["chunks"]


# ---------- 工具（Function Calling）----------
def get_stats(metric="max", field="temp", day=None):
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
            sql += " WHERE day = ?"; params = (day,)
        value = conn.execute(sql, params).fetchone()[0]
        return {"metric": metric, "field": field, "day": day or "全部", "value": value}
    finally:
        conn.close()


def list_alerts(limit=5):
    conn = sqlite3.connect(DB)
    try:
        rows = conn.execute("SELECT time, temp, threshold, device FROM alerts ORDER BY id DESC LIMIT ?",
                            (int(limit),)).fetchall()
        return [{"time": r[0], "temp": r[1], "threshold": r[2], "device": r[3]} for r in rows]
    finally:
        conn.close()


TOOLS = {"get_stats": get_stats, "list_alerts": list_alerts}


# ---------- 基础能力 ----------
def strip_json(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()



def normalize_text(t):
    """把中文全角标点统一成 ASCII，解决中文模型输出的常见问题"""
    table = {
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',
        "\u2018": "'", "\u2019": "'",
        "\uff5b": "{", "\uff5d": "}", "\uff08": "(", "\uff09": ")",
        "\uff3b": "[", "\uff3d": "]",
        "\uff0c": ",", "\uff1a": ":", "\uff1b": ";",
        "\u3000": " ",
        "\ufeff": "", "\u200b": "", "\u200c": "", "\u200d": "",
    }
    for k, v in table.items():
        t = t.replace(k, v)
    return t


def parse_json_lenient(text):
    """健壮解析: 归一化 -> 标准JSON -> 提取花括号 -> Python字面量 -> 正则兜底"""
    import re as _re
    t = normalize_text(strip_json(text))
    # 1) 标准 JSON
    try:
        return json.loads(t)
    except Exception:
        pass
    # 2) 提取第一个 {...}
    start, end = t.find("{"), t.rfind("}")
    cand = t[start:end + 1] if (start != -1 and end > start) else t
    try:
        return json.loads(cand)
    except Exception:
        pass
    # 3) Python 字面量（单引号）
    try:
        return ast.literal_eval(cand)
    except Exception:
        pass
    # 4) 正则兜底：直接抠出 action / tool / search
    out = {}
    m = _re.search(r'''action\s*[":=]+\s*["']?(\w+)''', cand)
    if m:
        out["action"] = m.group(1)
    m = _re.search(r'''tool\s*[":=]+\s*["']?(\w+)''', cand)
    if m:
        out["tool"] = m.group(1)
    m = _re.search(r'''search\s*[":=]+\s*["']([^"']+)''', cand)
    if m:
        out["search"] = m.group(1)
    m = _re.search(r'''metric\s*[":=]+\s*["']?(\w+)''', cand)
    if m:
        out.setdefault("args", {})["metric"] = m.group(1)
    m = _re.search(r'''field\s*[":=]+\s*["']?(\w+)''', cand)
    if m:
        out.setdefault("args", {})["field"] = m.group(1)
    m = _re.search(r'''day\s*[":=]+\s*["']([0-9\-]+)''', cand)
    if m:
        out.setdefault("args", {})["day"] = m.group(1)
    if out.get("action"):
        return out
    raise ValueError("无法解析模型输出: " + repr(t[:200]))


def chat_llm(messages, temperature=0.3):
    r = requests.post(API_URL, headers=HEADERS,
                      json={"model": MODEL, "messages": messages, "temperature": temperature}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError("LLM 错误 {}: {}".format(r.status_code, r.text[:200]))
    return r.json()["choices"][0]["message"]["content"]


def embed_one(text):
    r = requests.post(EMBED_URL, headers=HEADERS,
                      json={"model": EMBED_MODEL, "input": text}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError("embedding 失败: {}".format(r.text[:200]))
    return r.json()["data"][0]["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def retrieve(question, top_k=3):
    if not RAG_CHUNKS:
        return []
    qv = embed_one(question)
    scored = [(cosine(qv, c["vector"]), c) for c in RAG_CHUNKS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ---------- Agent 主流程 ----------
def run_agent(question):
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    system = (
        "你是物联网数据与项目文档智能助手。\n"
        "今天是 " + today + "，昨天是 " + yesterday + "。\n\n"
        "请判断用户问题属于哪一类，并只输出 JSON（不要解释、不要 markdown 代码块、字符串必须用英文双引号）：\n\n"
        '1) 查询传感器数据库的统计值或告警记录 -> '
        '{"action":"tool","tool":"get_stats","args":{"metric":"max","field":"temp","day":"' + yesterday + '"}}\n'
        '   工具: get_stats(metric=max/min/avg/count, field=temp/hum, day=YYYY-MM-DD)；'
        'list_alerts(limit=5)\n\n'
        '2) 查询项目文档 / 技术实现细节 -> {"action":"doc","search":"关键词"}\n\n'
        '3) 其他（闲聊、通用知识）-> {"action":"chat"}\n\n'
        '判断规则：问温度/湿度/告警/数据条数 -> tool；问怎么实现的/用了什么技术/项目介绍 -> doc。'
    )

    plan_raw = chat_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ], temperature=0.1)
    try:
        plan = parse_json_lenient(plan_raw)
    except Exception as e:
        print('[解析失败] 模型原始输出:', repr(plan_raw[:300]))
        print('[诊断] 前80字符编码:', [hex(ord(c)) for c in plan_raw[:80]])
        raise
    action = plan.get("action", "chat")
    trace = {"action": action}

    if action == "tool":
        name = plan.get("tool", "get_stats")
        args = plan.get("args", {}) or {}
        trace["tool"] = name; trace["args"] = args
        result = TOOLS.get(name, lambda **k: {"error": "未知工具"})(**args)
        trace["result"] = result
        answer = chat_llm([
            {"role": "system", "content": "你是物联网数据分析助手。根据数据用简洁中文回答，只依据数据，不要编造。"},
            {"role": "user", "content": "问题：{}\n查询结果：{}".format(question, json.dumps(result, ensure_ascii=False))},
        ])

    elif action == "doc":
        query = plan.get("search") or question
        top = retrieve(query)
        trace["search"] = query
        trace["sources"] = [{"id": c["id"], "score": round(s, 3),
                             "preview": c["text"][:60].replace("\n", " ")} for s, c in top]
        context = "\n\n".join("[片段{}] {}".format(c["id"], c["text"]) for _, c in top)
        answer = chat_llm([
            {"role": "system", "content": "根据资料回答问题，只依据资料，没有就说'资料中没有提到'，回答简洁。"},
            {"role": "user", "content": "资料：\n{}\n\n问题：{}".format(context, question)},
        ])

    else:
        answer = chat_llm([
            {"role": "system", "content": "你是物联网与 AI 方向的助手，回答简洁专业。"},
            {"role": "user", "content": question},
        ], temperature=0.7)

    return answer.strip(), trace


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "请输入问题", "trace": {}})
    try:
        answer, trace = run_agent(question)
        return jsonify({"answer": answer, "trace": trace})
    except Exception as e:
        return jsonify({"answer": "处理出错: {}: {}".format(type(e).__name__, e), "trace": {}})


if __name__ == "__main__":
    print("=" * 52)
    print("物联网 AI 助手已启动: http://localhost:5001")
    print("RAG 片段数:", len(RAG_CHUNKS))
    print("=" * 52)
    app.run(host="0.0.0.0", port=5001, debug=False)
