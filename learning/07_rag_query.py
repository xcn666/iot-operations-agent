# 07_rag_query.py - RAG 第二步：检索 + 生成回答
#
# 流程: 问题向量化 -> 与索引里每个片段算余弦相似度 -> 取最相关的 top_k 段 -> 交给 LLM 生成回答
import json
import math
import os

import requests

try:
    from config import API_KEY, API_URL, MODEL
except ImportError:
    print("请先配置 config.py")
    raise SystemExit(1)

EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBED_MODEL = "embedding-3"
INDEX_PATH = os.path.expanduser("~/iot-lab/ai/rag_index.json")
TOP_K = 3

HEADERS = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}


def embed_one(text):
    resp = requests.post(EMBED_URL, headers=HEADERS,
                         json={"model": EMBED_MODEL, "input": text}, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError("embedding 失败: {} {}".format(resp.status_code, resp.text[:200]))
    return resp.json()["data"][0]["embedding"]


def cosine(a, b):
    """余弦相似度: 纯 Python 实现(不需要 numpy)"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def load_index():
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)["chunks"]


def retrieve(question, chunks, top_k=TOP_K):
    """检索: 返回最相关的 top_k 个片段"""
    qv = embed_one(question)
    scored = []
    for c in chunks:
        scored.append((cosine(qv, c["vector"]), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def answer(question, chunks):
    print("\n① 问题向量化并检索...")
    top = retrieve(question, chunks)
    print("② 检索到最相关的 {} 个片段:".format(len(top)))
    for score, c in top:
        preview = c["text"][:70].replace("\n", " ")
        print("   相似度 {:.3f} | 片段{}: {}...".format(score, c["id"], preview))

    context = "\n\n".join(
        "[片段{}] {}".format(c["id"], c["text"]) for _, c in top
    )
    prompt = """请根据下面提供的资料回答用户问题。

要求：
1. 只依据资料回答，不要编造
2. 如果资料中没有相关信息，直接说"资料中没有提到"
3. 回答简洁，可以指出依据来自哪个片段

资料：
{}

用户问题：{}""".format(context, question)

    print("③ 把检索结果交给 LLM 生成回答...")
    resp = requests.post(API_URL, headers=HEADERS,
                         json={"model": MODEL,
                               "messages": [{"role": "user", "content": prompt}],
                               "temperature": 0.3}, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError("对话失败: {} {}".format(resp.status_code, resp.text[:200]))
    return resp.json()["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        print("索引不存在，请先运行: python3 06_rag_build.py")
        raise SystemExit(1)

    chunks = load_index()
    print("=" * 54)
    print("项目文档 AI 助手（已加载 {} 个片段）".format(len(chunks)))
    print("文档来源: ~/iot-lab/README.md")
    print("试试问: 我的项目用了什么数据库？ / 告警抑制是怎么实现的？")
    print("=" * 54)

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
            print("\nAI:", answer(q, chunks))
        except Exception as e:
            print("[出错]", type(e).__name__, e)
