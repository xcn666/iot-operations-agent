# 06_rag_build.py - RAG 第一步：把文档切分 + 向量化 + 建成"索引库"
#
# 流程: 读文档 -> 切分成小块 -> 调 embedding API 转成向量 -> 保存到 rag_index.json
import json
import os
import re
import time

import requests

try:
    from config import API_KEY
except ImportError:
    print("请先: cp config.example.py config.py 并填入 API Key")
    raise SystemExit(1)

EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBED_MODEL = "embedding-3"

DOC_PATH = os.path.expanduser("~/iot-lab/README.md")   # 被检索的文档(可改成自己的)
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_index.json")

CHUNK_SIZE = 220      # 每块大约多少字
OVERLAP = 40          # 相邻块重叠字数(避免切断语义)


def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """简单切分策略: 先按段落聚合, 过长再按字数切开"""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= size:
                buf = p
            else:
                # 超长段落按字数切
                start = 0
                while start < len(p):
                    chunks.append(p[start:start + size])
                    start += size - overlap
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def embed_texts(texts):
    """批量调用 embedding 接口, 返回向量列表"""
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json",
    }
    vectors = []
    batch = 8                                    # 每次最多 8 条, 避免超限
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        resp = requests.post(EMBED_URL, headers=headers,
                             json={"model": EMBED_MODEL, "input": part}, timeout=60)
        if resp.status_code != 200:
            print("embedding 调用失败:", resp.status_code, resp.text[:300])
            raise SystemExit(1)
        data = resp.json()["data"]
        data.sort(key=lambda d: d.get("index", 0))
        vectors.extend([d["embedding"] for d in data])
        print("  已向量化 {}/{} 段".format(min(i + batch, len(texts)), len(texts)))
        time.sleep(0.2)
    return vectors


if __name__ == "__main__":
    print("=" * 52)
    print("RAG 建库：文档 → 切分 → 向量化 → 索引")
    print("=" * 52)
    print("读取文档:", DOC_PATH)
    with open(DOC_PATH, encoding="utf-8") as f:
        text = f.read()
    print("文档长度:", len(text), "字符")

    chunks = split_text(text)
    print("切分成", len(chunks), "个片段")
    for i, c in enumerate(chunks[:3], 1):
        print("  片段{}: {}".format(i, c[:60].replace("\n", " ") + "..."))

    print("\n开始向量化（调用 embedding 接口）...")
    vectors = embed_texts(chunks)
    print("向量维度:", len(vectors[0]))

    index = [{"id": i, "text": c, "vector": v} for i, (c, v) in enumerate(zip(chunks, vectors))]
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": EMBED_MODEL, "chunks": index}, f, ensure_ascii=False)

    size_kb = os.path.getsize(INDEX_PATH) / 1024
    print("\n索引已保存:", INDEX_PATH, "({:.1f} KB)".format(size_kb))
    print("下一步: 运行 07_rag_query.py 提问")
