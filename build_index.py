"""Build the local RAG index from bundled or custom documents."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = ROOT / "docs" / "knowledge"
DEFAULT_INDEX = ROOT / "rag_index.json"
DEFAULT_EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
DEFAULT_EMBED_MODEL = "embedding-3"


def load_api_key() -> str:
    api_key = os.getenv("GLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            from config import API_KEY
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "missing embedding API key; set GLM_API_KEY or create config.py from config.example.py"
            ) from exc
        api_key = API_KEY
    if not api_key or "在这里填" in api_key:
        raise RuntimeError("embedding API key is not configured")
    return api_key


def resolve_documents(patterns: Iterable[str] | None = None) -> list[Path]:
    if not patterns:
        patterns = [str(DEFAULT_DOCS_DIR)]

    documents: list[Path] = []
    for raw_pattern in patterns:
        candidate = Path(raw_pattern).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()

        if candidate.is_dir():
            documents.extend(
                path for path in sorted(candidate.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".md", ".txt"}
            )
        elif candidate.is_file():
            documents.append(candidate)
        else:
            raise FileNotFoundError(f"document path not found: {candidate}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in documents:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise FileNotFoundError("no markdown or text documents found")
    return unique


def split_text(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    """Split text into overlapping chunks while preserving paragraph boundaries."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between 0 and chunk size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"\n(?=#{1,6}\s)", "\n\n", normalized)
    normalized = re.sub(r"\n(?=(?:[-*+]\s|\d+\.\s|\|))", "\n\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if not normalized:
        return []
    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= size:
            buffer = (buffer + "\n\n" + paragraph).strip()
            continue

        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= size:
            buffer = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + size, len(paragraph))
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                break
            start += size - overlap
        buffer = ""

    if buffer:
        chunks.append(buffer)
    return chunks


def iter_document_chunks(paths: Iterable[Path], size: int, overlap: int) -> Iterable[dict[str, object]]:
    chunk_id = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT) if ROOT in path.parents else path
        for chunk_index, chunk in enumerate(split_text(text, size=size, overlap=overlap)):
            yield {
                "id": chunk_id,
                "source": relative.as_posix(),
                "chunk_index": chunk_index,
                "text": chunk,
            }
            chunk_id += 1


def embed_texts(
    texts: list[str],
    api_key: str,
    embed_url: str,
    embed_model: str,
    batch_size: int,
    timeout: int,
) -> list[list[float]]:
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }
    vectors: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        batch = texts[index:index + batch_size]
        response = requests.post(
            embed_url,
            headers=headers,
            json={"model": embed_model, "input": batch},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"embedding HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()["data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid embedding response shape") from exc
        data.sort(key=lambda item: item.get("index", 0))
        vectors.extend(item["embedding"] for item in data)
        print(f"embedded {min(index + batch_size, len(texts))}/{len(texts)} chunks")
        time.sleep(0.2)
    return vectors


def build_index(args: argparse.Namespace) -> dict[str, object]:
    documents = resolve_documents(args.doc)
    chunks = list(iter_document_chunks(documents, size=args.chunk_size, overlap=args.overlap))
    if not chunks:
        raise RuntimeError("documents produced no chunks")

    api_key = load_api_key()
    embed_url = args.embed_url or os.getenv("EMBED_URL") or DEFAULT_EMBED_URL
    embed_model = args.embed_model or os.getenv("EMBED_MODEL") or DEFAULT_EMBED_MODEL
    vectors = embed_texts(
        [str(chunk["text"]) for chunk in chunks],
        api_key=api_key,
        embed_url=embed_url,
        embed_model=embed_model,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    if len(vectors) != len(chunks):
        raise RuntimeError("embedding count does not match chunk count")

    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector

    index = {
        "model": embed_model,
        "documents": [
            {"path": str(path.relative_to(ROOT) if ROOT in path.parents else path), "size": path.stat().st_size}
            for path in documents
        ],
        "chunks": chunks,
    }
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return {
        "output": str(output),
        "documents": len(documents),
        "chunks": len(chunks),
        "vector_dimension": len(vectors[0]),
        "model": embed_model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible local RAG index.")
    parser.add_argument(
        "--doc",
        action="append",
        help="Markdown/text file or directory. Repeat for multiple sources. Defaults to docs/knowledge/.",
    )
    parser.add_argument("--output", default=str(DEFAULT_INDEX), help="Output index path.")
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--embed-url", default="")
    parser.add_argument("--embed-model", default="")
    return parser.parse_args()


def main() -> None:
    try:
        summary = build_index(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
