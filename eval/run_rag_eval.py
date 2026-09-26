import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iot_agent.llm import LLMClient
from iot_agent.tools import RagIndex


def load_tasks(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def write_markdown(report, path: Path) -> None:
    lines = [
        '# RAG Retrieval Evaluation',
        '',
        '## Latest run',
        '',
        f"- Embedding model: `{report['embedding_model']}`",
        f"- Index: {report['index_chunks']} document chunks",
        f"- Dataset: {report['total']} manually labeled retrieval questions",
        f"- Top-k: {report['top_k']}",
        f"- Recall@{report['top_k']}: {report['hits']}/{report['total']} ({report['recall_at_k']:.1%})",
        f"- Top-1 accuracy: {report['top_1_correct']}/{report['total']} ({report['top_1_accuracy']:.1%})",
        f"- MRR: {report['mrr']:.4f}",
        '- Scoring: a hit is counted when at least one human-labeled relevant chunk appears in the top-k results.',
        '',
        '| Query | Expected chunk IDs | Retrieved chunk IDs | Hit rank |',
        '|---|---|---|---:|',
    ]
    for record in report['records']:
        expected = ', '.join(str(value) for value in record['expected_ids'])
        retrieved = ', '.join(str(value) for value in record['retrieved_ids'])
        hit_rank = record['hit_rank'] if record['hit_rank'] is not None else 'Miss'
        lines.append(f"| {record['query']} | {expected} | {retrieved} | {hit_rank} |")
    lines.extend([
        '',
        '## Current limitations',
        '',
        '- The dataset is a small development regression set, not a production benchmark.',
        '- The labels accept any relevant chunk, not a strict single gold answer.',
        '- This run evaluates retrieval only, not answer faithfulness or citation correctness.',
        '- The current index uses in-memory cosine plus IDF-weighted keyword retrieval for a small document set.',
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG retrieval quality with labeled questions.')
    parser.add_argument('--tasks', default=str(ROOT / 'eval' / 'rag_tasks.jsonl'))
    parser.add_argument('--index', default=str(ROOT / 'rag_index.json'))
    parser.add_argument('--output', default=str(ROOT / 'eval' / 'rag_results.json'))
    parser.add_argument('--markdown', default=str(ROOT / 'eval' / 'RAG_RESULTS.md'))
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()

    index_path = Path(args.index)
    index_payload = json.loads(index_path.read_text(encoding='utf-8'))
    tasks = load_tasks(Path(args.tasks))
    rag = RagIndex(index_path, LLMClient.from_config(), top_k=args.top_k)

    records = []
    reciprocal_rank = 0.0
    for item in tasks:
        started = time.perf_counter()
        results = rag.search(item['query'], top_k=args.top_k)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        retrieved_ids = [int(chunk['id']) for _, chunk in results]
        scores = [round(float(score), 4) for score, _ in results]
        expected = {int(value) for value in item['expected_ids']}
        hit_rank = next((rank for rank, cid in enumerate(retrieved_ids, 1) if cid in expected), None)
        recall = 1.0 if hit_rank else 0.0
        if hit_rank:
            reciprocal_rank += 1.0 / hit_rank
        records.append({
            'id': item['id'],
            'query': item['query'],
            'expected_ids': sorted(expected),
            'retrieved_ids': retrieved_ids,
            'scores': scores,
            'hit_rank': hit_rank,
            'recall_at_k': recall,
            'elapsed_ms': elapsed_ms,
        })

    total = len(records)
    hits = sum(int(record['recall_at_k']) for record in records)
    top_1_correct = sum(int(record['hit_rank'] == 1) for record in records)
    report = {
        'dataset': str(Path(args.tasks).name),
        'embedding_model': index_payload.get('model', 'unknown'),
        'index_chunks': len(index_payload.get('chunks', [])),
        'top_k': args.top_k,
        'total': total,
        'hits': hits,
        'recall_at_k': round(hits / total, 4) if total else 0,
        'top_1_correct': top_1_correct,
        'top_1_accuracy': round(top_1_correct / total, 4) if total else 0,
        'mrr': round(reciprocal_rank / total, 4) if total else 0,
        'records': records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_markdown(report, Path(args.markdown))
    print(json.dumps({key: report[key] for key in report if key != 'records'}, ensure_ascii=False, indent=2))
    print('detailed results:', output)


if __name__ == '__main__':
    main()
