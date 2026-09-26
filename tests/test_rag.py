import json
import tempfile
import unittest
from pathlib import Path

from iot_agent.tools import RagIndex


class KeywordLLM:
    def embed(self, text):
        return [1.0, 0.0]


class RagIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "index.json"
        chunks = [
            {"id": 0, "text": "MQTT 指数退避重连与断网缓存补发", "vector": [1.0, 0.0]},
            {"id": 1, "text": "Flask 与 Chart.js 实时看板", "vector": [1.0, 0.0]},
            {"id": 2, "text": "SQLite readings 与 alerts 双表设计", "vector": [1.0, 0.0]},
        ]
        path.write_text(json.dumps({"model": "test", "chunks": chunks}), encoding="utf-8")
        self.rag = RagIndex(path, KeywordLLM(), top_k=2)

    def tearDown(self):
        self.temp.cleanup()

    def test_keyword_idf_ranks_distinctive_topic_first(self):
        results = self.rag.search("MQTT 断线之后怎么重连", top_k=2)
        self.assertEqual(results[0][1]["id"], 0)

    def test_database_query_ranks_database_chunk_first(self):
        results = self.rag.search("SQLite 保存哪些表", top_k=2)
        self.assertEqual(results[0][1]["id"], 2)


if __name__ == "__main__":
    unittest.main()
