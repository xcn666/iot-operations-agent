import tempfile
import unittest
from pathlib import Path

from build_index import iter_document_chunks, resolve_documents, split_text


class BuildIndexTests(unittest.TestCase):
    def test_split_text_keeps_short_document_as_one_chunk(self):
        self.assertEqual(split_text("短文档", size=220), ["短文档"])

    def test_split_text_creates_overlapping_long_paragraph(self):
        chunks = split_text("a" * 25, size=10, overlap=2)
        self.assertEqual([len(chunk) for chunk in chunks], [10, 10, 9])
        self.assertEqual(chunks[0][-2:], chunks[1][:2])

    def test_split_text_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_text("content", size=10, overlap=10)

    def test_default_documents_are_bundled_with_repository(self):
        documents = resolve_documents()
        self.assertTrue(documents)
        self.assertTrue(any(path.name == "iot-monitor.md" for path in documents))

    def test_iter_document_chunks_adds_source_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.md"
            second = Path(directory) / "second.md"
            first.write_text("第一段\n\n第二段", encoding="utf-8")
            second.write_text("第三段", encoding="utf-8")
            chunks = list(iter_document_chunks([first, second], size=220, overlap=20))
        self.assertEqual([chunk["id"] for chunk in chunks], [0, 1])
        self.assertEqual(chunks[0]["source"], str(first))
        self.assertEqual(chunks[1]["source"], str(second))


if __name__ == "__main__":
    unittest.main()
