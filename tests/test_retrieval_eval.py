"""Checks for retrieval ranking metrics without calling an external model."""

import unittest

from langchain_core.documents import Document

from eval.retrieval_eval import evaluate


class FakeRetriever:
    def __init__(self):
        self.irrelevant = Document(page_content="无关片段", metadata={"source": "other.txt"})
        self.relevant = Document(page_content="正确答案片段", metadata={"source": "guide.txt"})

    def search_vector(self, question, top_k):
        return [self.irrelevant, self.relevant]

    def search_bm25(self, question, top_k):
        return [self.relevant, self.irrelevant]

    def _rrf_merge(self, vector_docs, bm25_docs):
        return [self.relevant, self.irrelevant]


class RetrievalEvalTests(unittest.TestCase):
    def test_recall_and_reciprocal_rank(self):
        report = evaluate(FakeRetriever(), [{
            "question": "问题", "source": "guide.txt", "phrase": "正确答案",
        }])
        self.assertEqual(report["summary"]["vector"]["Recall@3"], 1.0)
        self.assertEqual(report["summary"]["vector"]["MRR"], 0.5)
        self.assertEqual(report["summary"]["bm25"]["MRR"], 1.0)
        self.assertEqual(report["summary"]["hybrid_rrf"]["MRR"], 1.0)


if __name__ == "__main__":
    unittest.main()
