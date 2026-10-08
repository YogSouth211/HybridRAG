"""The HTTP layer reuses the same retrieved documents as the answer chain."""

import unittest

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import app, get_rag_service


class FakeChain:
    def invoke(self, payload, config):
        assert payload["retrieved_docs"][0].metadata["source"] == "guide.txt"
        assert config["configurable"]["session_id"] == "api_test"
        return "根据资料回答"


class FakeRagService:
    chain = FakeChain()

    def retrieve(self, question, mode):
        assert question == "如何选尺码？"
        assert mode == "bm25"
        return [Document(page_content="选码资料", metadata={"source": "guide.txt"})]


class ApiTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_ask_returns_answer_and_the_same_sources(self):
        response = self.client.post("/ask", json={
            "question": "如何选尺码？", "mode": "bm25", "session_id": "api_test",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "根据资料回答")
        self.assertEqual(response.json()["sources"], [
            {"source": "guide.txt", "content": "选码资料"},
        ])

    def test_invalid_mode_is_rejected(self):
        response = self.client.post("/ask", json={
            "question": "如何选尺码？", "mode": "unknown", "session_id": "api_test",
        })
        self.assertEqual(response.status_code, 422)

    def test_health_does_not_require_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
