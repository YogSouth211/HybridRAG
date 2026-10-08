"""Check that the displayed source documents match those passed into the answer chain."""

import unittest
from unittest.mock import patch

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from rag import RagService


class CountingRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = 0

    def invoke(self, question):
        self.calls += 1
        return self.docs


class FakeVectorService:
    def __init__(self, retriever):
        self.retriever = retriever

    def get_retriever(self, k=None):
        return self.retriever


class RagSourceTests(unittest.TestCase):
    def test_supplied_documents_are_used_without_a_second_retrieval(self):
        docs = [Document(page_content="尺码表：180cm 推荐 2XL", metadata={"source": "尺码推荐.txt"})]
        retriever = CountingRetriever(docs)
        service = object.__new__(RagService)
        service.vector_service = FakeVectorService(retriever)
        captured = []

        def answer_from_prompt(prompt):
            captured.append(prompt.to_messages()[0].content)
            return AIMessage(content="推荐 2XL")

        service.chat_model = RunnableLambda(answer_from_prompt)
        service.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "参考资料：{context}"),
                MessagesPlaceholder("history"),
                ("user", "{input}"),
            ]
        )
        history = InMemoryChatMessageHistory()

        with patch("rag.config.use_hybrid", False), patch("rag.get_history", return_value=history):
            chain = service._RagService__get_chain()
            answer = "".join(chain.stream(
                {"input": "180cm穿什么码", "retrieved_docs": docs},
                {"configurable": {"session_id": "source_test"}},
            ))

        self.assertEqual(answer, "推荐 2XL")
        self.assertEqual(retriever.calls, 0)
        self.assertIn("尺码推荐.txt", captured[0])
        self.assertIn("180cm 推荐 2XL", captured[0])


if __name__ == "__main__":
    unittest.main()
