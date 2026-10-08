"""Regression checks for document refresh and result fusion."""

import unittest
from uuid import uuid4

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hybrid_retriever import HybridRetriever


class FakeStore:
    def __init__(self):
        self.docs = []

    def get(self, include=None):
        return {
            "documents": [doc.page_content for doc in self.docs],
            "metadatas": [doc.metadata for doc in self.docs],
        }


class FakeVectorRetriever:
    def __init__(self, store):
        self.store = store

    def invoke(self, query):
        # Chroma returns new Document instances for the same stored fragments.
        return [Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc in self.store.docs]


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float("尺" in text), float("码" in text), float("报" in text)]


class HybridRetrieverTests(unittest.TestCase):
    def test_document_added_after_initialization_is_searchable(self):
        store = FakeStore()
        retriever = HybridRetriever(FakeVectorRetriever(store), store)
        store.docs.append(Document(page_content="报表导出说明", metadata={"source": "guide.txt"}))
        results = retriever.invoke("导出报表")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "报表导出说明")

    def test_same_fragment_from_both_retrievers_is_merged(self):
        store = FakeStore()
        store.docs.append(Document(page_content="尺码推荐表", metadata={"source": "sizes.txt"}))
        retriever = HybridRetriever(FakeVectorRetriever(store), store)
        results = retriever.invoke("尺码推荐")
        self.assertEqual(len(results), 1)

    def test_empty_collection_returns_no_results(self):
        store = FakeStore()
        retriever = HybridRetriever(FakeVectorRetriever(store), store)
        self.assertEqual(retriever.invoke("尺码"), [])

    def test_single_retrieval_modes_use_the_same_documents(self):
        store = FakeStore()
        store.docs.extend([
            Document(page_content="尺码推荐表", metadata={"source": "sizes.txt"}),
            Document(page_content="洗涤说明", metadata={"source": "care.txt"}),
        ])
        retriever = HybridRetriever(FakeVectorRetriever(store), store)
        self.assertEqual(retriever.search_bm25("尺码", 1)[0].metadata["source"], "sizes.txt")
        self.assertEqual(retriever.search_vector("尺码", 1)[0].metadata["source"], "sizes.txt")

    def test_two_chroma_clients_see_new_upload(self):
        client = chromadb.EphemeralClient()
        collection_name = f"rag_{uuid4().hex}"
        reader = Chroma(collection_name=collection_name, client=client,
                        embedding_function=FakeEmbeddings())
        writer = Chroma(collection_name=collection_name, client=client,
                        embedding_function=FakeEmbeddings())
        retriever = HybridRetriever(reader.as_retriever(search_kwargs={"k": 5}), reader)
        writer.add_texts(["尺码推荐表"], metadatas=[{"source": "sizes.txt"}])
        results = retriever.invoke("尺码")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "尺码推荐表")


if __name__ == "__main__":
    unittest.main()
