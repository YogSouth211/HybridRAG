"""Combine semantic and BM25 retrieval with reciprocal rank fusion."""

import math
import re
from collections import Counter

from langchain_core.documents import Document

import config_data as config


class HybridRetriever:
    """Small local corpus retriever; refresh its keyword index when documents change."""

    def __init__(self, vector_retriever, vector_store):
        self.vector_retriever = vector_retriever
        self.vector_store = vector_store
        self.all_texts: list[str] = []
        self.all_metadatas: list[dict] = []
        self.doc_term_counts: list[Counter] = []
        self.doc_lengths: list[int] = []
        self.df: Counter = Counter()
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self._refresh_index()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split English words and numbers; use characters for simple Chinese matching."""
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())

    def _refresh_index(self) -> None:
        # The Streamlit upload service writes to the same Chroma collection later.
        data = self.vector_store.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        if texts == self.all_texts and metadatas == self.all_metadatas:
            return

        self.all_texts = texts
        self.all_metadatas = metadatas
        self.total_docs = len(texts)
        self.doc_term_counts = []
        self.doc_lengths = []
        self.df = Counter()
        for content in texts:
            terms = self._tokenize(content or "")
            self.doc_term_counts.append(Counter(terms))
            self.doc_lengths.append(len(terms))
            self.df.update(set(terms))
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

    def _bm25_score(self, query_terms: set[str], doc_idx: int) -> float:
        k1, b = 1.5, 0.75
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0
        for term in query_terms:
            tf = self.doc_term_counts[doc_idx][term]
            if not tf:
                continue
            idf = math.log(
                1 + (self.total_docs - self.df[term] + 0.5) / (self.df[term] + 0.5)
            )
            denominator = tf + k1 * (1 - b + b * doc_len / max(self.avg_doc_length, 1))
            score += idf * tf * (k1 + 1) / denominator
        return score

    @staticmethod
    def _doc_key(doc: Document) -> tuple[str, str | None]:
        metadata = doc.metadata or {}
        return doc.page_content, metadata.get("source")

    def _rrf_merge(self, vector_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
        """Add rank scores for the same content and source across both result lists."""
        fused: dict[tuple[str, str | None], tuple[Document, float]] = {}
        for docs in (vector_docs, bm25_docs):
            for rank, doc in enumerate(docs, start=1):
                key = self._doc_key(doc)
                previous = fused.get(key)
                fused[key] = (
                    previous[0] if previous else doc,
                    (previous[1] if previous else 0.0) + 1.0 / (60 + rank),
                )
        ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in ranked[:config.final_top_k]]

    def invoke(self, query: str) -> list[Document]:
        self._refresh_index()
        if not self.all_texts:
            return []

        vector_docs = self.vector_retriever.invoke(query)
        query_terms = set(self._tokenize(query))
        scores = [(self._bm25_score(query_terms, i), i) for i in range(self.total_docs)]
        scores.sort(key=lambda item: item[0], reverse=True)
        bm25_docs = [
            Document(
                page_content=self.all_texts[i],
                metadata=self.all_metadatas[i] if i < len(self.all_metadatas) and self.all_metadatas[i] else {},
            )
            for score, i in scores[:config.bm25_top_k] if score > 0
        ]
        return self._rrf_merge(vector_docs, bm25_docs)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)
