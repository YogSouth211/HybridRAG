"""三重混合检索器：BM25关键词 + 向量语义 + RRF融合排序"""
import math
import re
from collections import Counter, OrderedDict
from typing import List
from langchain_core.documents import Document
import config_data as config


class HybridRetriever:
    """混合检索器，融合向量检索和BM25关键词检索，通过RRF排序"""

    def __init__(self, vector_retriever, vector_store):
        self.vector_retriever = vector_retriever

        # 从Chroma获取所有文档文本，用于关键词检索
        all_data = vector_store.get(include=["documents", "metadatas"])
        self.all_texts = all_data.get("documents", []) if all_data else []
        self.all_metadatas = all_data.get("metadatas", []) if all_data else []
        self._build_index()

    def _build_index(self):
        """构建BM25索引"""
        self.doc_term_counts = []
        self.df = Counter()
        self.total_docs = len(self.all_texts)
        for text in self.all_texts:
            terms = self._tokenize(text)
            self.doc_term_counts.append(Counter(terms))
            for term in set(terms):
                self.df[term] += 1

    def _tokenize(self, text):
        """简易分词"""
        tokens = re.findall(r"[\w]+", text.lower())
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return tokens + chinese_chars

    def _bm25_score(self, query_terms, doc_idx):
        """BM25评分"""
        doc_len = len(self._tokenize(self.all_texts[doc_idx]))
        total_len = sum(len(self._tokenize(t)) for t in self.all_texts)
        avg_len = total_len / max(self.total_docs, 1)
        k1, b = 1.5, 0.75

        score = 0
        for term in query_terms:
            if term in self.doc_term_counts[doc_idx]:
                tf = self.doc_term_counts[doc_idx][term]
                idf = math.log(
                    (self.total_docs - self.df.get(term, 0) + 0.5)
                    / (self.df.get(term, 0) + 0.5) + 1
                )
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_len, 1)))
        return score

    def _rrf_merge(self, vector_docs, bm25_docs):
        """RRF融合排序（Reciprocal Rank Fusion）"""
        all_items = OrderedDict()
        for rank, doc in enumerate(vector_docs):
            all_items[id(doc)] = {"doc": doc, "score": 1.0 / (60 + rank + 1)}
        for rank, doc in enumerate(bm25_docs):
            doc_id = id(doc)
            if doc_id in all_items:
                all_items[doc_id]["score"] += 1.0 / (60 + rank + 1)
            else:
                all_items[doc_id] = {"doc": doc, "score": 1.0 / (60 + rank + 1)}
        sorted_items = sorted(all_items.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items[:config.final_top_k]]

    def invoke(self, query: str) -> List[Document]:
        """执行混合检索"""
        if not self.all_texts:
            return []

        # 1) 向量检索（找语义相似的）
        vector_results = self.vector_retriever.invoke(query)

        # 2) BM25关键词检索（找关键词匹配的）
        query_terms = self._tokenize(query)
        bm25_results = []
        if query_terms:
            scored = [(i, self._bm25_score(query_terms, i)) for i in range(self.total_docs)]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:config.bm25_top_k]
            bm25_results = [
                Document(
                    page_content=self.all_texts[i],
                    metadata=(self.all_metadatas[i] if i < len(self.all_metadatas) and self.all_metadatas[i] else {})
                ) for i, s in top if s > 0
            ]

        # 3) RRF融合排序
        return self._rrf_merge(vector_results, bm25_results)

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)

