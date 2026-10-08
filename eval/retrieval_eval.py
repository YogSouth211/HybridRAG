"""Compare BM25, vector, and RRF retrieval on the same labelled questions."""

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from hybrid_retriever import HybridRetriever


DATA_DIR = config.BASE_DIR / "assets"
QUESTIONS_PATH = Path(__file__).with_name("questions.json")
RESULTS_PATH = Path(__file__).with_name("results.json")


def load_cases() -> list[dict[str, str]]:
    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not cases or any(not all(case.get(key) for key in ("question", "source", "phrase")) for case in cases):
        raise ValueError("问题集为空，或缺少 question/source/phrase 字段")
    for case in cases:
        source_path = DATA_DIR / case["source"]
        if not source_path.is_file() or case["phrase"] not in source_path.read_text(encoding="utf-8-sig"):
            raise ValueError(f"标注片段不在示例文档中：{case['source']} / {case['phrase']}")
    return cases


def build_sample_retriever() -> HybridRetriever:
    """Use a temporary collection so evaluation never changes the user's knowledge base."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        length_function=len,
    )
    store = Chroma(
        collection_name="rag_eval",
        client=chromadb.EphemeralClient(),
        embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
    )
    texts, metadatas = [], []
    for path in sorted(DATA_DIR.glob("*.txt")):
        chunks = splitter.split_text(path.read_text(encoding="utf-8-sig"))
        texts.extend(chunks)
        metadatas.extend({"source": path.name} for _ in chunks)
    for case in load_cases():
        if not any(meta["source"] == case["source"] and case["phrase"] in text
                   for text, meta in zip(texts, metadatas)):
            raise ValueError(f"标注片段被切分到不同文本块：{case['source']} / {case['phrase']}")
    store.add_texts(texts, metadatas=metadatas)
    return HybridRetriever(store.as_retriever(search_kwargs={"k": config.vector_top_k}), store)


def relevant_rank(docs, case: dict[str, str]) -> int | None:
    for rank, doc in enumerate(docs, start=1):
        if (doc.metadata or {}).get("source") == case["source"] and case["phrase"] in doc.page_content:
            return rank
    return None


def evaluate(retriever: HybridRetriever, cases: list[dict[str, str]]) -> dict:
    records = []
    for case in cases:
        question = case["question"]
        vector_docs = retriever.search_vector(question, config.vector_top_k)
        bm25_docs = retriever.search_bm25(question, config.bm25_top_k)
        results = {
            "vector": vector_docs[:config.final_top_k],
            "bm25": bm25_docs[:config.final_top_k],
            "hybrid_rrf": retriever._rrf_merge(vector_docs, bm25_docs),
        }
        records.append({
            **case,
            "rank": {mode: relevant_rank(docs, case) for mode, docs in results.items()},
        })
    summary = {}
    for mode in ("vector", "bm25", "hybrid_rrf"):
        ranks = [record["rank"][mode] for record in records]
        summary[mode] = {
            f"Recall@{config.final_top_k}": round(sum(rank is not None for rank in ranks) / len(ranks), 3),
            "MRR": round(sum(1 / rank for rank in ranks if rank is not None) / len(ranks), 3),
        }
    return {"question_count": len(cases), "top_k": config.final_top_k, "summary": summary, "cases": records}


def main() -> None:
    load_dotenv(config.BASE_DIR / ".env")
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("缺少 DASHSCOPE_API_KEY；请先配置项目根目录的 .env")
    cases = load_cases()
    retriever = build_sample_retriever()
    report = evaluate(retriever, cases)
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已评估 {report['question_count']} 道题，Top-K={report['top_k']}")
    recall_name = f"Recall@{report['top_k']}"
    for mode, scores in report["summary"].items():
        print(f"{mode}: {recall_name}={scores[recall_name]:.3f}, MRR={scores['MRR']:.3f}")
    print(f"逐题结果：{RESULTS_PATH}")


if __name__ == "__main__":
    main()
