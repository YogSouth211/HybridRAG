"""A small local HTTP interface for the existing RAG answer chain."""

import os
from functools import lru_cache
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import RagService


load_dotenv()
app = FastAPI(title="DocFusion RAG", description="本地混合检索知识库问答接口")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    session_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class Source(BaseModel):
    source: str
    content: str


class AskResponse(BaseModel):
    answer: str
    mode: str
    session_id: str
    sources: list[Source]


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise HTTPException(status_code=503, detail="尚未配置 DASHSCOPE_API_KEY")
    return RagService()


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_configured": bool(os.getenv("DASHSCOPE_API_KEY"))}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    docs = service.retrieve(question, mode=request.mode)
    answer = service.chain.invoke(
        {"input": question, "retrieved_docs": docs},
        {"configurable": {"session_id": request.session_id}},
    )
    return AskResponse(
        answer=answer,
        mode=request.mode,
        session_id=request.session_id,
        sources=[Source(source=str((doc.metadata or {}).get("source") or "未记录文件名"),
                        content=doc.page_content) for doc in docs],
    )
