"""Streamlit entry point for the HybridRAG knowledge assistant."""

import json
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

import config_data as config
from file_history_store import get_history
from knowledge_base import KnowledgeBaseService
from rag import RagService
from ui_styles import APP_CSS

load_dotenv()

st.set_page_config(page_title="HybridRAG · 知识问答", page_icon="🔎", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

FILENAMES_PATH = config.BASE_DIR / "uploaded_files.json"


def load_file_list() -> list[str]:
    if not FILENAMES_PATH.exists():
        return []
    try:
        with FILENAMES_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return []


def save_file_list(files: list[str]) -> None:
    with FILENAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(files, file, ensure_ascii=False, indent=2)


def start_new_conversation() -> None:
    get_history(st.session_state["session_id"]).clear()
    st.session_state["messages"] = []


if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = load_file_list()
if "upload_version" not in st.session_state:
    st.session_state["upload_version"] = 0
model_ready = bool(os.getenv("DASHSCOPE_API_KEY"))
if model_ready and "kb_service" not in st.session_state:
    st.session_state["kb_service"] = KnowledgeBaseService()
if model_ready and "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

doc_count = st.session_state["kb_service"].chroma._collection.count() if model_ready else 0

with st.sidebar:
    st.markdown(
        '<div class="side-brand"><span class="brand-icon">✦</span>'
        '<div><strong>HybridRAG</strong><small>本地知识库</small></div></div>',
        unsafe_allow_html=True,
    )
    st.metric("已入库片段", doc_count)
    st.caption(f"已记录 {len(st.session_state['uploaded_files'])} 个文档 · 支持 TXT 格式")
    st.divider()
    st.subheader("添加知识")
    uploaded_file = st.file_uploader(
        "上传 UTF-8 编码的 TXT 文档", type=["txt"],
        disabled=not model_ready, key=f"upload_{st.session_state['upload_version']}",
    )
    if uploaded_file is not None:
        try:
            document_text = uploaded_file.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("文件编码无法识别，请保存为 UTF-8 后重新上传。")
        else:
            with st.spinner("正在切分并写入知识库…"):
                result = st.session_state["kb_service"].upload_by_str(
                    document_text, uploaded_file.name
                )
            if result.startswith("[Success]"):
                if uploaded_file.name not in st.session_state["uploaded_files"]:
                    st.session_state["uploaded_files"].append(uploaded_file.name)
                    save_file_list(st.session_state["uploaded_files"])
                st.session_state["upload_notice"] = f"{uploaded_file.name} 已加入知识库"
                st.session_state["upload_version"] += 1
                st.rerun()
            elif result.startswith("[Repeat]"):
                st.info("相同内容已经入库，无需重复上传。")
            else:
                st.warning("文档为空，请上传包含文本的文件。")

    if notice := st.session_state.pop("upload_notice", None):
        st.success(notice)

    st.subheader("已上传文档")
    if st.session_state["uploaded_files"]:
        for filename in st.session_state["uploaded_files"]:
            st.write("📄", filename)
    else:
        st.caption("还没有文档。上传后即可开始提问。")
    st.divider()
    st.button("开启新对话", on_click=start_new_conversation, use_container_width=True)

st.markdown(
    '<section class="hero"><div class="hero-label">知识问答工作台</div>'
    '<h1>让资料，变成答案。</h1>'
    '<p>上传文档后提问，系统会检索相关片段，再依据资料生成回答。</p>'
    '</section>',
    unsafe_allow_html=True,
)

left, middle, right = st.columns(3)
with left:
    st.markdown('<div class="info-card"><b>01　上传资料</b><span>TXT 文档自动切分并存入知识库</span></div>', unsafe_allow_html=True)
with middle:
    st.markdown('<div class="info-card"><b>02　混合检索</b><span>向量语义与 BM25 关键词协同查找</span></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="info-card"><b>03　生成回答</b><span>RRF 融合结果，提供回答参考</span></div>', unsafe_allow_html=True)

st.markdown('<div class="chat-title">对话区 <span>基于当前知识库回答</span></div>', unsafe_allow_html=True)
if not model_ready:
    st.warning("尚未配置阿里云百炼 API Key，请在项目根目录的 .env 中填写后重启。")
elif doc_count == 0:
    st.info("先在左侧上传一份 TXT 文档，再开始提问。")

if not st.session_state["messages"]:
    st.markdown(
        '<div class="welcome-card"><span>✳</span><strong>你好，我是 HybridRAG 知识助手</strong>'
        '<p>可以问我文档中的具体信息，例如“身高 180 厘米推荐什么尺码？”</p></div>',
        unsafe_allow_html=True,
    )

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("输入与已上传文档相关的问题…", disabled=not model_ready or doc_count == 0):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        output = st.empty()
        chunks = []
        try:
            with st.spinner("正在检索资料并生成回答…"):
                session_config = {"configurable": {"session_id": st.session_state["session_id"]}}
                for chunk in st.session_state["rag"].chain.stream(
                    {"input": question}, session_config
                ):
                    chunks.append(chunk)
                    output.markdown("".join(chunks) + "▌")
            answer = "".join(chunks) or "没有生成回答，请换一种问法重试。"
            output.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception:
            output.error("处理问题时出现错误，请检查模型配置或稍后重试。")
