"""HybridRAG - 混合检索知识问答系统"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag import RagService
from knowledge_base import KnowledgeBaseService
import config_data as config
import json, os

FILENAMES_PATH = "./uploaded_files.json"


def load_file_list():
    """从 JSON 文件读取已上传的文件名列表"""
    if os.path.exists(FILENAMES_PATH):
        try:
            with open(FILENAMES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_file_list(files):
    """将文件名列表保存到 JSON 文件"""
    with open(FILENAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False)


def count_chroma_docs():
    """获取 Chroma 向量库中的文档片段数量"""
    try:
        vs = st.session_state["rag"].vector_service.vector_store
        return vs._collection.count()
    except:
        return 0

st.set_page_config(page_title="HybridRAG", page_icon="\U0001f50d", layout="wide")

# 初始化会话状态
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "您好！我是 HybridRAG 知识助手，请问有什么可以帮您？"}]
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()
if "kb_service" not in st.session_state:
    st.session_state["kb_service"] = KnowledgeBaseService()
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = load_file_list()

# 侧边栏：知识库管理与文件上传
with st.sidebar:
    st.title("\U0001f4c2 知识库")
    doc_count = count_chroma_docs()
    if doc_count > 0:
        st.caption(f"共 {doc_count} 个内容片段 \u00b7 {len(st.session_state["uploaded_files"])} 个文件")
    if st.session_state["uploaded_files"]:
        st.markdown("**已入库文档：**")
        for f in st.session_state["uploaded_files"]:
            st.markdown(f"- \U0001f4c4 {f}")
    st.markdown("---")
    uploaded_file = st.file_uploader("上传文档", type=["txt"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        text = uploaded_file.getvalue().decode("utf-8")
        with st.spinner("正在处理..."):
            result = st.session_state["kb_service"].upload_by_str(text, file_name)
        if "[Success]" in result:
            if file_name not in st.session_state["uploaded_files"]:
                st.session_state["uploaded_files"].append(file_name)
                save_file_list(st.session_state["uploaded_files"])
            st.success(f"\u2705 {file_name} 已入库")
            st.rerun()
        elif "[Repeat]" in result:
            st.info(f"\u2139\ufe0f {file_name} 已存在")
            if file_name not in st.session_state["uploaded_files"]:
                st.session_state["uploaded_files"].append(file_name)
                save_file_list(st.session_state["uploaded_files"])
                st.rerun()
        else:
            st.error(f"\u274c 处理失败")
    if not st.session_state["uploaded_files"] and doc_count == 0:
        st.caption("暂无文档，请上传 TXT 文件")

# 主界面：聊天问答
st.title("\U0001f4ac HybridRAG - 混合检索知识问答")

for msg in st.session_state["message"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.chat_message("user").markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    ai_res = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # 转圈动画表示正在检索和生成
        with st.spinner("正在检索知识库..."):
            res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
            for chunk in res_stream:
                ai_res.append(chunk)
                placeholder.markdown("".join(ai_res) + "\u258c")
        placeholder.markdown("".join(ai_res))

    st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res)})