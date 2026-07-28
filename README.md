# HybridRAG - 混合检索知识问答系统

基于 **LangChain + Chroma + Qwen** 的本地知识库问答系统，支持 **三重混合检索**（BM25 + 向量语义 + RRF 融合排序）。

## 功能

- 📄 上传 TXT 文档，自动切分并存入向量库
- 💬 聊天式问答，基于知识库内容检索增强回答
- 🔀 **三重混合检索**：BM25 关键词 + 向量语义 + RRF 融合排序
- ⚡ 流式输出，逐字显示回答
- 📋 知识库管理，上传文件自动记录

## 技术栈

| 组件 | 说明 |
|------|------|
| Python 3.13 | 运行环境 |
| Streamlit | Web 界面 |
| LangChain | RAG 链编排 |
| Chroma | 向量数据库 |
| DashScope (Qwen) | Embedding + 对话模型 |

## 快速开始

### 1. 配置 API Key

推荐使用阿里云 DashScope（注册 https://dashscope.console.aliyun.com/ 获取 API Key）。
你也可以替换为其他模型，修改 config_data.py 中的 embedding_model_name 和 chat_model_name 即可。

在项目根目录新建 `.env` 文件：

```
DASHSCOPE_API_KEY=sk-你的key
```

### 2. 安装依赖

```bash
pip install streamlit langchain langchain-community langchain-chroma langchain-text-splitters chromadb dashscope python-dotenv python-multipart
```

### 3. 启动

```bash
streamlit run app.py
```

## 项目结构

```
HybridRAG/
├── app.py                 # 主入口（聊天 + 上传）
├── rag.py                 # RAG 链核心
├── hybrid_retriever.py    # 三重混合检索器
├── knowledge_base.py      # 知识库处理
├── vector_stores.py       # 向量库封装
├── file_history_store.py  # 对话历史存储
├── config_data.py         # 配置参数
└── assets/                # 测试文档
```

## 检索方式

| 模式 | 说明 |
|------|------|
| 向量检索 | 语义相似度搜索（默认） |
| BM25 检索 | 关键词匹配搜索 |
| 混合检索 | BM25 + 向量 + RRF 融合（推荐） |

在 `config_data.py` 中修改 `use_hybrid` 可切换检索模式。
