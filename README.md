# DocFusion RAG｜混合检索知识库问答系统

基于 **LangChain + Chroma + 阿里云百炼 Qwen** 的本地知识库问答系统。用户上传 TXT 后，系统切分、向量化并写入 Chroma；提问时使用向量与 BM25 两路召回，再通过 RRF 融合排序，把相关片段交给模型回答。

## 功能

- 📄 上传 TXT 文档，自动切分并存入向量库
- 💬 聊天式问答，基于知识库内容检索增强回答
- 🔀 **混合检索**：BM25 关键词 + 向量语义，两路结果由 RRF 融合排序
- ⚡ 流式输出，逐段显示回答
- 🔖 回答后可展开查看本次检索的文件名与原文片段
- 🧪 在页面切换混合、向量或 BM25 检索；附带 24 道题的检索对比脚本
- 📋 记录已上传的文件名，以内容 MD5 避免重复入库
- 🧠 按浏览会话保存对话历史到本地文件

**术语提醒：**这里是“两路检索 + 一次融合”，不是三种独立检索器。展示的是本次实际提供给模型的检索片段，并不代表答案每一句都已经核实；目前没有实现文档删除。

## 技术栈

| 组件 | 说明 |
|------|------|
| Python 3.13 | 运行环境 |
| Streamlit | Web 界面 |
| FastAPI | 可选的本地问答接口 |
| LangChain | RAG 链编排 |
| Chroma | 向量数据库 |
| DashScope (Qwen) | Embedding + 对话模型 |

## 快速开始

### 1. 配置 API Key

当前代码固定使用阿里云百炼的 Embedding 与 Qwen 对话模型；`config_data.py` 只配置所用的百炼模型名称。切换其他服务商还需要改模型客户端代码。

在项目根目录新建 `.env` 文件：

```
DASHSCOPE_API_KEY=你的百炼APIKey
```

### 2. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 3. 启动

```powershell
python -m streamlit run app.py
```

## 项目结构

```
DocFusion-RAG/               # GitHub 仓库根目录
├── app.py                 # 主入口（聊天 + 上传）
├── api.py                 # 可选 FastAPI 问答接口
├── rag.py                 # RAG 链核心
├── hybrid_retriever.py    # 两路检索与 RRF 融合
├── ui_styles.py           # 页面样式
├── knowledge_base.py      # 知识库处理
├── vector_stores.py       # 向量库封装
├── file_history_store.py  # 对话历史存储
├── config_data.py         # 配置参数
├── eval/                  # 24 题检索对比及逐题结果
├── docs/learning-guide.md # 按步骤理解项目和面试追问
└── assets/                # 测试文档
```

## 检索方式

| 模式 | 说明 |
|------|------|
| 向量检索 | 语义相似度搜索 |
| BM25 检索 | 关键词匹配搜索 |
| 混合检索 | BM25 + 向量 + RRF 融合（推荐） |

页面左侧可切换混合、向量或 BM25 检索，默认启用混合检索。`config_data.py` 中的 `use_hybrid` 控制代码直接调用问答链时的默认模式。页面为每个浏览会话分配独立的对话 ID。`.streamlit/config.toml` 将服务限制在本机，并将单次上传限制为 10 MB。

### 检索对比

在已配置百炼 API Key 的环境中运行：

```powershell
python -m eval.retrieval_eval
```

脚本将 `assets/` 中的 3 份示例文档放入临时 Chroma 集合，对 24 道标注了目标片段的问题计算 Recall@3 和 MRR，不会改动页面知识库。逐题名次保存到 `eval/results.json`。这组小样例主要用于检查代码与学习指标，不能代表真实业务数据。当前结果：三种方式的 Recall@3 都为 1.000；MRR 分别为向量 0.861、BM25 1.000、混合 0.896，**没有观察到混合检索优于 BM25**。

想理解从上传到回答的代码路径，请看 [学习指南与面试准备](docs/learning-guide.md)。

当前仅支持 UTF-8 TXT 文档。BM25 使用本地简易中英文分词并在查询时检查知识库变化，适合小规模演示；大量文档应使用专门的全文索引。正式判断检索收益还需要更大、更难、与业务一致的问题集。

### 可选：本地问答 API

在另一个终端启动：

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs` 可查看接口。`POST /ask` 接收问题、检索方式（`hybrid`、`vector`、`bm25`）和可选会话 ID，返回回答及本次检索片段；`GET /health` 只报告服务和模型配置状态。接口读取与页面相同的本地 Chroma 知识库，不负责上传文档。请先在 Streamlit 页面上传资料。该接口只用于本机学习演示。

请求示例：

```json
{"question":"身高180厘米推荐什么尺码？","mode":"hybrid"}
```

## 来源与许可

本项目基于 [lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System) 的 MIT 许可代码学习和改造，保留原项目的 [LICENSE](LICENSE) 与版权声明；此版本增加了混合检索、统一页面、会话隔离及相关说明。
