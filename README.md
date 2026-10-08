# HybridRAG - 混合检索知识问答系统

基于 **LangChain + Chroma + 阿里云百炼 Qwen** 的本地知识库问答系统。用户上传 TXT 后，系统切分、向量化并写入 Chroma；提问时使用向量与 BM25 两路召回，再通过 RRF 融合排序，把相关片段交给模型回答。

## 功能

- 📄 上传 TXT 文档，自动切分并存入向量库
- 💬 聊天式问答，基于知识库内容检索增强回答
- 🔀 **混合检索**：BM25 关键词 + 向量语义，两路结果由 RRF 融合排序
- ⚡ 流式输出，逐段显示回答
- 📋 记录已上传的文件名，以内容 MD5 避免重复入库
- 🧠 按浏览会话保存对话历史到本地文件

**术语提醒：**这里是“两路检索 + 一次融合”，不是三种独立检索器。当前页面没有单独展示引用片段，也没有实现文档删除或检索效果的量化评测。

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
HybridRAG/
├── app.py                 # 主入口（聊天 + 上传）
├── rag.py                 # RAG 链核心
├── hybrid_retriever.py    # 两路检索与 RRF 融合
├── ui_styles.py           # 页面样式
├── knowledge_base.py      # 知识库处理
├── vector_stores.py       # 向量库封装
├── file_history_store.py  # 对话历史存储
├── config_data.py         # 配置参数
├── docs/learning-guide.md # 按步骤理解项目和面试追问
└── assets/                # 测试文档
```

## 检索方式

| 模式 | 说明 |
|------|------|
| 向量检索 | 语义相似度搜索 |
| BM25 检索 | 关键词匹配搜索 |
| 混合检索 | BM25 + 向量 + RRF 融合（推荐） |

在 `config_data.py` 中修改 `use_hybrid` 可切换检索模式。默认启用混合检索。页面为每个浏览会话分配独立的对话 ID。`.streamlit/config.toml` 将服务限制在本机，并将单次上传限制为 10 MB。

想理解从上传到回答的代码路径，请看 [学习指南与面试准备](docs/learning-guide.md)。

当前仅支持 UTF-8 TXT 文档。BM25 使用本地简易中英文分词并在查询时检查知识库变化，适合小规模演示；大量文档应使用专门的全文索引。检索效果尚未经过定量评估，不能据此声称准确率提升。

## 来源与许可

本项目基于 [lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System) 的 MIT 许可代码学习和改造，保留原项目的 [LICENSE](LICENSE) 与版权声明；此版本增加了混合检索、统一页面、会话隔离及相关说明。
