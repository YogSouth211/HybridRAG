from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "请仅依据参考资料回答用户问题。"
                 "如果资料不足，请明确说没有查到，不要猜测。参考资料：{context}"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""

        from hybrid_retriever import HybridRetriever
        vector_retriever = self.vector_service.get_retriever(k=config.vector_top_k)
        self.hybrid = None
        if hasattr(self.vector_service, "vector_store"):
            self.hybrid = HybridRetriever(vector_retriever, self.vector_service.vector_store)
        retriever = (
            RunnableLambda(lambda q: self.hybrid.invoke(q))
            if self.hybrid is not None and config.use_hybrid else vector_retriever
        )
        self.retriever = retriever

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def retrieve_for_context(value: dict) -> list[Document]:
            # 页面可传入本次检索的文档，让答案和展示的来源使用同一批片段。
            if "retrieved_docs" in value:
                return value["retrieved_docs"]
            return self.retrieve(value["input"])

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(retrieve_for_context) | format_document
            }| RunnableLambda(format_for_prompt_template) |self.prompt_template |self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(       # 增强的链
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

    def retrieve(self, question: str, mode: str | None = None) -> list[Document]:
        """检索一次，返回用于生成回答的原始文档片段。"""
        if mode is None:
            return self.retriever.invoke(question)
        if self.hybrid is None:
            raise RuntimeError("当前向量存储不支持切换检索方式")
        if mode == "hybrid":
            return self.hybrid.invoke(question)
        if mode == "vector":
            return self.hybrid.search_vector(question)
        if mode == "bm25":
            return self.hybrid.search_bm25(question)
        raise ValueError(f"不支持的检索方式：{mode}")


if __name__ == '__main__':
    # session id 配置
    session_config ={
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
    print(res)

