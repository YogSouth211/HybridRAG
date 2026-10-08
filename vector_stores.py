from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self,embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding= embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self, k=None):
            """返回向量检索器，方便加入chain"""
            if k is None:
                k = config.vector_top_k if config.use_hybrid else config.final_top_k
            return self.vector_store.as_retriever(search_kwargs={"k": k})

if __name__ =='__main__':
        from langchain_community.embeddings import DashScopeEmbeddings
        retriever= VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

        res= retriever.invoke("我的身高180，尺码推荐")
        print(res)
