
md5_path = "./md5.text"

# Chroma
collection_name="rag"
persist_directory="./chroma_db"

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

max_spliter_char_number= 1000  # 文本分割阈值

# 相似度K值
similarity_threshold =1     # 检索返回匹配的文档数量

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
# ��ϼ�������


# 混合检索配置
use_hybrid = True           # True=混合检索，False=只用向量检索
bm25_top_k = 5              # BM25关键词检索取前几条
vector_top_k = 5            # 向量检索取前几条
final_top_k = 3             # 最终RRF融合后取前几条