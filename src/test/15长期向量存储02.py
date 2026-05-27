import os
import uuid

import dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

dotenv.load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"))

collection_name="my_collection"
need = client.collection_exists(collection_name)

if not need:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )
    print(f"集合 '{collection_name}' 创建成功")
else:
    print(f"集合 '{collection_name}' 已存在")


# 3. 初始化向量存储（关联 LangChain 与 Qdrant）
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    # 1. 初始化 embedding 模型（使用阿里 DashScope）
    embedding=DashScopeEmbeddings()
)

if not need:
    # 2. 准备你要存入知识库的文档数据
    documents = [
        "这是一个关于红烧肉做法的文档：首先切块，然后焯水，最后加冰糖上色慢炖。",
        "这是一篇关于如何安装 Windows 11 系统的技术指南。",
        "人工智能（AI）和检索增强生成（RAG）是当下的热门技术。"
    ]
    vector_store.add_texts(
        texts=documents,
        ids=[str(uuid.uuid4()) for _ in range(len(documents))]
    )

    print("文档已写入 Qdrant 向量库")

# 7. 相似度检索
results = vector_store.similarity_search(
    query="什么是人工智能？",
    k=1
)
print("结果类型:", type(results))
print("结果数量:", len(results))
if results:
    print(results[0].page_content)
else:
    print("未查询到结果")
