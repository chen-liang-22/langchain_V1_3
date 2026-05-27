import os
import uuid
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

dotenv.load_dotenv()

# 1. 连接本地通过 Docker 跑起来的 Qdrant
client = QdrantClient(url=os.getenv("QDRANT_URL"))

# 2. 创建集合（如果不存在）
collection_name = "my_rag_collection"
need_load = not client.collection_exists(collection_name)
if need_load:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print(f"集合 '{collection_name}' 创建成功")
else:
    print(f"集合 '{collection_name}' 已存在，跳过文档写入")

# 3. 初始化向量存储（关联 LangChain 与 Qdrant）
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=DashScopeEmbeddings(),
)

# 4. 仅在首次创建集合时加载并写入文档
if need_load:
    loader = PyPDFLoader(
        file_path="./chat_history/关于启用北森系统并调整钉钉使用范围的通知.pdf",
        mode="page"
    )
    documents = loader.load()
    print(f"加载的文档数量：{len(documents)}")

    vector_store.add_documents(
        documents=documents,
        ids=[str(uuid.uuid4()) for _ in range(len(documents))]
    )
    print("文档已写入 Qdrant 向量库")

# 5. 相似度检索
results = vector_store.similarity_search(
    query="什么时候北森系统全面投入使用",
    k=1
)

print("结果类型:", type(results))
print("结果数量:", len(results))
if results:
    print(results[0].page_content)
else:
    print("未查询到结果")
