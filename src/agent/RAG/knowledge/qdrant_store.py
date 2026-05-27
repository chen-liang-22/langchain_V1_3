import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()


class QdrantStore:
    """
    Qdrant 向量库操作类
    负责文档的分割、向量化存储和相似度检索
    所有配置项均从 .env 环境变量中读取
    """

    def __init__(self):
        # 集合名称，从环境变量读取
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        # 连接 Qdrant 服务
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"))
        # 向量化模型（阿里 DashScope）
        self.embedding = DashScopeEmbeddings()

        # 文本分割器
        # CHUNK_SIZE: 每个分块的最大字符数
        # CHUNK_OVERLAP: 相邻分块之间的重叠字符数，保证上下文连贯
        # separators: 分割优先级，优先按自然段落划分，逐级降级到标点和空格
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
            separators=os.getenv(
                "CHUNK_SEPARATORS",
                "\n\n,\n,。,！,？,；,.,!,?,;, ,"
            ).split(","),
        )

        # 集合不存在则自动创建（向量维度1536，余弦相似度）
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

        # 初始化 LangChain 向量存储，关联 Qdrant 客户端与向量化模型
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
