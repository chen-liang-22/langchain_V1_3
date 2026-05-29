import os

import dashscope
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

    职责：
    1. 管理与 Qdrant 向量数据库的连接
    2. 提供文本分割能力（RecursiveCharacterTextSplitter）
    3. 封装文档的向量化存储（写入）
    4. 封装基于语义相似度的文档检索（读取）

    所有配置项均从 .env 环境变量中读取，包括：
    - QDRANT_URL: Qdrant 服务地址
    - QDRANT_COLLECTION_NAME: 向量集合名称
    - CHUNK_SIZE: 分块最大字符数
    - CHUNK_OVERLAP: 分块重叠字符数
    - CHUNK_SEPARATORS: 分割符优先级列表
    """

    def __init__(self):
        """
        初始化 Qdrant 向量库
        初始化顺序：连接客户端 -> 配置向量化模型 -> 配置分割器 -> 创建集合 -> 初始化存储
        """
        # 集合名称，从环境变量读取，用于在 Qdrant 中隔离不同业务的向量数据
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")

        # 连接 Qdrant 服务（默认 http://localhost:6333）
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"))

        # 向量化模型（阿里 DashScope Embedding）
        # 输出维度为 1536，与下方创建集合时的 size 参数对应
        self.embedding = DashScopeEmbeddings()

        # 文本分割器（RecursiveCharacterTextSplitter）
        # 工作原理：按 separators 列表从前往后尝试分割，优先保留更大的语义单元
        # 例如：先尝试按段落（\n\n）分割，如果单段超过 chunk_size，再按换行（\n）分割，
        # 再不行就按句号等标点分割，最后才按字符级别分割
        self.text_splitter = RecursiveCharacterTextSplitter(
            # 每个分块的最大字符数，超过此长度会被进一步分割
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            # 相邻分块之间的重叠字符数
            # 作用：防止分割时截断关键信息，保证上下文连贯性
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
            # 分割符优先级列表（从左到右优先级递减）
            # \n\n=段落 > \n=换行 > 。！？；=中文句末 > .!?;=英文句末 > 空格 > 空字符串(逐字)
            separators=os.getenv(
                "CHUNK_SEPARATORS",
                "\n\n,\n,。,！,？,；,.,!,?,;, ,"
            ).split(","),
        )

        # 集合不存在则自动创建
        # size=1536 对应 DashScope Embedding 的输出维度
        # distance=COSINE 余弦相似度，适合语义相似度匹配（值越大越相似）
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

        # 初始化 LangChain 向量存储
        # 作用：将 Qdrant 客户端包装为 LangChain 标准接口
        # 提供 add_documents()、similarity_search() 等标准方法
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )

    def search(self, query: str, k: int = 3) -> list:
        """
        根据查询文本进行相似度检索

        工作原理：
        1. 将 query 通过 DashScope Embedding 转换为 1536 维向量
        2. 在 Qdrant 中计算与所有已存储向量的余弦相似度
        3. 返回相似度最高的 k 个文档

        :param query: 用户查询文本（自然语言）
        :param k: 返回的最相似文档数量，默认3条
                  k值越大召回越多但可能引入噪音，k值越小越精确但可能遗漏信息
        :return: Document 列表，每个 Document 包含：
                 - page_content: 文档文本内容
                 - metadata: 元数据（如来源文件名、页码等）
        """
        return self.vector_store.similarity_search(query=query, k=k)

    def search_with_rerank(self, query: str, initial_k: int = 20, final_k: int = 3) -> list:
        """
        带 Rerank 重排序的两阶段检索

        工作流程：
        1. 第一阶段（粗召回）：用向量相似度检索 initial_k 个候选文档
        2. 第二阶段（精排序）：用 DashScope Rerank 模型对候选文档重新打分
        3. 返回重排后得分最高的 final_k 个文档

        优势：
        - 粗召回阶段覆盖面广（k=20），不容易遗漏相关内容
        - Rerank 阶段精准筛选，过滤掉语义相似但实际不相关的噪音
        - 最终只取 top-3 给 LLM，控制 token 消耗

        :param query: 用户查询文本
        :param initial_k: 粗召回数量，默认20
        :param final_k: 最终返回数量，默认3
        :return: 重排后的 Document 列表（按相关性从高到低）
        """
        # 第一阶段：向量相似度粗召回
        candidates = self.vector_store.similarity_search(query=query, k=initial_k)

        if not candidates:
            return []

        # 第二阶段：调用 DashScope Rerank API 重排序
        # 将候选文档的文本内容提取出来
        documents_text = [doc.page_content for doc in candidates]

        # 调用 DashScope 的 text-reranking 模型
        rerank_response = dashscope.TextReRank.call(
            model="gte-rerank",
            query=query,
            documents=documents_text,
            top_n=final_k,
            return_documents=False  # 只返回索引和分数，不重复返回文本
        )

        # 按 Rerank 返回的索引取出对应的原始 Document
        reranked_docs = []
        if rerank_response.output and rerank_response.output.results:
            for result in rerank_response.output.results:
                idx = result.index
                reranked_docs.append(candidates[idx])

        return reranked_docs
