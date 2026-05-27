import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
dotenv.load_dotenv()

# 1. 初始化向量库
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings()
)

# 初始化PDF加载器
loader = PyPDFLoader(
    file_path="./chat_history/关于启用北森系统并调整钉钉使用范围的通知.pdf",  # PDF文件路径
    mode="page"               # 读取模式：single/page
)

# 加载文档
documents = loader.load()

# 查看结果
# print(f"加载的文档数量：{len(documents)}")
# for document in documents:
#     print(document.page_content)

vector_store.add_documents(
    documents= documents,
    ids = [f"doc_{i}" for i in range(len(documents))]
)

# vector_store.delete(
#    [f"doc_{i}" for i in range(len(documents))]
# )

results = vector_store.similarity_search(
    documents= documents,
    query="什么时候北森系统全面投入使用",
    k=1
)

print(results[0].page_content)


