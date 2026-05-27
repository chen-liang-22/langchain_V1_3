from langchain_community.document_loaders import PyPDFLoader

# 初始化PDF加载器
loader = PyPDFLoader(
    file_path="./chat_history/关于启用北森系统并调整钉钉使用范围的通知.pdf",  # PDF文件路径
    mode="page"               # 读取模式：single/page
)

# 加载文档
documents = loader.load()

# 查看结果
print(f"加载的文档数量：{len(documents)}")
for document in documents:
    print(document.page_content)

