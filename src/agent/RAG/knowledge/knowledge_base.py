import hashlib
import os
import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.agent.RAG.knowledge.qdrant_store import QdrantStore


class KnowledgeBase:
    """
    知识库管理类
    职责：
    1. 通过 MD5 对文件内容进行去重校验，防止重复导入
    2. 根据文件类型（PDF/TXT）加载文档内容
    3. 判断文档是否需要分割（超过 chunk_size 才分割）
    4. 将文档向量化后存入 Qdrant 向量库
    """

    def __init__(self, md5_file_path: str = None):
        """
        初始化知识库
        :param md5_file_path: MD5记录文件路径，默认为当前目录下的 md5_records.text
        """
        if md5_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            md5_file_path = os.path.join(current_dir, "md5_records.text")
        self.md5_file_path = md5_file_path
        # 确保MD5记录文件存在，不存在则创建空文件
        if not os.path.exists(self.md5_file_path):
            with open(self.md5_file_path, "w", encoding="utf-8") as f:
                pass
        # 初始化向量库（连接Qdrant、加载分割器和向量化模型）
        self.qdrant_store = QdrantStore()

    def to_md5(self, content: str) -> str:
        """
        将字符串内容转成MD5哈希值
        :param content: 需要计算MD5的字符串
        :return: 32位MD5十六进制字符串
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def check_md5(self, md5_value: str) -> bool:
        """
        校验MD5是否已存在于记录文件中
        :param md5_value: 待校验的MD5值
        :return: True-不存在（可以上传），False-已存在（重复文件）
        """
        with open(self.md5_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_value:
                    return False
        return True

    def write_md5(self, md5_value: str):
        """
        将MD5值写入记录文件（写入前会再次校验，防止重复写入）
        :param md5_value: 需要记录的MD5值
        """
        if self.check_md5(md5_value):
            with open(self.md5_file_path, "a", encoding="utf-8") as f:
                f.write(md5_value + "\n")

    def upload_file(self, file_bytes: bytes, file_name: str) -> str:
        """
        上传文件到向量库（核心方法）
        完整流程：
        1. 对文件字节内容计算MD5
        2. 校验MD5是否已存在，重复则跳过
        3. 根据文件后缀选择加载方式（PDF用PyPDFLoader，其他按文本处理）
        4. 判断是否需要分割：任一文档块超过chunk_size则触发分割
        5. 将文档写入Qdrant向量库
        6. 记录MD5到文件，标记为已上传
        :param file_bytes: 文件的原始字节内容
        :param file_name: 文件名（用于判断文件类型）
        :return: 处理结果描述信息
        """
        # 1. 计算文件内容的MD5（基于原始字节，与编码无关）
        md5_value = hashlib.md5(file_bytes).hexdigest()

        # 2. 校验是否重复，已存在则直接返回
        if not self.check_md5(md5_value):
            return f"文件 '{file_name}' 已存在，跳过上传"

        # 3. 根据文件类型选择不同的加载方式
        if file_name.endswith(".pdf"):
            documents = self._load_pdf(file_bytes)
        else:
            documents = self._load_text(file_bytes)

        # 文档内容为空则跳过
        if not documents:
            return f"文件 '{file_name}' 内容为空，跳过上传"

        # 4. 判断是否需要分割：遍历所有文档块，任一超过chunk_size就分割
        chunk_size = self.qdrant_store.text_splitter._chunk_size
        need_split = any(len(doc.page_content) > chunk_size for doc in documents)

        if need_split:
            # 使用RecursiveCharacterTextSplitter按自然段落优先分割
            documents = self.qdrant_store.text_splitter.split_documents(documents)

        # 5. 写入向量库，每个文档块生成唯一ID
        self.qdrant_store.vector_store.add_documents(
            documents=documents,
            ids=[str(uuid.uuid4()) for _ in range(len(documents))]
        )

        # 6. 上传成功后记录MD5，防止下次重复上传
        self.write_md5(md5_value)

        return f"文件 '{file_name}' 上传成功，共 {len(documents)} 个文档块"

    def _load_pdf(self, file_bytes: bytes) -> list[Document]:
        """
        加载PDF文件
        由于PyPDFLoader需要文件路径，先将字节写入临时文件再加载
        :param file_bytes: PDF文件的字节内容
        :return: Document列表，每页一个Document
        """
        # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            return loader.load()
        finally:
            # 无论成功失败都清理临时文件
            os.unlink(tmp_path)

    def _load_text(self, file_bytes: bytes) -> list[Document]:
        """
        加载文本文件
        优先尝试UTF-8解码，失败则降级为GBK（兼容Windows中文环境）
        :param file_bytes: 文本文件的字节内容
        :return: Document列表（单个Document包含全部文本内容）
        """
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # UTF-8解码失败，降级为GBK，忽略无法解码的字节
            content = file_bytes.decode("gbk", errors="ignore")

        # 内容为空则返回空列表
        if not content.strip():
            return []
        return [Document(page_content=content)]




if __name__ == "__main__":
    kb = KnowledgeBase()

    # 测试 to_md5
    content = "这是一段测试内容"
    md5_value = kb.to_md5(content)
    print(f"内容: {content}")
    print(f"MD5: {md5_value}")

    # 测试 check_md5 - 第一次应该返回True（不存在）
    result = kb.check_md5(md5_value)
    print(f"第一次校验（应为True）: {result}")

    # 测试 write_md5 - 写入
    kb.write_md5(md5_value)
    print("已写入MD5")

    # 测试 check_md5 - 第二次应该返回False（已存在）
    result = kb.check_md5(md5_value)
    print(f"第二次校验（应为False）: {result}")

    # 测试文件上传（需要Qdrant服务运行）
    # test_content = b"This is a test document content for vector store."
    # result = kb.upload_file(test_content, "test.txt")
    # print(result)

