import hashlib
import os
import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.agent.RAG.knowledge.qdrant_store import QdrantStore
from src.agent.my_llm import llm

# 生成文档摘要的提示词
SUMMARY_PROMPT = """请对以下文档内容生成结构化摘要，包含：
1. 文档主题
2. 目录结构（列出所有章节/段落标题）
3. 各章节核心内容概要（每章1-2句话）
4. 文档总体信息（页数、关键词等）

请用中文回答，保持简洁。

文档内容：
{content}"""


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
        4. 生成文档摘要，作为 type=summary 的 Document 存入向量库
        5. 判断是否需要分割：任一文档块超过chunk_size则触发分割
        6. 将分块文档写入Qdrant向量库（type=chunk）
        7. 记录MD5到文件，标记为已上传
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

        # 过滤掉空内容的文档块（PDF某些页可能提取不到文字）
        documents = [doc for doc in documents if doc.page_content.strip()]

        if not documents:
            return f"文件 '{file_name}' 提取不到有效文本内容，跳过上传"

        # 4. 生成文档摘要并存入向量库（用于回答全局性问题）
        self._generate_and_store_summary(documents, file_name)

        # 5. 判断是否需要分割：遍历所有文档块，任一超过chunk_size就分割
        chunk_size = self.qdrant_store.text_splitter._chunk_size
        need_split = any(len(doc.page_content) > chunk_size for doc in documents)

        if need_split:
            documents = self.qdrant_store.text_splitter.split_documents(documents)

        # 再次过滤分割后可能产生的空块
        documents = [doc for doc in documents if doc.page_content.strip()]

        # 6. 为每个分块添加 metadata 标记类型和来源
        for doc in documents:
            doc.metadata["type"] = "chunk"
            doc.metadata["source"] = file_name

        # 写入向量库，每个文档块生成唯一ID
        self.qdrant_store.vector_store.add_documents(
            documents=documents,
            ids=[str(uuid.uuid4()) for _ in range(len(documents))]
        )

        # 7. 上传成功后记录MD5，防止下次重复上传
        self.write_md5(md5_value)

        return f"文件 '{file_name}' 上传成功，共 {len(documents)} 个文档块 + 1 条摘要"

    def _generate_and_store_summary(self, documents: list[Document], file_name: str):
        """
        生成文档摘要并存入向量库

        工作流程：
        - 短文档（<=8000字）：直接全文生成摘要
        - 长文档（>8000字）：分段生成局部摘要，再合并为完整摘要
        这样不管文档多长，都能覆盖全文内容

        :param documents: 原始文档块列表
        :param file_name: 文件名，记录到 metadata 中
        """
        # 拼接全文
        full_text = "\n".join(doc.page_content for doc in documents)

        if len(full_text) <= 8000:
            # 短文档：直接生成摘要
            summary_content = llm.invoke(SUMMARY_PROMPT.format(content=full_text)).content
        else:
            # 长文档：分段摘要 -> 合并
            # 按8000字分段，确保每段都能被 LLM 完整处理
            chunks = [full_text[i:i + 8000] for i in range(0, len(full_text), 8000)]

            # 每段生成局部摘要
            partial_summaries = []
            for i, chunk in enumerate(chunks):
                partial = llm.invoke(
                    f"请对以下文档的第{i + 1}部分生成摘要，包含章节标题和核心内容：\n\n{chunk}"
                ).content
                partial_summaries.append(partial)

            # 合并所有局部摘要，生成最终完整摘要
            merged = "\n\n".join(partial_summaries)
            summary_content = llm.invoke(
                f"以下是一篇文档各部分的摘要，请合并为一份完整的结构化摘要，"
                f"包含：文档主题、完整目录结构、各章节概要。\n\n{merged}"
            ).content

        # 构建摘要 Document，metadata 标记为 summary 类型
        summary_doc = Document(
            page_content=summary_content,
            metadata={"type": "summary", "source": file_name}
        )

        # 存入向量库
        self.qdrant_store.vector_store.add_documents(
            documents=[summary_doc],
            ids=[str(uuid.uuid4())]
        )

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

