from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda

from src.agent.my_llm import llm_precise, llm_tongyi
from src.agent.RAG.knowledge.qdrant_store import QdrantStore
from src.agent.RAG.prompt.rag_prompt import create_rag_prompt
from src.agent.RAG.chat_history.file_chat_history import FileChatMessageHistory

class RAGChain:
    """
    完整的 RAG（Retrieval-Augmented Generation）对话链

    核心流程：
    1. 接收用户问题
    2. 从 Qdrant 向量库中检索语义最相关的文档片段
    3. 将检索结果作为上下文，连同对话历史一起组装成提示词
    4. 发送给 LLM 生成回答
    5. 自动将本轮对话（用户问题 + AI回答）保存到历史记录

    依赖模块：
    - QdrantStore: 向量检索
    - create_rag_prompt: 提示词模板
    - FileChatMessageHistory: 对话历史持久化
    - llm_precise: 低温度LLM（temperature=0.1），适合事实性问答
    """

    def __init__(self, storage_path: str = None):
        """
        初始化 RAG 对话链
        :param storage_path: 对话历史 JSON 文件的存储目录
                            默认为 chat_history/sessions/ 目录
        """
        # 向量库实例，用于检索与用户问题语义相关的文档
        self.qdrant_store = QdrantStore()
        # 提示词模板（system + chat_history + human 三段式结构）
        self.prompt = create_rag_prompt()
        # 对话历史存储路径（传给 FileChatMessageHistory）
        self.storage_path = storage_path
        # 构建完整的 LCEL 链路
        self._chain = self._build_chain()

    def _retrieve_context(self, input_dict: dict) -> dict:
        """
        检索相关文档并拼接为上下文字符串

        工作流程：
        1. 从 input_dict 中提取用户问题
        2. 调用 QdrantStore.search 进行向量相似度检索，返回 top-k 文档
        3. 将多个文档的 page_content 用双换行拼接为一个完整的上下文字符串
        4. 将 context 字段追加到原字典中返回

        :param input_dict: 包含 "input" 字段的字典（用户问题）
        :return: 原字典 + "context" 字段（检索到的文档内容拼接字符串）
        """
        query = input_dict["input"]
        # 检索最相关的3个文档片段
        docs = self.qdrant_store.search(query, k=3)
        # 将多个文档内容用双换行分隔，形成完整的参考上下文
        context = "\n\n".join(doc.page_content for doc in docs)
        return {**input_dict, "context": context}

    def _build_chain(self):
        """
        构建完整的 LCEL（LangChain Expression Language）链

        链路结构：
        RunnableLambda(_retrieve_context)  # 第1步：向量检索，填充 context
            |
        ChatPromptTemplate                 # 第2步：组装提示词（system+history+human）
            |
        llm_precise                        # 第3步：调用 LLM 生成回答
            |
        StrOutputParser                    # 第4步：提取纯文本回答

        外层包装 RunnableWithMessageHistory：
        - 在调用前：自动从 JSON 文件加载历史消息，注入到 chat_history 占位符
        - 在调用后：自动将本轮的 HumanMessage 和 AIMessage 追加保存到文件
        """
        # 基础链：检索 -> 提示词 -> LLM -> 输出解析
        base_chain = (
            RunnableLambda(self._retrieve_context)
            | self.prompt
            | llm_tongyi
            | StrOutputParser()
        )

        # 包装为带对话历史的链
        # input_messages_key="input": 告诉框架用户输入在 dict 的哪个 key
        # history_messages_key="chat_history": 告诉框架历史消息注入到模板的哪个变量
        return RunnableWithMessageHistory(
            base_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def _get_history(self, session_id: str) -> FileChatMessageHistory:
        """
        获取指定 session 的对话历史实例

        此方法作为工厂函数传给 RunnableWithMessageHistory，
        框架会在每次 invoke 时调用它来获取对应 session 的历史存储对象

        :param session_id: 会话ID（从 config["configurable"]["session_id"] 传入）
        :return: FileChatMessageHistory 实例
        """
        return FileChatMessageHistory(session_id, storage_path=self.storage_path)

    def chat(self, query: str, session_id: str = "default") -> str:
        """
        发起一次 RAG 对话（对外核心接口）

        使用示例：
            rag = RAGChain()
            answer = rag.chat("北森系统什么时候启用？", session_id="user_001")

        :param query: 用户问题（自然语言）
        :param session_id: 会话ID，用于隔离不同用户/对话的历史记录
                          同一个 session_id 的多次调用会共享对话历史
        :return: AI 回答文本（纯字符串）
        """
        # configurable.session_id 是 RunnableWithMessageHistory 约定的配置项
        # 框架会将此值传给 _get_history 方法来定位对应的历史文件
        config = {"configurable": {"session_id": session_id}}
        return self._chain.invoke({"input": query}, config)


if __name__ == "__main__":
    rag = RAGChain()

    # 第一轮对话：直接提问
    answer1 = rag.chat("北森系统什么时候启用？", session_id="test_session")
    print("回答1:", answer1)

    # 第二轮对话：追问（框架会自动带上第一轮的对话历史）
    answer2 = rag.chat("具体有哪些功能？", session_id="test_session")
    print("回答2:", answer2)
