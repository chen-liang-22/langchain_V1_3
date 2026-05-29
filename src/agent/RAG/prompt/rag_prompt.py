from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_rag_prompt() -> ChatPromptTemplate:
    """
    创建 RAG 对话提示模板

    模板结构（按消息顺序）：
    1. system 消息:
       - 定义 AI 的角色（知识库助手）
       - 注入检索到的文档内容 {context}，作为回答的参考依据
       - 明确指令：没有相关信息时如实告知，避免编造
    2. chat_history 占位符:
       - 由 RunnableWithMessageHistory 在运行时自动填充历史消息
       - 包含之前的 HumanMessage 和 AIMessage，实现多轮对话
    3. human 消息:
       - {input} 用户当前的问题

    变量说明：
    - {context}: 从向量库检索到的相关文档内容，由 RAGChain._retrieve_context 填充
    - {chat_history}: 对话历史消息列表，由 RunnableWithMessageHistory 自动管理
    - {input}: 用户当前输入的问题文本

    :return: ChatPromptTemplate 实例，可直接用于 LCEL 链
    """
    return ChatPromptTemplate.from_messages([
        # system 消息：设定角色和行为规范，注入检索到的上下文
        ("system",
         "你是一个知识库助手，请根据以下检索到的相关文档内容回答用户问题。"
         "如果文档中没有相关信息，请如实告知用户。\n\n"
         "相关文档内容：\n{context}"),
        # 对话历史占位符：RunnableWithMessageHistory 会将历史消息注入此位置
        # 位于 system 和 human 之间，让 LLM 能感知上下文对话
        MessagesPlaceholder("chat_history"),
        # 用户当前问题
        ("human", "{input}")
    ])
