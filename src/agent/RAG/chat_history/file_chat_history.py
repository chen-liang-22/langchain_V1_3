import json
import os
from typing import Sequence, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    基于本地 JSON 文件的对话历史持久化存储

    设计思路：
    - 每个 session_id 对应一个独立的 JSON 文件，实现会话隔离
    - 文件格式为 JSON 数组，每个元素是一条序列化后的消息
    - 继承 BaseChatMessageHistory，兼容 LangChain 的 RunnableWithMessageHistory

    存储结构示例（sessions/user_001.json）：
    [
      {"type": "human", "data": {"content": "你好", ...}},
      {"type": "ai", "data": {"content": "你好！有什么可以帮你的？", ...}}
    ]
    """

    def __init__(self, session_id: str, storage_path: str = None):
        """
        初始化对话历史存储
        :param session_id: 会话ID，用于区分不同对话（一个ID对应一个JSON文件）
        :param storage_path: 存储目录路径，默认为当前模块下的 sessions/ 目录
        """
        self.session_id = session_id
        # 默认存储路径：当前文件所在目录/sessions/
        if storage_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            storage_path = os.path.join(current_dir, "sessions")
        self.storage_path = storage_path
        # 每个 session 对应一个 JSON 文件，文件名即 session_id
        self.file_path = os.path.join(self.storage_path, f"{self.session_id}.json")
        # 确保存储目录存在（首次使用时自动创建）
        os.makedirs(self.storage_path, exist_ok=True)

    @property
    def messages(self) -> List[BaseMessage]:
        """
        读取当前会话的所有历史消息

        工作流程：
        1. 从 JSON 文件读取序列化的消息数据
        2. 通过 messages_from_dict 反序列化为 BaseMessage 对象列表
        3. 文件不存在时返回空列表（首次对话）

        :return: BaseMessage 列表（包含 HumanMessage 和 AIMessage）
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
            # messages_from_dict: LangChain 提供的反序列化方法
            # 将 dict 列表还原为 HumanMessage/AIMessage 等对象
            return messages_from_dict(messages_data)
        except FileNotFoundError:
            # 文件不存在说明是新会话，返回空历史
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        追加消息到对话历史（自动持久化到文件）

        工作流程：
        1. 先读取已有的全部历史消息
        2. 将新消息追加到末尾
        3. 全量序列化后写回文件（覆盖写入）

        注意：RunnableWithMessageHistory 会在每次对话后自动调用此方法，
        传入本轮的 HumanMessage 和 AIMessage，无需手动调用

        :param messages: 要追加的消息列表（通常是本轮的用户输入和AI回复）
        """
        # 读取已有历史
        all_messages = list(self.messages)
        # 追加新消息
        all_messages.extend(messages)
        # message_to_dict: LangChain 提供的序列化方法，将 BaseMessage 转为可JSON化的dict
        serialized = [message_to_dict(msg) for msg in all_messages]
        # 全量写回文件（ensure_ascii=False 保证中文正常存储）
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """
        清空当前会话的所有历史消息
        将文件内容重置为空数组，文件本身保留
        """
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)
