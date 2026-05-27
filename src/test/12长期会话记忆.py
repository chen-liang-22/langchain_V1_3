import os
import json
from typing import Sequence, List

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import (
    message_to_dict,
    messages_from_dict,
    BaseMessage
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
load_dotenv()


class FileChatMessageHistory(BaseChatMessageHistory):
    """基于本地文件持久化的对话历史存储类"""

    def __init__(self, session_id: str, storage_path: str):
        self.session_id = session_id  # 会话ID
        self.storage_path = storage_path  # 存储目录
        # 拼接完整文件路径（每个会话一个独立JSON文件）
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保存储目录存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    @property
    def messages(self) -> List[BaseMessage]:
        """从文件读取对话历史并转为消息对象列表"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
            # 字典列表 -> BaseMessage对象列表
            return messages_from_dict(messages_data)
        except FileNotFoundError:
            # 文件不存在时返回空列表
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """添加新消息并同步写入文件"""
        # 读取已有消息，合并新消息
        all_messages = list(self.messages)
        all_messages.extend(messages)

        # BaseMessage对象列表 -> 字典列表
        new_messages = [message_to_dict(msg) for msg in all_messages]

        # 写入本地JSON文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """清空当前会话的对话历史"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)


# 1. 初始化模型
model = ChatTongyi(model="qwen3-max")

# 2. 定义带对话历史的聊天模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据会话历史回答用户问题。对话历史："),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答如下问题：{input}")
    ]
)

str_parser = StrOutputParser()


# 3. 辅助函数：打印生成的完整提示词
def print_prompt(full_prompt):
    print("=" * 20, full_prompt.to_string(), "=" * 20)
    return full_prompt


# 4. 构建基础LCEL链
base_chain = prompt | print_prompt | model | str_parser


# 5. 会话历史获取函数
def get_history(session_id):
    return FileChatMessageHistory(session_id, storage_path="./chat_history")


# 6. 用RunnableWithMessageHistory增强链，实现对话历史管理
conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 7. 主程序：多轮对话测试
if __name__ == '__main__':
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }

    res = conversation_chain.invoke({"input": "小明有2个猫"}, session_config)
    print("第1次执行：", res)

    res = conversation_chain.invoke({"input": "小刚有1只狗"}, session_config)
    print("第2次执行：", res)

    res = conversation_chain.invoke({"input": "总共有几个宠物"}, session_config)
    print("第3次执行：", res)