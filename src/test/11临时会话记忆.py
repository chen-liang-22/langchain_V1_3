from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()
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
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

# 4. 构建基础LCEL链
base_chain = prompt | print_prompt | model | str_parser

# 5. 会话历史存储与获取函数
store = {}  # key: session_id, value: InMemoryChatMessageHistory对象

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 6. 用RunnableWithMessageHistory增强链，实现对话历史管理
conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",        # 用户输入变量名
    history_messages_key="chat_history" # 对话历史占位符变量名
)

# 7. 主程序：多轮对话测试
if __name__ == '__main__':
    # 会话配置：指定session_id
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }

    # 第1次对话
    res = conversation_chain.invoke({"input": "小明有2个猫"}, session_config)
    print("第1次执行：", res)

    # 第2次对话
    res = conversation_chain.invoke({"input": "小刚有1只狗"}, session_config)
    print("第2次执行：", res)

    # 第3次对话：依赖历史上下文提问
    res = conversation_chain.invoke({"input": "总共有几个宠物"}, session_config)
    print("第3次执行：", res)





















































































































































