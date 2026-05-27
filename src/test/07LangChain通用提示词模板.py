from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from agent.my_llm import llm_deepseek

# 定义示例的格式模板
example_template = PromptTemplate.from_template("单词:{word}, 反义词:{antonym}")

# 示例数据（Few-Shot 的“例子”）
example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]

# 创建 FewShotPromptTemplate 对象
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="给出给定词的反义词，有如下示例：",
    suffix="基于示例告诉我：{input_word} 的反义词是？",
    input_variables=['input_word']
)

# # 打印生成的提示词文本（可选，用于调试）
# prompt_text = few_shot_prompt.invoke(input={"input_word": "左"}).to_string()
# print(prompt_text)

# 组成 chain 并调用模型
chain = few_shot_prompt | llm_deepseek
res = chain.invoke({"input_word": "左"})
print(res.content)


from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate  # 原代码漏了这个导入，我补上了

"""
PromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
ChatPromptTemplate -> BaseChatPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
"""

# 1. 创建 PromptTemplate 对象
template = PromptTemplate.from_template("我的邻居是: {lastname}, 最喜欢: {hobby}")

# 2. 使用 format() 方法格式化模板（返回字符串）
res = template.format(lastname="张大明", hobby="钓鱼")
print(res, type(res))

# 3. 使用 invoke() 方法调用模板（返回 PromptValue 对象，兼容 LCEL 链式调用）
res2 = template.invoke({"lastname": "周杰伦", "hobby": "唱歌"})
print(res2, type(res2))


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

# 1. 构建带历史对话的聊天模板
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),  # 对话历史的占位符
        ("human", "请再来一首唐诗"),
    ]
)

# 2. 模拟对话历史数据
history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

# 3. 生成最终提示词文本
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
print("生成的完整提示词：")
print(prompt_text)

# 4. 初始化通义千问模型（需要配置DASHSCOPE_API_KEY环境变量）
model = ChatTongyi(model="qwen3-max")

# 5. 用模板+模型构建LCEL链并调用
chain = chat_prompt_template | model
response = chain.invoke({"history": history_data})
print("\n模型回复：")
print(response.content)

response = chain.stream({"history": history_data})
print("\n模型回复：")
for chunk in response:
    print(chunk.content, end="", flush=True)




