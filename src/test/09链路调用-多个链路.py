from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

# 初始化解析器
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

# 初始化通义千问模型
model = ChatTongyi(model="qwen3-max")

# 第一个提示词：生成名字并返回JSON格式
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，并封装到JSON格式返回给我，"
    "要求key是name，value就是起的名字。请严格遵守格式要求"
)

# 第二个提示词：解析名字含义
second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)

# LCEL链式调用
chain = first_prompt | model | json_parser | second_prompt | model | str_parser

# 执行链
res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)
print(type(res))










