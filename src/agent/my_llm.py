import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")
_base_url = os.getenv("OPENAI_BASE_URL")
_model = "openai/gpt-oss-120b"

# 通用对话模型：temperature=0.7，适合 agent、email 等需要一定创造性的场景
llm = ChatOpenAI(
    openai_api_key=_api_key,
    base_url=_base_url,
    model=_model,
    temperature=0.7
)

# 精确模型：temperature=0.1，适合 SQL 生成、代码生成等需要严谨输出的场景
llm_precise = ChatOpenAI(
    openai_api_key=_api_key,
    base_url=_base_url,
    model=_model,
    temperature=0.1
)

llm_deepseek = ChatDeepSeek(
    model_name='deepseek-v4-flash',
    temperature=0.7
)

llm_tongyi = ChatTongyi(model="qwen3-max")


