from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(model="deepseek-chat")
messages = [SystemMessage(content="你是一个专业的写诗助手，古诗风格"),
            HumanMessage(content="帮我写一个加班的诗"),
            AIMessage(content="《夜班偶题》更深灯影倦，铁案自相磨。冷月悬天幕，残星坠砚河。茶烟浮世味，键盘叩心歌。忽觉东方白，方惊晓色多。"),
            HumanMessage(content="根据上面的风格秀一个更好的")]


res = model.stream(messages)
for chunk in res:
    print(chunk.content, end="", flush=True)










