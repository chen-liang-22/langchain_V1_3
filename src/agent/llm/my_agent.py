from langchain.agents import create_agent


from agent.my_llm import llm
from agent.tools.eMailTools.e_mail_tool import send_email
from agent.tools.eMailTools.e_mail_tool_class import MyWebSearchTool
from agent.tools.webtool.tool_web import web_search

email = MyWebSearchTool()
#这个要和langgraph.json这个配置中的./src/my_agent.py:agent一致
agent = create_agent(
    llm,
    tools=[email, web_search]
)




