from langchain.agents import create_agent

from agent.my_llm import llm
from agent.tools.eMailTools.e_mail_tool import send_email

#这个要和langgraph.json这个配置中的./src/my_agent.py:agent一致
email = create_agent(
    llm,
    tools=[send_email]
)




