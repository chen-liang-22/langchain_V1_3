from agent.my_llm import llm as model
from pydantic import BaseModel, Field

# res = model.invoke("什么是大模型？")
# print(f"模型返回：{res}")
# print(f"模型的类型是：{type(res)}")
# print(f"模型的返回结果是：{res.content}")


# cconversation = [
#     {"role": "system", "content": "你是一个乐于助人的助手，负责将汉语翻译成英语。"},
#     {"role": "user", "content": "翻译：我热爱编程。"},
#     {"role": "assistant", "content": "I love programming."},  # 这句是法语，无需翻译
#     {"role": "user", "content": "翻译：我热爱开发应用程序。"}
# ]
#
# response = model.invoke(cconversation)
# print(response.content)

# from langchain.messages import HumanMessage, AIMessage, SystemMessage
#
# conversation = [
#     SystemMessage("你是一个乐于助人的助手，负责将汉语翻译成英语。"),
#     HumanMessage("翻译：我热爱编程。"),
#     AIMessage("I love programming."),
#     HumanMessage("翻译：我热爱开发应用程序。")
# ]
#
# response = model.invoke(conversation)
# print(response.content)

for chunk in model.stream("为什么鹦鹉的羽毛颜色鲜艳？"):
    # print(type(chunk))
    # print(chunk)
    print(chunk.text, end="|", flush=True)
#
#
# for chunk in model.stream("天空是什么颜色？"):
#     for block in chunk.content_blocks:
#         if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
#             print(f"Reasoning: {reasoning}")
#         elif block["type"] == "tool_call_chunk":
#             print(f"Tool call chunk: {block}")
#         elif block["type"] == "text":
#             print(block["text"])
#         else:
#             ...
# responses = model.batch([
#     "鹦鹉为何拥有色彩斑斓的羽毛？",
#     "飞机是如何飞行的？",
#     "什么是量子计算？"
# ])
# for response in responses:
#     print(response)
#     print(response.content)


# questions = [
#     "鹦鹉为何拥有色彩斑斓的羽毛？",
#     "飞机是如何飞行的？",
#     "什么是量子计算？"
# ]
# for index, res in model.batch_as_completed(questions):
#     print(f"问题：{questions[index]}")
#     print(res.content)

