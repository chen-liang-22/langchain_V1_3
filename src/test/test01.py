from agent.my_llm import llm

# messsge = llm.invoke('请先展示你的思考过程，然后再给出最终答案。\n\n问题：用三句话介绍什么是劳动法？')
#
# print(messsge.content)
#
# print(type(messsge))
#
# print(messsge)


for chunk in llm.stream('请先展示你的思考过程，然后再给出最终答案。\n\n问题：用三句话介绍什么是治安法？'):

    print(chunk)

    print(type(chunk))







