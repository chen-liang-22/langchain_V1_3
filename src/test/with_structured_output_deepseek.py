# 1. 导入所有需要的包
import json
import logging
from typing import List
from pydantic import BaseModel, Field

from agent.my_llm import llm, llm_deepseek

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 4. 定义结构化输出格式（Pydantic）
class Actor(BaseModel):
    real_name: str = Field(description="演员的真实姓名")
    role_name: str = Field(description="演员在电影中扮演的角色姓名")
    gender: str = Field(description="演员性别，男或女")
    age_at_filming: int = Field(description="演员拍摄该电影时的年龄")


class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="电影上映年份")
    director: str = Field(description="电影导演")
    rating: float = Field(description="电影评分")
    actors: List[Actor] = Field(description="主要演员列表，最多返回10个")
    

# 5. 绑定结构化输出
model_with_structured_output = llm_deepseek.with_structured_output(Movie, method="json_mode")

# 6. 调用模型
logger.info("开始调用模型...\n%s", json.dumps(Movie.model_json_schema(), ensure_ascii=False, indent=2))
resp = model_with_structured_output.invoke([
    ("system", f"请严格按照以下 JSON Schema 返回结果：\n{Movie.model_json_schema()}"),
    ("human", "介绍一下电影《建国大业》,不要给我瞎编")
])
logger.info("模型调用完成")

# 7. 输出结果
logger.info("返回类型：%s", type(resp))
logger.info("电影名：%s | 年份：%s | 导演：%s | 评分：%s", resp.title, resp.year, resp.director, resp.rating)
logger.info("演员数量：%d", len(resp.actors))
print("返回类型：", type(resp))
print("完整对象：", resp)
print("\n单独取出每个字段：")
print("电影名：", resp.title)
print("年份：", resp.year)
print("导演：", resp.director)
print("评分：", resp.rating)
print("\n演员列表：")
for actor in resp.actors:
    logger.info("演员：%s | 角色：%s", actor.real_name, actor.role_name)
    print(f"  真实姓名: {actor.real_name} | 角色名: {actor.role_name} | 性别: {actor.gender} | 拍摄时年龄: {actor.age_at_filming}")


