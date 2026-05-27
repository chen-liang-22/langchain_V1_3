from pydantic import BaseModel, Field

from agent.my_llm import llm


class Movie(BaseModel):
    title: str       = Field(description="标题")
    year: int        = Field(description="年份")
    director: str    = Field(description="导演")
    rating: float    = Field(description="评分")
    description: str = Field(description="描述")
rs = llm.with_structured_output(Movie, method="json_mode").invoke(
    "提供动画片 -喜羊羊与灰太狼 的详细信息，以JSON格式返回，包含字段：title、year、director、rating、description"
)
print(rs)
