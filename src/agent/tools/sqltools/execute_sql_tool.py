from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.utils.db_utils import MySQLDatabaseManager
from agent.utils.log_utils import log


class ExecuteSqlInput(BaseModel):
    query: str = Field(description="要执行的 SQL 查询语句，只允许 SELECT / WITH 查询")


class ExecuteSqlTool(BaseTool):
    """LangChain 工具类：执行 SQL 查询并返回结果"""

    name: str = "execute_sql"
    description: str = (
        "执行 SQL 查询语句并返回结果，结果以 Markdown 表格格式呈现。"
        "只允许 SELECT 和 WITH 查询，禁止任何数据修改操作。"
        "最多返回 100 行数据。"
    )
    args_schema: Type[BaseModel] = ExecuteSqlInput

    db_manager: MySQLDatabaseManager = Field(default_factory=MySQLDatabaseManager, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        log.info("[工具调用] execute_sql: 执行查询，SQL={}", query)
        return self.db_manager.execute_query(query)
