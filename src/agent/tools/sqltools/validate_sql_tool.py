from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.utils.db_utils import MySQLDatabaseManager
from agent.utils.log_utils import log


class ValidateSqlInput(BaseModel):
    query: str = Field(description="需要验证语法是否正确的 SQL 查询语句")


class ValidateSqlTool(BaseTool):
    """LangChain 工具类：验证 SQL 查询语法是否正确"""

    name: str = "validate_sql"
    description: str = (
        "验证 SQL 查询语句的语法是否正确，不会实际执行查询。"
        "在执行 SQL 前可先调用此工具进行语法检查，避免执行错误的语句。"
    )
    args_schema: Type[BaseModel] = ValidateSqlInput

    db_manager: MySQLDatabaseManager = Field(default_factory=MySQLDatabaseManager, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        log.info("[工具调用] validate_sql: 验证 SQL 语法，SQL={}", query)
        return self.db_manager.validate_query(query)
