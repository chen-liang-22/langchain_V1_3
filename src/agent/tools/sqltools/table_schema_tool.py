from typing import Type, Optional, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.utils.db_utils import MySQLDatabaseManager
from agent.utils.log_utils import log


class TableSchemaInput(BaseModel):
    table_names: Optional[List[str]] = Field(
        default=None,
        description="要查询的表名列表，不传则返回所有表的结构信息"
    )


class TableSchemaTool(BaseTool):
    """LangChain 工具类：获取数据库表的模式信息（字段、主键、外键、索引等）"""

    name: str = "get_table_schema"
    description: str = (
        "获取数据库表的模式信息，包含字段名、字段类型、主键、外键、索引及字段注释。"
        "可指定一个或多个表名，不指定则返回所有表的结构。"
        "当需要了解表的具体结构时调用此工具。"
    )
    args_schema: Type[BaseModel] = TableSchemaInput

    db_manager: MySQLDatabaseManager = Field(default_factory=MySQLDatabaseManager, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, table_names: Optional[List[str]] = None) -> str:
        log.info("[工具调用] get_table_schema: 查询表结构，表名={}", table_names or "全部")
        return self.db_manager.get_table_schema(table_names)
