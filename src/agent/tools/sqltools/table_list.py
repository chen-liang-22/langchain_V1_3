from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from agent.utils.db_utils import MySQLDatabaseManager
from agent.utils.log_utils import log




class TableListInput(BaseModel):
    placeholder: str = Field(default="", description="无需填写，直接调用即可")


class TableListTool(BaseTool):
    """LangChain 工具类：获取数据库中所有表的名称和描述信息"""

    name: str = "list_database_tables"
    description: str = (
        "获取数据库中所有表的名称和描述（注释）信息。"
        "当需要了解数据库中有哪些表、表的用途时调用此工具。"
        "返回格式为：表名: 表描述，每行一条记录。"
    )
    args_schema: Type[BaseModel] = TableListInput

    db_manager: MySQLDatabaseManager = Field(default_factory=MySQLDatabaseManager, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, placeholder: str = "") -> str:
        log.info("[工具调用] list_database_tables: 获取所有表名及描述")
        tables_info = self.db_manager.get_tables_with_comments()
        if not tables_info:
            return "数据库中未找到任何表。"
        lines = [
            f"{row['table_name']}: {row['table_comment'] or '（无描述）'}"
            for row in tables_info
        ]
        return "\n".join(lines)
