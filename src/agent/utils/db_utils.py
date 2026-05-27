import os
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from agent.utils.log_utils import log

load_dotenv()


class MySQLDatabaseManager:
    """MySQL数据库管理器，负责数据库连接和基本操作"""

    def __init__(self, connection_string: Optional[str] = None):
        """
        初始化MySQL数据库连接

        Args:
            connection_string: MySQL连接字符串，格式为:
                mysql+pymysql://username:password@host:port/database
                不传时自动从环境变量读取（DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME）
        """
        if connection_string is None:
            connection_string = (
                "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT"),
                    database=os.getenv("DB_NAME"),
                )
            )
        self.engine = create_engine(
            connection_string,
            pool_size=5,          # 连接池大小
            pool_recycle=3600,    # 连接回收时间（秒），防止连接超时
            echo=False            # 是否打印SQL语句，调试时可设为True
        )

    def get_table_names(self) -> list[str]:
        """获取数据库中所有表名"""
        try:
            # 创建一个Inspector（数据库映射）对象
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:

            log.exception(e)
            raise ValueError(f"获取表名失败: {str(e)}")

    def get_tables_with_comments(self) -> List[dict]:
        """
        获取数据库中所有表的名称和描述信息。

        Returns:
            List[dict]: 一个字典列表，每个字典包含 'table_name' 和 'table_comment' 键。
        """
        try:
            # 构建查询语句，从 INFORMATION_SCHEMA.TABLES 中获取表名和注释
            query = text("""
                SELECT TABLE_NAME, TABLE_COMMENT
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)

            with self.engine.connect() as connection:
                result = connection.execute(query)
                # 将结果转换为字典列表，便于后续处理
                tables_info = [{'table_name': row[0], 'table_comment': row[1]} for row in result]
                return tables_info
        except SQLAlchemyError as e:
            log.exception(e)
            raise ValueError(f"获取表名及描述信息失败: {str(e)}")

    def get_table_schema(self, table_names: Optional[List[str]] = None) -> str:
        """
        获取指定表的模式信息（包含字段注释）

        Args:
            table_names: 表名列表，如果为None则获取所有表
        """
        try:
            inspector = inspect(self.engine)
            schema_info = []

            tables_to_process = table_names if table_names else self.get_table_names()

            for table_name in tables_to_process:
                # 获取表结构信息
                columns = inspector.get_columns(table_name)
                # 使用 get_pk_constraint 替代已弃用的 get_primary_keys
                pk_constraint = inspector.get_pk_constraint(table_name)
                primary_keys = pk_constraint['constrained_columns'] if pk_constraint else []
                foreign_keys = inspector.get_foreign_keys(table_name)
                indexes = inspector.get_indexes(table_name)

                # 构建表模式描述
                table_schema = f"表名: {table_name}\n"
                table_schema += "列信息:\n"

                for column in columns:
                    # 检查该列是否在主键列表中
                    pk_indicator = " (主键)" if column['name'] in primary_keys else ""
                    # 获取字段注释，如果不存在则显示“无注释”
                    comment = column.get('comment', '无注释')
                    table_schema += f"  - {column['name']}: {str(column['type'])}{pk_indicator} [注释: {comment}]\n"

                if foreign_keys:
                    table_schema += "外键约束:\n"
                    for fk in foreign_keys:
                        table_schema += f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"

                if indexes:
                    table_schema += "索引信息:\n"
                    for idx in indexes:
                        if not idx['name'].startswith('sqlite_'):
                            unique_flag = "唯一" if idx['unique'] else "普通"
                            table_schema += f"  - {idx['name']}: {idx['column_names']} ({unique_flag}索引)\n"

                schema_info.append(table_schema)

            return "\n".join(schema_info) if schema_info else "未找到匹配的表"

        except SQLAlchemyError as e:
            log.exception(e)
            raise ValueError(f"获取表结构信息失败: {str(e)}")

    def execute_query(self, query: str) -> str:
        """
        执行SQL查询并返回结果

        Args:
            query: SQL查询语句
        """
        # 安全检查: 防止数据修改操作
        forbidden_keywords = ['insert', 'update', 'delete', 'drop', 'alter', 'create', 'grant', 'truncate']
        query_lower = query.lower().strip()

        # 检查是否以SELECT开头（允许子查询等复杂情况）
        if not query_lower.startswith(('select', 'with')) and any(
            keyword in query_lower for keyword in forbidden_keywords):
            raise ValueError("出于安全考虑，只允许执行SELECT查询和WITH查询")

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))

                # 获取列名
                columns = result.keys()

                # 获取数据（限制返回行数防止内存溢出）
                rows = result.fetchmany(100)

                if not rows:
                    return "查询结果为空"

                # 格式化结果（处理无法序列化的数据类型，同时生成Markdown表格）
                result_data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        try:
                            # 处理datetime、bytes等无法直接序列化的类型
                            value = row[i]
                            if hasattr(value, 'isoformat'):
                                value = value.isoformat()
                            elif isinstance(value, bytes):
                                value = value.decode('utf-8', errors='replace')
                            row_dict[col] = value
                        except Exception as e:
                            row_dict[col] = f"无法处理的数据: {str(e)}"
                    result_data.append(row_dict)

                # 生成Markdown表格
                header = "| " + " | ".join(columns) + " |\n"
                separator = "| " + " | ".join(["---"] * len(columns)) + " |\n"
                body = ""
                for row in result_data:
                    body += "| " + " | ".join(str(row[col]) for col in columns) + " |\n"

                return header + separator + body

        except SQLAlchemyError as e:
            log.exception(e)
            raise ValueError(f"执行SQL查询失败: {str(e)}")

    def validate_query(self, query: str) -> str:
        """
        验证SQL查询语法是否正确

        Args:
            query: 要验证的SQL查询
        """
        # 基本语法检查
        if not query or not query.strip():
            return "错误: 查询不能为空"

        # 检查是否以SELECT或WITH开头
        query_lower = query.lower().strip()
        if not query_lower.startswith(('select', 'with')):
            return "警告: 建议使用SELECT或WITH查询，其他操作可能被限制"

        # 尝试解析查询（不实际执行）
        try:
            with self.engine.connect() as connection:
                # 使用SQLAlchemy的text()来解析但不执行
                parsed_query = text(query)
                # 尝试编译查询来检查语法
                parsed_query.compile(compile_kwargs={"literal_binds": True})
                return "SQL查询语法看起来正确"
        except Exception as e:
            log.exception(e)
            return f"SQL语法错误: {str(e)}"
if __name__ == '__main__':
    # 配置数据库连接信息
    DB_CONFIG = {
        "host": "39.107.111.31",
        "port": 7001,
        "username": "root",
        "password": "oSidkh.7jq",
        "database": "lms"
    }

    # 拼接SQLAlchemy连接字符串
    conn_str = (
        f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

    # 初始化数据库管理器
    manager = MySQLDatabaseManager(conn_str)

    # 1.测试获取所有表名
    tables = manager.get_table_names()
    log.info(f"数据库中的表名: {tables}")

    # 2. 测试获取表名+注释
    tables_info = manager.get_tables_with_comments()
    log.info(f"数据库表信息: {tables_info}")

    # 3. 测试获取表结构（这里获取所有表）
    schema = manager.get_table_schema()
    log.info("数据库表结构:\n" + schema)

    # 4. 测试SQL语法验证
    test_sql = "SELECT * FROM ai_simulation_training LIMIT 5"
    validate_result = manager.validate_query(test_sql)
    log.info(f"SQL验证结果: {validate_result}")

    # 5. 测试执行查询
    query_result = manager.execute_query(test_sql)
    log.info("查询结果:\n" + query_result)



