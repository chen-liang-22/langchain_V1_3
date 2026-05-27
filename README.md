# LangChain V1.3 - AI Agent 项目

基于 LangChain + LangGraph 构建的多工具 AI Agent 系统，支持知识库问答（RAG）、SQL 查询、邮件发送、Web 搜索、代码分析等功能。

## 环境要求

- Python >= 3.12
- MySQL 数据库
- Qdrant 向量数据库（Docker 部署）

## 安装依赖

```bash
pip install langgraph langchain-core langchain-community langchain-openai langchain-deepseek langchain-anthropic langchain-ollama langchain-qdrant langchain-text-splitters python-dotenv sqlalchemy pymysql pydantic qdrant-client streamlit loguru ddgs pypdf dashscope
```

## 中间件部署

### Qdrant（向量数据库）

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### MySQL

项目连接远程 MySQL，需确保数据库可访问。

## 环境变量配置

在项目根目录创建  文件：

```env
# OpenAI
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_base_url

# Anthropic Claude
CLAUDE_API_KEY=your_key

# Deepseek
DEEPSEEK_API_KEY=your_key

# 阿里 DashScope
DASHSCOPE_API_KEY=your_key

# SMTP 邮件
SMTP_HOST=smtp.qq.com
SMTP_PORT=587
SMTP_USER=your_email@qq.com
SMTP_PASS=your_auth_code
SMTP_USE_TLS=true

# MySQL
DB_HOST=your_host
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=your_db

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=my_rag_collection

# 文本分割配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNK_SEPARATORS=

,
,。,！,？,；,.,!,?,;, ,

# LangSmith（可选，链路追踪）
LANGSMITH_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
```

## 项目结构

```
src/
├── agent/
│   ├── my_llm.py           # LLM 模型配置
│   ├── RAG/                # 知识库问答模块
│   │   ├── knowledge/      # 知识库管理（MD5校验、向量存储）
│   │   └── streamlit/      # Web 上传界面
│   ├── tools/              # Agent 工具集
│   │   ├── sqlTool/        # SQL 查询工具
│   │   ├── eMailTools/     # 邮件发送工具
│   │   ├── webtool/        # Web 搜索工具
│   │   └── codingTool/     # 代码分析工具
│   └── utils/              # 工具类（数据库、日志）
└── test/                   # 测试用例
```

## 运行

```bash
# 运行主 Agent
python src/my_agent01.py

# 运行知识库上传页面
streamlit run src/agent/RAG/streamlit/app_file_uploader.py
```
