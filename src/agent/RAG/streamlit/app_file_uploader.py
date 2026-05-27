import sys
import os

import streamlit as st
from dotenv import load_dotenv

# 将项目根目录加入 sys.path，确保能导入 src 下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from src.agent.RAG.knowledge.knowledge_base import KnowledgeBase

load_dotenv()

# 添加网页标题
st.title("知识库更新服务")

# 文件上传组件，支持 PDF 和 TXT
uploader_file = st.file_uploader(
    label="请上传文件（支持 PDF、TXT）",
    type=['pdf', 'txt'],
    accept_multiple_files=False,
)

if uploader_file is not None:
    file_name = uploader_file.name
    file_size = uploader_file.size / 1024
    st.subheader(f"文件名: {file_name}")
    st.write(f"大小: {file_size:.2f} KB")

    # 点击按钮触发上传到向量库
    if st.button("上传到知识库"):
        with st.spinner("正在处理文件并上传到向量库..."):
            kb = KnowledgeBase()
            result = kb.upload_file(uploader_file.getvalue(), file_name)
        st.success(result)



    # # 保存到临时文件
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    #     tmp.write(uploader_file.getvalue())
    #     tmp_path = tmp.name
    #
    # # 用 PyPDFLoader 加载
    # loader = PyPDFLoader(tmp_path)
    # documents = loader.load()
    #
    # # 显示每页内容
    # for doc in documents:
    #     st.write(f"--- 第 {doc.metadata['page'] + 1} 页 ---")
    #     st.write(doc.page_content)
    #
    # # 清理临时文件
    # os.unlink(tmp_path)