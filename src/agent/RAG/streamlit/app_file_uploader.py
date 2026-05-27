import io
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()


# 添加网页标题
st.title("知识库更新服务")

# 文件上传组件，仅接受单个TXT文件
uploader_file = st.file_uploader(
    label="请上传pdf文件",
    type=['pdf'],
    accept_multiple_files=False,
)

# if uploader_file is not None:
#     # 提取文件信息
#     file_name = uploader_file.name
#     file_type = uploader_file.type
#     file_size = uploader_file.size / 1024  # 转换为KB单位
#
#     # 显示文件信息
#     st.subheader(f"文件名: {file_name}")
#     st.write(f"格式: {file_type} | 大小: {file_size:.2f} KB")
#
#     # 读取文件内容（bytes -> 字符串）
#     text = uploader_file.getvalue().decode("utf-8")
#     st.write(text)

if uploader_file is not None:
    file_name = uploader_file.name
    file_size = uploader_file.size / 1024
    st.subheader(f"文件名: {file_name}")
    st.write(f"大小: {file_size:.2f} KB")

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploader_file.getvalue())
        tmp_path = tmp.name

    # 用 PyPDFLoader 加载
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # 显示每页内容
    for doc in documents:
        st.write(f"--- 第 {doc.metadata['page'] + 1} 页 ---")
        st.write(doc.page_content)

    # 清理临时文件
    os.unlink(tmp_path)



