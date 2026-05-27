import hashlib
import os


class KnowledgeBase:
    """知识库MD5校验，防止重复导入"""

    def __init__(self, md5_file_path: str = None):
        if md5_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            md5_file_path = os.path.join(current_dir, "md5_records.text")
        self.md5_file_path = md5_file_path
        # 确保文件存在
        if not os.path.exists(self.md5_file_path):
            with open(self.md5_file_path, "w", encoding="utf-8") as f:
                pass

    def to_md5(self, content: str) -> str:
        """将传入的内容转成MD5"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def check_md5(self, md5_value: str) -> bool:
        """校验MD5是否存在，不存在返回True，存在返回False"""
        with open(self.md5_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_value:
                    return False
        return True

    def write_md5(self, md5_value: str):
        """将不存在的MD5写入文件"""
        if self.check_md5(md5_value):
            with open(self.md5_file_path, "a", encoding="utf-8") as f:
                f.write(md5_value + "\n")




if __name__ == "__main__":
    kb = KnowledgeBase()

    # 测试 to_md5
    content = "这是一段测试内容"
    md5_value = kb.to_md5(content)
    print(f"内容: {content}")
    print(f"MD5: {md5_value}")

    # 测试 check_md5 - 第一次应该返回True（不存在）
    result = kb.check_md5(md5_value)
    print(f"第一次校验（应为True）: {result}")

    # 测试 write_md5 - 写入
    kb.write_md5(md5_value)
    print("已写入MD5")

    # 测试 check_md5 - 第二次应该返回False（已存在）
    result = kb.check_md5(md5_value)
    print(f"第二次校验（应为False）: {result}")

    # 再次写入相同内容，不应重复写入
    kb.write_md5(md5_value)
    print("尝试重复写入（不会实际写入）")

