from ddgs import DDGS
from langchain_core.tools import tool


@tool('my_web_search', parse_docstring=True)
def web_search(query: str) -> str:
    """互联网搜索的工具，可以搜索所有公开的信息。

    Args:
        query: 需要进行互联网查询的信息。

    Returns:
        返回搜索的结果信息，该信息是一个文本字符串。
    """
    try:
        results = DDGS().text(query, max_results=5, backend="lite")
        if not results:
            return "未找到相关结果。"
        return "\n\n".join(
            f"标题: {r['title']}\n链接: {r['href']}\n摘要: {r['body']}"
            for r in results
        )
    except Exception as e:
        return f"搜索失败: {e}"
