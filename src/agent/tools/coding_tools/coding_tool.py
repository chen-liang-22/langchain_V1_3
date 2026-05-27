import os
from pathlib import Path
from typing import Type, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.my_llm import llm_precise as llm
from agent.utils.log_utils import log

_CODE_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".h",
    ".cs", ".kt", ".swift", ".rb", ".php", ".html", ".css", ".sql",
    ".yaml", ".yml", ".json", ".toml", ".xml", ".sh", ".md"
}

_LOCATE_PROMPT = """你是一个代码分析专家。
用户会给你一个项目的所有文件路径列表，以及一个需求说明。
请从路径列表中找出与需求相关的文件，只返回完整的相对路径，每行一个，不要输出任何其他内容。

命名映射规则（重要）：
- 数据库表名是下划线命名（如 sys_oper_log），对应 Java 实体类是大驼峰命名（如 SysOperLog.java）
- 实体类通常位于 domain、entity、model、pojo 目录
- Mapper 接口位于 mapper 目录，Service 位于 service 目录，Controller 位于 controller 目录
- 返回的路径必须与列表中的路径完全一致，不要修改或缩短路径
"""

_DESIGN_PROMPT = """你是一个资深软件架构师。
用户会提供若干代码文件内容和一个需求说明。
请根据需求给出详细的技术设计方案，包括：

1. 需求分析：理解需求的核心目标
2. 涉及模块：列出需要改动的模块和文件
3. 设计方案：详细描述每个模块的改动思路、接口设计、数据流转
4. 注意事项：潜在风险、边界情况、依赖关系

只输出设计文档，不要输出任何代码。
"""

_ASK_PROMPT = """你是一个资深软件架构师兼工程师。
用户会提供若干代码文件内容和一个需求说明。
请先给出详细的技术设计方案（同设计模式），然后在末尾询问用户是否确认按此方案修改代码。

输出格式：
1. 需求分析
2. 涉及模块
3. 设计方案
4. 注意事项

---
以上是本次的改动设计方案，是否确认按此方案修改代码？（回复"确认"后将执行代码修改）
"""

_DEVELOP_PROMPT = """你是一个资深软件工程师。
用户会提供若干代码文件内容和一个修改需求。
请直接根据需求修改代码，按以下格式输出需要改动的文件（严格遵守格式）：

===FILE: 相对文件路径===
完整的文件内容（包含修改后的代码）
===END===

- 只输出需要改动的文件，未改动的文件不要输出
- 每个文件输出完整内容，不要省略未修改的部分
- 文件内容块之外不要输出任何解释性文字
"""


class CodeToolInput(BaseModel):
    requirement: str = Field(description="对代码的具体需求说明")
    mode: str = Field(
        default="ask",
        description=(
            "操作模式：\n"
            "design=设计模式，只输出技术设计方案，不改代码；\n"
            "ask=询问模式，先输出设计方案，询问用户是否确认后再改代码；\n"
            "dev=开发模式，直接根据需求修改代码并写回文件，无需确认。"
        )
    )
    confirmed: Optional[bool] = Field(
        default=False,
        description="ask 模式下用户确认后设为 True，工具将执行代码写入"
    )
    last_design: Optional[str] = Field(
        default=None,
        description="ask 模式确认写入时，将上一次返回的设计内容传入，工具据此生成并写入代码"
    )


class CodingTool(BaseTool):
    """代码工具：自动扫描项目文件，支持设计/询问/开发三种模式"""

    name: str = "coding_tool"
    description: str = (
        "读取项目代码，根据需求执行设计或修改操作。项目路径已预先配置，无需用户提供。\n"
        "mode=design：只输出技术设计方案，不修改任何文件。\n"
        "mode=ask：先输出设计方案并询问用户是否确认，confirmed=True 时再写入代码。\n"
        "mode=dev：直接根据需求修改代码并写回文件，无需任何确认。"
    )
    args_schema: Type[BaseModel] = CodeToolInput

    project_path: str = Field(default_factory=lambda: os.getenv("CODING_PROJECT_PATH", ""))

    class Config:
        arbitrary_types_allowed = True

    def _get_all_paths(self) -> list[str]:
        root = Path(self.project_path)
        paths = []
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix in _CODE_EXTENSIONS:
                paths.append(str(file_path.relative_to(root)).replace("\\", "/"))
        return paths

    def _locate_relevant_files(self, all_paths: list[str], requirement: str) -> list[str]:
        path_list = "\n".join(all_paths)
        messages = [
            {"role": "system", "content": _LOCATE_PROMPT},
            {"role": "user", "content": f"需求：{requirement}\n\n文件路径列表：\n{path_list}"},
        ]
        response = llm.invoke(messages)
        llm_lines = [
            line.strip()
            for line in response.content.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        log.info("[coding_tool] LLM 返回候选路径: {}", llm_lines)

        path_set = set(all_paths)
        relevant = []
        for ret in llm_lines:
            ret_norm = ret.replace("\\", "/")
            if ret_norm in path_set:
                if ret_norm not in relevant:
                    relevant.append(ret_norm)
                continue
            for actual in all_paths:
                if actual.endswith(ret_norm) or ret_norm.endswith(actual):
                    if actual not in relevant:
                        relevant.append(actual)

        log.info("[coding_tool] 匹配到 {} 个相关文件: {}", len(relevant), relevant)
        return relevant

    def _read_files(self, relative_paths: list[str]) -> dict[str, str]:
        root = Path(self.project_path)
        files = {}
        for rel in relative_paths:
            file_path = root / rel
            if file_path.exists():
                try:
                    files[rel] = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    log.warning("[coding_tool] 读取文件失败: {} - {}", rel, e)
            else:
                log.warning("[coding_tool] 文件不存在: {}", rel)
        return files

    def _build_code_message(self, files: dict[str, str], requirement: str) -> str:
        parts = [f"需求：{requirement}\n\n代码文件："]
        for rel, content in files.items():
            parts.append(f"\n===FILE: {rel}===\n{content}\n===END===")
        return "\n".join(parts)

    def _parse_and_write(self, response: str) -> list[str]:
        root = Path(self.project_path)
        written = []
        blocks = response.split("===FILE:")
        for block in blocks[1:]:
            try:
                header, rest = block.split("===", 1)
                relative_path = header.strip()
                content = rest.split("===END===")[0].lstrip("\n")
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(relative_path)
                log.info("[coding_tool] 已写入文件: {}", relative_path)
            except Exception as e:
                log.warning("[coding_tool] 解析写入失败: {}", e)
        return written

    def _load_files(self, requirement: str) -> tuple[dict[str, str], str | None]:
        """扫描项目并读取相关文件，返回 (files, error_msg)"""
        root = Path(self.project_path)
        if not root.exists():
            return {}, f"项目路径不存在: {self.project_path}"

        all_paths = self._get_all_paths()
        if not all_paths:
            return {}, f"未在路径 {self.project_path} 下找到任何代码文件"

        log.info("[coding_tool] 项目共扫描到 {} 个代码文件", len(all_paths))
        relevant_paths = self._locate_relevant_files(all_paths, requirement)
        if not relevant_paths:
            return {}, f"未找到与需求相关的文件，项目共 {len(all_paths)} 个文件，请检查需求描述是否包含正确的类名或表名。"

        files = self._read_files(relevant_paths)
        if not files:
            return {}, "相关文件读取失败"

        return files, None

    def _run(
        self,
        requirement: str,
        mode: str = "ask",
        confirmed: Optional[bool] = False,
        last_design: Optional[str] = None,
    ) -> str:
        log.info("[工具调用] coding_tool: mode={}, confirmed={}, 需求={}", mode, confirmed, requirement)

        # ── 设计模式：只输出设计方案，不动代码 ──────────────────────────
        if mode == "design":
            files, err = self._load_files(requirement)
            if err:
                return err
            messages = [
                {"role": "system", "content": _DESIGN_PROMPT},
                {"role": "user", "content": self._build_code_message(files, requirement)},
            ]
            response = llm.invoke(messages)
            log.info("[coding_tool] design 完成")
            return response.content

        # ── 询问模式：先设计，用户确认后再写代码 ────────────────────────
        if mode == "ask":
            if not confirmed:
                # 第一次调用：输出设计方案并询问
                files, err = self._load_files(requirement)
                if err:
                    return err
                messages = [
                    {"role": "system", "content": _ASK_PROMPT},
                    {"role": "user", "content": self._build_code_message(files, requirement)},
                ]
                response = llm.invoke(messages)
                log.info("[coding_tool] ask 设计完成，等待用户确认")
                return response.content
            else:
                # 用户已确认：基于原需求重新读文件并直接生成代码写入
                if not last_design:
                    return "ask 确认模式需要传入 last_design（上一次返回的设计内容）"
                files, err = self._load_files(requirement)
                if err:
                    return err
                messages = [
                    {"role": "system", "content": _DEVELOP_PROMPT},
                    {"role": "user", "content": self._build_code_message(files, f"{requirement}\n\n参考设计方案：\n{last_design}")},
                ]
                response = llm.invoke(messages)
                written = self._parse_and_write(response.content)
                if not written:
                    return "未生成任何文件改动。\n\nAI 回复：\n" + response.content
                log.info("[coding_tool] ask 确认写入完成")
                return f"已完成代码修改，共写入 {len(written)} 个文件：\n" + "\n".join(f"- {f}" for f in written)

        # ── 开发模式：直接修改代码，无需确认 ────────────────────────────
        if mode == "dev":
            files, err = self._load_files(requirement)
            if err:
                return err
            messages = [
                {"role": "system", "content": _DEVELOP_PROMPT},
                {"role": "user", "content": self._build_code_message(files, requirement)},
            ]
            response = llm.invoke(messages)
            written = self._parse_and_write(response.content)
            if not written:
                return "未生成任何文件改动。\n\nAI 回复：\n" + response.content
            log.info("[coding_tool] dev 写入完成")
            return f"已完成代码修改，共写入 {len(written)} 个文件：\n" + "\n".join(f"- {f}" for f in written)

        return f"未知模式: {mode}，请使用 design / ask / dev"
