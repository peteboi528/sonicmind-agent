"""GSSC（Globally Shared Sliding Context）同步版上下文管理。

借鉴 SoulTuner 的全局上下文预算思想，但落到同步架构：
- 多源按优先级分配 token 预算：用户输入(0) > 记忆(1) > 历史(2) > 检索(3)
- 每源有 min_tokens 保底，剩余预算按优先级从高到低分配
- 超预算时按行截断兜底（绝不同步调 LLM 压缩——会阻塞主流程）
- 全程 Token 追踪，产出 before/after/saved 报告供透明展示

不做异步预压缩缓存（同步架构下需线程池，复杂度不划算）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """中英混合 token 估算（对齐 SoulTuner 公式）。

    英文约 4 字符/token，中文约 1.5 字符/token。分别统计后相加，
    无需依赖真实 tokenizer，足够做预算分配。
    """
    if not text:
        return 0
    chinese = len(re.findall(r"[一-鿿]", text))
    non_chinese = len(text) - chinese
    return int(chinese / 1.5 + non_chinese / 4) + 1


@dataclass
class ContextSource:
    """一个上下文来源。priority 越小越重要，越后被截断。"""

    name: str
    content: str
    priority: int
    min_tokens: int = 0
    # 对话历史应优先保留最近几轮；规则、画像等来源仍保留开头。
    preserve_tail: bool = False

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


@dataclass
class BudgetReport:
    """Token 预算分配报告，供透明度面板展示。"""

    total_budget: int
    allocations: dict[str, int] = field(default_factory=dict)  # name -> 分配的 token
    original: dict[str, int] = field(default_factory=dict)  # name -> 原始 token
    truncated: list[str] = field(default_factory=list)  # 被截断的源名

    @property
    def original_total(self) -> int:
        return sum(self.original.values())

    @property
    def final_total(self) -> int:
        return sum(self.allocations.values())

    @property
    def saved(self) -> int:
        return max(0, self.original_total - self.final_total)

    def as_lines(self) -> list[str]:
        lines = [
            f"[gssc] 预算 {self.total_budget} tokens，原始 {self.original_total} → 最终 {self.final_total}（节省 {self.saved}）"
        ]
        for name, alloc in self.allocations.items():
            orig = self.original.get(name, 0)
            mark = " (截断)" if name in self.truncated else ""
            lines.append(f"  - {name}: {orig} → {alloc}{mark}")
        return lines


def _truncate_to_tokens(text: str, max_tokens: int, *, preserve_tail: bool = False) -> str:
    """按行截断到目标 token 预算内。

    默认保留开头；对话历史可指定 ``preserve_tail``，避免长会话把最新几轮丢掉。
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    lines = text.splitlines()
    if preserve_tail:
        lines = list(reversed(lines))
    kept: list[str] = []
    used = 0
    for line in lines:
        line_tokens = estimate_tokens(line) + 1
        if used + line_tokens > max_tokens:
            break
        kept.append(line)
        used += line_tokens
    if preserve_tail:
        kept.reverse()
    result = "\n".join(kept)
    # 一行都放不下时，按字符硬截断兜底
    if not result and text:
        approx_chars = max_tokens * 2
        result = text[-approx_chars:] if preserve_tail else text[:approx_chars]
    return result


class ContextBudgetManager:
    """按优先级把多源上下文压进 token 预算。"""

    def __init__(self, total_budget: int = 3000) -> None:
        self.total_budget = total_budget

    def allocate(self, sources: list[ContextSource]) -> tuple[dict[str, str], BudgetReport]:
        """返回 (name -> 截断后内容, 预算报告)。

        策略：先给每源保底 min_tokens；剩余预算按优先级(小→大)依次满足；
        超出预算的低优先级源被按行截断。
        """
        report = BudgetReport(total_budget=self.total_budget)
        for s in sources:
            report.original[s.name] = s.tokens

        ordered = sorted(sources, key=lambda s: s.priority)
        # 第一遍：保底
        reserved = sum(min(s.min_tokens, s.tokens) for s in ordered)
        remaining = max(0, self.total_budget - reserved)

        result: dict[str, str] = {}
        allocations: dict[str, int] = {}
        for s in ordered:
            floor = min(s.min_tokens, s.tokens)
            want = s.tokens - floor  # 保底之外还想要的
            grant = min(want, remaining)
            remaining -= grant
            budget_for_source = floor + grant
            if budget_for_source >= s.tokens:
                result[s.name] = s.content
                allocations[s.name] = s.tokens
            else:
                result[s.name] = _truncate_to_tokens(
                    s.content,
                    budget_for_source,
                    preserve_tail=s.preserve_tail,
                )
                allocations[s.name] = estimate_tokens(result[s.name])
                report.truncated.append(s.name)

        # 按原始顺序回填报告
        report.allocations = {s.name: allocations.get(s.name, 0) for s in sources}
        return result, report
