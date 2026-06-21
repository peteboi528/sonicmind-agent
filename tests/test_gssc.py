"""Phase 2 GSSC 上下文预算回归测试。"""

from __future__ import annotations

from app.context.gssc import (
    ContextBudgetManager,
    ContextSource,
    estimate_tokens,
)


def test_estimate_tokens_mixed():
    assert estimate_tokens("") == 0
    # 纯中文：约 1.5 字符/token
    assert estimate_tokens("推荐音乐") >= 2
    # 纯英文：约 4 字符/token
    assert estimate_tokens("recommend some music") >= 4


def test_high_priority_source_survives_truncation():
    mgr = ContextBudgetManager(total_budget=50)
    sources = [
        ContextSource(name="user_query", content="重要的用户问题" * 2, priority=0, min_tokens=20),
        ContextSource(name="history", content="历史对话内容很长" * 50, priority=2, min_tokens=5),
    ]
    out, report = mgr.allocate(sources)
    # 用户输入完整保留，历史被截断
    assert out["user_query"] == "重要的用户问题" * 2
    assert "history" in report.truncated
    assert report.final_total <= report.original_total


def test_budget_report_accounts_savings():
    mgr = ContextBudgetManager(total_budget=30)
    sources = [ContextSource(name="memory", content="记忆内容" * 100, priority=1, min_tokens=5)]
    _, report = mgr.allocate(sources)
    assert report.saved > 0
    assert any("gssc" in line for line in report.as_lines())


def test_small_content_not_truncated():
    mgr = ContextBudgetManager(total_budget=5000)
    sources = [ContextSource(name="user_query", content="短问题", priority=0, min_tokens=10)]
    out, report = mgr.allocate(sources)
    assert out["user_query"] == "短问题"
    assert report.truncated == []


def test_long_dialogue_history_keeps_latest_turns():
    mgr = ContextBudgetManager(total_budget=24)
    history = "\n".join(f"user: 第 {i} 轮消息" for i in range(20))
    sources = [
        ContextSource(
            name="history",
            content=history,
            priority=2,
            min_tokens=4,
            preserve_tail=True,
        )
    ]

    out, report = mgr.allocate(sources)

    assert "第 19 轮消息" in out["history"]
    assert "第 0 轮消息" not in out["history"]
    assert "history" in report.truncated
