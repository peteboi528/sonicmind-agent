"""Reflection prompt — ReAct 每轮工具调用后让 LLM 判断是否继续。"""

REFLECTION_VERSION = "v1-2026-06-05"

REFLECTION_SYSTEM = """\
你刚执行了一些工具调用。根据工具返回结果，判断是否已经收集到足够信息回答用户。

判断标准：
- 用户的核心诉求是否已被覆盖？
- 工具返回的数据是否质量足够（不是空、不是错误）？
- 是否还有明显遗漏的维度（如：推荐了但没说理由、给了候选但没整合）？

输出 JSON：{"done": true/false, "reason": "简短理由", "next_intent": "如果未完成，建议下一步关注什么"}
"""
