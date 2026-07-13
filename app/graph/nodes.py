"""Graph nodes — 纯门面（facade），无业务逻辑。

业务逻辑已按 stage 拆分到同包各子模块：

  _shared.py       共享工具（意图判定 / 曲目选取 / prompt 版本）
  budget.py        单轮预算 / 降级
  continuation.py  跨轮对话 / 延续实体
  planning.py      意图 / 计划
  execution.py     工具执行 / web 兜底
  recovery.py      反射 / 空结果恢复 / 候选过滤
  finalize.py      答案组装 / final payload / trace 摘要

本模块仅 re-export 这些符号，保持 ``from app.graph.nodes import X`` 的旧入口不变：
builder / tests / evals / handlers 以及子模块内部的 lazy seam（planning、recovery、finalize
均 ``from app.graph.nodes import select_llm`` 等，使外部对 ``nodes.<name>`` 的 monkeypatch 生效）
都依赖此门面集中可注入。新增业务逻辑请落到对应子模块，而非此处。
"""
from __future__ import annotations

from app.answer import guard_answer  # noqa: F401  # re-export: finalize lazy 从本模块读取
from app.graph._shared import (  # noqa: F401
    _is_knowledge_intent,
    _select_listed_tracks,
)
from app.graph.continuation import (  # noqa: F401
    _apply_dialogue_continuation,
    _persist_dialogue_state,
    _query_with_entities,
)
from app.graph.execution import (  # noqa: F401
    _infer_aux_arguments,
    _needs_web_fallback,
    _record_runtime_result,
    _run_tool_async,
    execute_tools_async,
    route_after_execute,
    web_fallback_async,
)
from app.graph.finalize import (  # noqa: F401
    _artist_info_prompt,
    _chunk_for_stream,
    _compose_deterministic_answer,
    _discussion_prompt,
    _finalize_fallback,
    _finalize_tail_async,
    _taste_experiment_card,
    _trace_summary,
    compose_answer_stream_async,
    finalize_stream_async,
)
from app.graph.planning import (  # noqa: F401
    _finish_plan_intent,
    _inject_preference_seeds,
    _materialize_tool_stages,
    _merge_multi_intent_stages,
    _plan_from_query_payload,
    _planned_arguments,
    build_agent_plan,
    load_context,
    plan_intent_async,
    plan_with_llm_async,
    plan_with_llm_with_meta_async,
)
from app.graph.recovery import (  # noqa: F401
    _drop_tracks_from_results,
    _prepare_empty_result_recovery_async,
    _track_key,
    evaluate,
    reflect_async,
    route_after_reflect,
)
from app.llm.routing import select_llm  # noqa: F401  # re-export: planning/recovery/finalize lazy seam + tests
from app.tools.handlers import _apply_language_filter  # noqa: F401  # re-export: tests/test_query_rewrite 从本模块导入
