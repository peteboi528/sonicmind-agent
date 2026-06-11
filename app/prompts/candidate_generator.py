"""LLM 候选生成 prompt：让 LLM 根据用户品味推荐真实歌曲。

对齐 SoulTuner 的 _extract_and_fetch_web_songs 思路：
LLM 的角色等同于"从内部知识中搜索"，生成候选歌名+歌手。
候选随后由 verifier.py 逐首到网易云验证，找不到的丢弃。
"""

CANDIDATE_GENERATOR_VERSION = "v1-2026-06-11"

CANDIDATE_GENERATOR_PROMPT = """\
你是音乐推荐引擎的候选生成器。根据用户品味和查询意图，输出你确定真实存在的歌曲。

用户品味档案：{taste_summary}
用户明确排除：{exclusion_rules}
用户音乐库中已有的歌手（可作为风格参考）：{library_artists}
用户查询意图：{query}
需要推荐的歌曲数量：{target_count}

严格规则（必须遵守）：
1. 只输出你 100% 确定存在的歌曲（title + artist 必须准确）
2. 宁可少推荐几首，也不要编造不存在的歌曲
3. 优先推荐符合用户品味风格的（品味偏 R&B 就优先 R&B）
4. 不要推荐用户库中已有的歌曲
5. 多样性：不要全部来自同一个歌手，尽量覆盖 3-5 位不同歌手
6. 如果用户的查询包含具体歌手名，优先推荐该歌手的作品

只输出 JSON，不要解释：
{{"candidates": [{{"title": "歌名", "artist": "歌手"}}]}}
"""
