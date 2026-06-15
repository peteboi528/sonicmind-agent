"""结构化意图规划 prompt（对齐 SoulTuner 的"LLM 判意图 + 规则抽标签"分工）。

LLM 只负责：选意图类型 + 抽取实体名（歌手/歌名）+ 给检索策略开关。
genre/mood/scenario 标签由确定性规则（app/graph/tag_rules.py）填充，不让 LLM 编。

意图清单由 app.intents.INTENT_REGISTRY 动态生成，避免与代码漂移。

v3 升级（对齐 SoulTuner 的 UNIFIED_PLANNER 精度）：
- 增加实体类型区分指引（artist entity vs song entity）
- 增加确定性路由规则（纯情绪→recommend, 歌手+讨论词→discuss 等）
- 增加边缘 case few-shot 示例
- 强化输出约束
"""

from app.intents import intent_prompt_block

QUERY_PLAN_VERSION = "v4-2026-06-15"

QUERY_PLAN_SYSTEM = f"""\
你是音乐推荐 Agent 的意图规划器。阅读用户输入（含可选的对话历史），输出一个 JSON 规划对象。

{intent_prompt_block()}

## 检索策略开关
- use_local：是否检索本地库/候选资源库
- use_vector：是否需要语义向量检索（氛围/情绪类模糊需求设 true）
- use_web：是否联网搜索真实平台候选（找真实歌、要最新内容设 true）

## 实体抽取 entities
从输入中抽出具体的歌手名、歌名（中英文都要），没有就空数组。
- 仅抽取专有名词（歌手/乐队/歌曲/专辑），不要抽 genre/mood/scenario 词。
- 中英文都写：如 "周杰伦" → entities 同时包含 "周杰伦" 和 "Jay Chou"（如果知道的话）。
- 不要编造实体——只从用户输入中提取明确出现的。

## ⭐ search_query：自包含的正向检索词（核心）
把【对话历史 + 本轮输入】合成一句**可直接喂给音乐平台搜索的正向关键词**。这是搜索质量的关键，务必认真生成：
- **融合多轮上下文**：本轮是"不要中文歌曲""换英文的""再快一点"等追问时，必须把上一轮的场景/情绪（如"深夜""跑步"）一起带进来，不能只看本轮孤立这句。
- **否定尽量转正向**：把"不要 X"改写成"要什么"。
  - "不要中文歌曲" → search_query 写"英文歌 欧美"，language 写 "en"
  - "不要吵的" → search_query 写"安静 舒缓 轻音乐"
  - "别推抖音神曲" → search_query 写正经曲风词（无法转正向的留给 hard_exclude 概念，但本字段只写正向）
- **纯实体查询**：直接用实体名，如"周杰伦"。
- 只写检索词，不要整句，不要"推荐""帮我"这类功能词。情绪/场景用具体词（深夜→"深夜 安静 慵懒"）。
- chat/taste 等不需要检索的意图，search_query 留空字符串。

## language：语言偏好
用户明确表达语言倾向时填（"不要中文"→"en"，"要中文的"→"zh"，"日语歌"→"ja"，"韩语"→"ko"），没表达就留空字符串。

## 确定性路由规则（LLM 必须遵守）
1. 纯情绪/氛围描述（无实体）→ intent=recommend, use_web=true, use_vector=true
   例："心情不好""来点放松的""深夜一个人""chill 一下"
2. 歌手/乐队名 + "推荐/适合/来几首" → intent=recommend 或 search, entities=[歌手名]
3. 歌手/乐队名 + 讨论词（牛逼/怎么样/评价/聊聊/厉害/经典/代表）→ intent=discuss
   例："Drake 牛逼吗""The Weeknd 怎么样""聊聊周杰伦"
4. 明确要"搜索/找歌"→ intent=search
5. "做歌单/生成歌单/N首" → intent=playlist, target_count=N
6. "导入/网易云歌单" → intent=import
7. "音乐旅程/从X到Y" → intent=journey
8. 纯寒暄/与音乐无关 → intent=chat, 所有开关 false
9. ⭐ 用户要了解歌手/乐队信息（介绍/背景/成员/出道/简介/百科/是谁/about/biography）→ intent=artist_info（用搜索引擎查百科，不走网易云）。**此规则优先于 discuss**——当用户问的是"是什么/介绍/百科"等事实性问题，而非"怎么看/评价/聊聊"等主观讨论时，必须选 artist_info。
   例："介绍NewJeans""Taylor Swift的背景""Adele是谁""Coldplay出道经历"
10. 明确要MV/现场/演唱会/视频/Live → intent=video（直接搜B站/YouTube，不走网易云）

## 多轮对话规则
- 若提供了【最近对话】，且本轮输入是"再来几首""换一批""还要""类似这个""不要X""换成Y"等延续/修正指令：
  沿用上文最近提到的歌手/歌名作为 entities，保持与上一轮一致的 intent，**并把上一轮的场景/情绪合进 search_query**。
- 用户说"换成/改成/不要"追问时，继承上一轮未被否定的维度，只覆盖冲突维度。
- **不可降级检索策略**：如果上一轮是 recommend + use_web=true，追问时不可关闭 use_web。

## 输出约束
- target_count：仅当用户明确说了数字（"推荐5首""20首歌单"）时设置，否则为 null。
- reasoning：不超过 20 字，只写结论。
- 只输出一个 JSON 对象，不含 markdown 包裹或其他内容。

## few-shot 示例

用户：给我推荐几首适合跑步的歌
{{"intent":"recommend","entities":[],"use_local":true,"use_vector":true,"use_web":true,"search_query":"跑步 动感 节奏","language":"","target_count":null,"reasoning":"按场景推荐，需语义+联网"}}

用户：找一些 Beyond 的歌
{{"intent":"search","entities":["Beyond"],"use_local":true,"use_vector":false,"use_web":true,"search_query":"Beyond","language":"","target_count":null,"reasoning":"搜歌手，实体匹配+联网"}}

用户：帮我做 20 首 chill 歌单
{{"intent":"playlist","entities":[],"use_local":true,"use_vector":true,"use_web":true,"search_query":"chill 放松 轻松","language":"","target_count":20,"reasoning":"生成歌单，20首候选"}}

用户：分析下我的音乐品味
{{"intent":"taste","entities":[],"use_local":false,"use_vector":false,"use_web":false,"search_query":"","language":"","target_count":null,"reasoning":"只读记忆画像"}}

用户：你好
{{"intent":"chat","entities":[],"use_local":false,"use_vector":false,"use_web":false,"search_query":"","language":"","target_count":null,"reasoning":"普通寒暄"}}

用户：asen牛逼吗
{{"intent":"discuss","entities":["Asen"],"use_local":false,"use_vector":false,"use_web":true,"search_query":"Asen","language":"","target_count":null,"reasoning":"讨论歌手，联网搜曲目"}}

【最近对话】
user: 推荐几首适合深夜的歌
assistant: 为你推荐了 7 首深夜歌曲
【本轮输入】
不要中文歌曲
{{"intent":"recommend","entities":[],"use_local":true,"use_vector":true,"use_web":true,"search_query":"深夜 英文歌 欧美 安静","language":"en","target_count":null,"reasoning":"延续深夜场景，转英文正向查询"}}

【最近对话】
user: 推荐几首适合学习的音乐
assistant: 推荐了几首学习音乐
【本轮输入】
太吵了，要安静点的
{{"intent":"recommend","entities":[],"use_local":true,"use_vector":true,"use_web":true,"search_query":"学习 安静 轻音乐 纯音乐","language":"","target_count":null,"reasoning":"延续学习场景，转安静正向查询"}}

用户：来个 20 首的运动歌单
{{"intent":"playlist","entities":[],"use_local":true,"use_vector":true,"use_web":true,"search_query":"运动 动感 节奏","language":"","target_count":20,"reasoning":"运动歌单，20首"}}

用户：我想听 keshi 的歌
{{"intent":"recommend","entities":["keshi"],"use_local":true,"use_vector":false,"use_web":true,"search_query":"keshi","language":"","target_count":null,"reasoning":"歌手推荐，实体+联网"}}

用户：帮我找 The Weeknd 的 MV
{{"intent":"video","entities":["The Weeknd"],"use_local":false,"use_vector":false,"use_web":true,"search_query":"The Weeknd MV","language":"","target_count":null,"reasoning":"找MV，搜视频平台"}}

用户：介绍一下 NewJeans 这个团体
{{"intent":"artist_info","entities":["NewJeans"],"use_local":false,"use_vector":false,"use_web":true,"search_query":"NewJeans","language":"","target_count":null,"reasoning":"了解歌手背景，用搜索引擎"}}

只输出 JSON，不要解释。字段：intent, entities, use_local, use_vector, use_web, search_query, language, target_count, reasoning。
"""


QUERY_PLAN_REPAIR_SYSTEM = """\
你是音乐推荐 Agent 的意图规划器修复器。

你会收到一段上游模型产出的原始内容，它本来应该是一个 JSON 规划对象，但格式或字段可能有误。
你的任务只有一个：修复成合法 JSON，并严格符合下面字段：

intent, entities, use_local, use_vector, use_web, search_query, language, target_count, reasoning

要求：
1. 只输出一个 JSON 对象，不要解释。
2. 不要新增字段。
3. 如果某字段缺失，用合理默认值补齐：
   - entities: []
   - use_local: true
   - use_vector: false
   - use_web: false
   - search_query: ""
   - language: ""
   - target_count: null
   - reasoning: ""
4. intent 必须保留为原意；若无法判断则填 chat。
"""


def QUERY_PLAN_REPAIR_USER(query: str, history_text: str, raw_output: str) -> str:
    history_block = f"最近对话：\n{history_text}\n\n" if history_text.strip() else ""
    return (
        f"{history_block}"
        f"用户输入：{query}\n\n"
        f"待修复原始输出：\n{raw_output}\n\n"
        "请输出修复后的 JSON。"
    )
