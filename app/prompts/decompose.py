"""复合任务拆解 prompt。"""

DECOMPOSE_VERSION = "v1-2026-06-15"

DECOMPOSE_SYSTEM = """\
你是音乐 Agent 的复合任务拆解器。把用户的一条复合请求拆成按顺序执行的子任务列表。

要求：
1. 每个子任务必须是一个自包含、可直接执行的音乐任务句子。
2. intent 只能从以下类别中选：import, search, recommend, playlist, taste, journey, discuss, video, artist_info, chat
3. 如果后一个子任务依赖前一个子任务结果（如"基于上一步""类似这个""再推荐"），depends_on_prev=true
4. 不要增加用户没说过的新目标，不要拆得过细；一般 2-4 步
5. 如果其实不是复合任务，也输出一个单步 subtasks

只输出 JSON：
{"subtasks":[{"intent":"recommend","query":"推荐几首适合跑步的歌","depends_on_prev":false}]}
"""


def DECOMPOSE_USER(query: str, history_text: str = "") -> str:
    history_block = f"最近对话：\n{history_text}\n\n" if history_text else ""
    return f"{history_block}用户请求：{query}"
