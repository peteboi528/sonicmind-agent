"""用户画像子系统（可解释、可纠错、可驱动推荐的音乐品味仪表盘）。

不是标签墙：把零散偏好/收听/反馈组织成「声音指纹 / 情绪地图 / 场景偏好 /
艺术家关系 / 探索倾向 / 置信度洞察」几个有意义的维度，每个维度都能解释「为什么
这样理解你」，并允许用户纠错。

模块分工（对齐计划 §16）：
- models.py    画像数据契约（与 app/models.py 分离，避免那个大文件继续膨胀）。
- evidence.py  从 UserMemory / listening_history / ratings / taste_profile 收集证据。
- builder.py   把证据组装成结构化画像。
- insights.py  把结构化数据转成用户可读、带置信度的洞察。
- service      已迁至 app/services/profile.py：API 层调用 + insight 反馈落地。
"""

from app.profile.models import (
    ArtistRelation,
    DiscoveryStyle,
    MoodLandscape,
    MoodPoint,
    ProfileInsight,
    ScenePreference,
    SoundFingerprint,
    TasteSummary,
    UserProfileResponse,
)

__all__ = [
    "ArtistRelation",
    "DiscoveryStyle",
    "MoodLandscape",
    "MoodPoint",
    "ProfileInsight",
    "ScenePreference",
    "SoundFingerprint",
    "TasteSummary",
    "UserProfileResponse",
]
