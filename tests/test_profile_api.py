"""用户画像 API 端点测试（计划 §13, §22）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import agent, app


def _reset_user(user_id: str) -> None:
    """清掉持久化 store 里该用户的记忆与画像反馈，保证测试可重复。

    API 用的是单例 agent（settings.store_root，跨进程持久化），不像构建器单测
    那样有 tmp_path 隔离。不清理的话「全新用户→空状态」会被上一次运行写入的数据污染。
    """
    agent.store.delete_key("memory", user_id)
    agent.store.delete_key("profile_feedback", user_id)


def test_profile_api_empty_then_populated():
    client = TestClient(app)
    _reset_user("profile-api-user")

    # 1) 全新用户 → 空状态引导
    empty = client.get("/profile/profile-api-user")
    assert empty.status_code == 200
    assert empty.json()["is_empty"] is True

    # 2) 喂偏好 + 排除项
    client.post(
        "/memory/update",
        json={
            "user_id": "profile-api-user",
            "event": "我喜欢 R&B 和治愈的流行",
        },
    )
    client.post("/exclusions/profile-api-user", json={"rule": "抖音热歌"})

    populated = client.get("/profile/profile-api-user")
    assert populated.status_code == 200
    body = populated.json()
    assert body["is_empty"] is False
    assert body["summary"]["headline"]
    assert body["summary"]["chips"]
    assert body["sound_fingerprint"]["dimensions"]
    assert "抖音热歌" in body["hard_constraints"]
    assert body["insights"]


def test_profile_insight_feedback_flow():
    client = TestClient(app)
    _reset_user("profile-fb-user")
    client.post(
        "/memory/update",
        json={
            "user_id": "profile-fb-user",
            "event": "我喜欢电子和律动感强的歌",
        },
    )
    profile = client.get("/profile/profile-fb-user").json()
    assert profile["insights"]
    insight_id = profile["insights"][0]["insight_id"]

    # reject
    resp = client.post(
        f"/profile/insights/{insight_id}/feedback", json={"user_id": "profile-fb-user", "action": "reject"}
    )
    assert resp.status_code == 200
    status = next(i["status"] for i in resp.json()["insights"] if i["insight_id"] == insight_id)
    assert status == "rejected"

    # 未知 action → 400
    bad = client.post(
        f"/profile/insights/{insight_id}/feedback", json={"user_id": "profile-fb-user", "action": "nonsense"}
    )
    assert bad.status_code == 400

    # 删除反馈 → 恢复默认
    delete = client.delete(f"/profile/insights/profile-fb-user/{insight_id}")
    assert delete.status_code == 200
    assert delete.json()["deleted"] is True
    reread = client.get("/profile/profile-fb-user").json()
    assert next(i["status"] for i in reread["insights"] if i["insight_id"] == insight_id) == "active"


def test_clear_profile_feedback():
    client = TestClient(app)
    _reset_user("profile-clear-user")
    client.post(
        "/memory/update",
        json={
            "user_id": "profile-clear-user",
            "event": "我喜欢爵士",
        },
    )
    profile = client.get("/profile/profile-clear-user").json()
    insight_id = profile["insights"][0]["insight_id"]
    client.post(f"/profile/insights/{insight_id}/feedback", json={"user_id": "profile-clear-user", "action": "confirm"})

    cleared = client.delete("/profile/profile-clear-user")
    assert cleared.status_code == 200
    # 清除后该 insight 回到 active
    reread = client.get("/profile/profile-clear-user").json()
    assert all(i["status"] == "active" for i in reread["insights"])
