from fastapi.testclient import TestClient

from app.api.main import app


def test_api_flow():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert "auth_mode" in health.json()["details"]
    assert "smoke" in health.json()["details"]
    assert "frontend_build_hash" in health.json()["details"]

    ingest = client.post("/assets/ingest", json={"url": "https://example.com/api-test"})
    assert ingest.status_code == 200
    asset_id = ingest.json()["asset_id"]

    analyze = client.post(f"/assets/{asset_id}/analyze")
    assert analyze.status_code == 200
    assert len(analyze.json()["segments"]) == 6

    listen = client.post("/listen", json={"user_id": "api-user", "asset_id": asset_id, "duration": 180, "completed": True})
    assert listen.status_code == 200
    assert listen.json()["memory_updated"] is True

    update = client.post("/memory/update", json={"user_id": "api-user", "event": "我喜欢电子音乐和放松的氛围"})
    assert update.status_code == 200
    assert update.json()["updated"] is True

    taste = client.get("/taste/api-user")
    assert taste.status_code == 200

    experiment = client.post("/taste/experiment/generate", json={
        "user_id": "api-user",
        "prompt": "推荐点不一样的，做个品味实验",
        "total": 9,
    })
    assert experiment.status_code == 200
    exp_json = experiment.json()
    assert [s["name"] for s in exp_json["segments"]] == ["safe", "stretch", "bold"]
    first_item = next(item for s in exp_json["segments"] for item in s["tracks"])
    track = first_item["track"]
    track_key = f'{track["source"]}:{track["source_id"]}' if track["source_id"] else f'title:{track["title"].lower()}:{track["artist"].lower()}'
    feedback = client.post("/taste/experiment/feedback", json={
        "user_id": "api-user",
        "experiment_id": exp_json["experiment_id"],
        "track_key": track_key,
        "signal": "liked",
    })
    assert feedback.status_code == 200
    report = client.post("/taste/experiment/report", json={
        "user_id": "api-user",
        "experiment_id": exp_json["experiment_id"],
    })
    assert report.status_code == 200
    assert "bucket_stats" in report.json()

    daily = client.post("/recommend/daily", json={"user_id": "api-user", "time_of_day": "evening"})
    assert daily.status_code == 200
    assert daily.json()["tracks"]

    search = client.post("/search", json={"user_id": "api-user", "query": "电子 放松"})
    assert search.status_code == 200
    assert "summary" in search.json()
    assert search.json()["agent_trace"]

    chat = client.post("/chat", json={"user_id": "api-user", "message": "推荐一些适合工作时听的音乐"})
    assert chat.status_code == 200
    chat_data = chat.json()
    assert chat_data["answer"]
    assert chat_data["agent_trace"]
    # QW5：同步 /chat 也应携带与 SSE 路径一致的透明度摘要。
    assert chat_data.get("trace_summary")
    assert chat_data["trace_summary"].get("intent")

    agent_run = client.post("/agent/run", json={"user_id": "api-user", "message": "分析我的品味"})
    assert agent_run.status_code == 200
    run_data = agent_run.json()
    assert run_data["answer"]
    assert run_data["agent_trace"]
    assert run_data.get("trace_summary")
    assert run_data["trace_summary"].get("intent")

    enrich = client.post(f"/assets/{asset_id}/enrich", json={"use_network": False})
    assert enrich.status_code == 200
    assert enrich.json()["mode"] == "offline"

    assets = client.get("/assets")
    assert assets.status_code == 200
    assert len(assets.json()["assets"]) >= 1

    refreshed = client.post("/assets/ingest", json={"url": "https://example.com/api-test", "force_refresh": True})
    assert refreshed.status_code == 200
    assert refreshed.json()["asset_id"] == asset_id

    forced_analyze = client.post(f"/assets/{asset_id}/analyze?force_refresh=true")
    assert forced_analyze.status_code == 200
    assert len(forced_analyze.json()["segments"]) == 6

    delete_asset = client.delete(f"/assets/{asset_id}")
    assert delete_asset.status_code == 200
    assert delete_asset.json()["deleted"] is True

    clear_cache = client.delete("/cache?preserve_memory=false")
    assert clear_cache.status_code == 200
    assert "assets" in clear_cache.json()["cleared"]


def test_ingest_full_runs_three_steps():
    """ingest_full 必须串起 ingest→enrich→analyze（修复 Web 入库只调一步、歌曲不识别的回归）。"""
    client = TestClient(app)
    resp = client.post("/assets/ingest_full", json={"url": "https://example.com/full-test"})
    assert resp.status_code == 200
    asset = resp.json()
    assert asset["asset_id"]
    # analyze 步骤跑过 → 状态应为 analyzed（而非停在 ingested 占位）
    assert asset["status"] == "analyzed"
    client.delete(f"/assets/{asset['asset_id']}")


def test_artist_album_tracks_returns_full_ordered_album(monkeypatch):
    """专辑点击不能复用 top_k=12 的搜索结果；应返回专辑详情原始顺序。"""
    from app.sources import netease as ns

    monkeypatch.setattr(ns, "search_netease_album", lambda artist, album: {
        "id": "18893",
        "name": album,
        "artist": artist,
        "cover": "cover",
        "track_count": 14,
    })
    monkeypatch.setattr(ns, "fetch_netease_album_tracks", lambda album_id, limit=100: {
        "id": album_id,
        "name": "Ordered Album",
        "artist": "Artist",
        "cover": "cover",
        "track_count": 14,
        "tracks": [
            {"song_id": str(i), "title": f"Track {i}", "artist": "Artist", "album": "Ordered Album", "cover": "cover"}
            for i in range(1, 15)
        ],
    })

    client = TestClient(app)
    resp = client.post("/artist/album_tracks", json={"artist": "Artist", "album": "Ordered Album"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["album"]["id"] == "18893"
    assert len(data["tracks"]) == 14
    assert [t["title"] for t in data["tracks"][:3]] == ["Track 1", "Track 2", "Track 3"]


def test_discover_trending_uses_current_billboard_chart(monkeypatch):
    """Billboard 不能再读取停更的“美国热门歌曲”歌单 11641012。"""
    from app.models import ExternalTrack
    from app.search import netease_playlist

    requested_ids = []

    def fake_detail(playlist_id, limit=8):
        requested_ids.append(playlist_id)
        names = {
            3778678: "热歌榜",
            19723756: "飙升榜",
            60198: "美国Billboard榜",
            180106: "UK排行榜周榜",
            3812895: "Beatport全球电子舞曲榜",
        }
        return {
            "id": str(playlist_id),
            "name": names[playlist_id],
            "updated_at": "2026-06-18T00:00:00+00:00",
            "track_count": 1,
            "tracks": [ExternalTrack(
                external_id=str(playlist_id), title="Current Track", artist="Artist", source="netease"
            )],
        }

    monkeypatch.setattr(netease_playlist, "get_playlist_detail", fake_detail)
    monkeypatch.setattr("app.api.main.settings.lastfm_api_key", "")

    response = TestClient(app).post("/discover/trending", json={"user_id": "api-user", "limit": 8})

    assert response.status_code == 200
    charts = response.json()["charts"]
    billboard = next(chart for chart in charts if chart["name"] == "美国 Billboard")
    assert billboard["chart_id"] == "60198"
    assert billboard["source_name"] == "美国Billboard榜"
    assert billboard["updated_at"] == "2026-06-18T00:00:00+00:00"
    assert 11641012 not in requested_ids
    assert 60198 in requested_ids


def test_artist_album_tracks_not_found_returns_empty(monkeypatch):
    """找不到专辑时不能回退到乱序搜索结果：返回 200 + 空 tracks + 可读 summary。"""
    from app.sources import netease as ns

    monkeypatch.setattr(ns, "search_netease_album", lambda artist, album: None)
    monkeypatch.setattr(ns, "fetch_netease_album_tracks", lambda album_id, limit=100: None)

    client = TestClient(app)
    resp = client.post("/artist/album_tracks", json={"artist": "谁", "album": "不存在的专辑"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["tracks"] == []
    assert data["summary"]  # 可读非空提示
    assert data["album"]["name"] == "不存在的专辑"


def test_artist_info_top_albums_use_netease_ids(monkeypatch):
    """歌手页代表专辑优先用网易云带真实 id 的专辑，而非 Last.fm 无 id 的（点击免二次猜匹配）。"""
    from app.api import main as main_module
    from app.models import ExternalTrack
    from app.sources import netease as ns

    # 强制离线：关掉 Last.fm（否则开发机的 key 会触发真实网络请求）
    monkeypatch.setattr("app.api.main.settings.lastfm_api_key", "")
    monkeypatch.setattr(ns, "search_netease_artist_albums", lambda artist, limit=6: [
        {"id": "18893", "name": "依然范特西", "image": "cover", "artist": "周杰伦", "track_count": 10},
        {"id": "18894", "name": "叶惠美", "image": "cover2", "artist": "周杰伦", "track_count": 11},
    ])
    monkeypatch.setattr(ns, "search_netease_artist_image", lambda artist: None)
    monkeypatch.setattr(main_module.agent, "search_web_music", lambda *args, **kwargs: [
        ExternalTrack(external_id="good", title="晴天", artist="周杰伦", source="netease"),
        ExternalTrack(external_id="noise", title="热门歌曲", artist="热门歌曲", source="netease"),
    ])

    client = TestClient(app)
    resp = client.post("/artist/info", json={"artist": "周杰伦"})

    assert resp.status_code == 200
    albums = resp.json()["top_albums"]
    assert albums
    assert albums[0]["id"] == "18893"
    assert albums[0]["track_count"] == 10
    assert albums[1]["id"] == "18894"
    assert resp.json()["matched"] is True
    assert [track["title"] for track in resp.json()["top_tracks"]] == ["晴天"]


def test_artist_info_localizes_english_bio_and_keeps_twelve_albums(monkeypatch):
    from app.api import main as main_module
    from app.sources import netease as ns
    from app.sources.lastfm_client import LastfmClient

    monkeypatch.setattr("app.api.main.settings.lastfm_api_key", "dummy-key")
    monkeypatch.setattr("app.api.main.settings.llm_api_key", "dummy-key")
    monkeypatch.setattr(LastfmClient, "get_artist_info", lambda self, artist: {
        "name": "Kanye West",
        "image": "lfm-image",
        "bio": (
            "Kanye West, born Kanye Omari West on June 8, 1977, is an American rapper, singer, "
            "songwriter, record producer, and fashion designer. Read more on Last.fm"
        ),
        "tags": ["hip hop", "producer"],
    })
    monkeypatch.setattr(LastfmClient, "get_artist_top_albums", lambda self, artist, limit=12: [
        {"name": f"Lastfm Album {idx}", "image": ""}
        for idx in range(limit)
    ])
    monkeypatch.setattr(ns, "search_netease_artist_albums", lambda artist, limit=12: [
        {"id": str(idx), "name": f"Album {idx}", "image": "", "artist": "Kanye West", "track_count": 10 + idx}
        for idx in range(limit)
    ])
    monkeypatch.setattr(ns, "search_netease_artist_image", lambda artist: None)
    monkeypatch.setattr("app.sources.web_search.search_web_info", lambda *args, **kwargs: [])
    monkeypatch.setattr(main_module.agent, "search_web_music", lambda *args, **kwargs: [])

    async def _fake_agenerate(prompt, system=None, temperature=0.2, thinking=None):
        return "Kanye West 是美国说唱歌手、制作人和时尚设计师，以多变的制作风格和强烈个人表达闻名。"

    monkeypatch.setattr(main_module.agent.llm_fast, "agenerate", _fake_agenerate)

    resp = TestClient(app).post("/artist/info", json={"artist": "Kanye West"})

    assert resp.status_code == 200
    data = resp.json()
    assert "美国说唱歌手" in data["bio"]
    assert "Read more on Last.fm" not in data["bio"]
    assert len(data["top_albums"]) == 12


def test_artist_info_falls_back_to_chinese_web_bio_when_localization_fails(monkeypatch):
    from app.api import main as main_module
    from app.sources import netease as ns
    from app.sources.lastfm_client import LastfmClient

    monkeypatch.setattr("app.api.main.settings.lastfm_api_key", "dummy-key")
    monkeypatch.setattr("app.api.main.settings.llm_api_key", "dummy-key")
    monkeypatch.setattr(LastfmClient, "get_artist_info", lambda self, artist: {
        "name": "Kanye West",
        "image": "lfm-image",
        "bio": (
            "Ye, born Kanye Omari West on June 8, 1977, is an American rapper, singer, songwriter, "
            "record producer, and fashion designer. He is one of the most prominent figures in hip hop."
        ),
        "tags": ["hip hop", "producer"],
    })
    monkeypatch.setattr(LastfmClient, "get_artist_top_albums", lambda self, artist, limit=12: [])
    monkeypatch.setattr(ns, "search_netease_artist_albums", lambda artist, limit=12: [
        {"id": str(idx), "name": f"Album {idx}", "image": "", "artist": "Kanye West", "track_count": 10 + idx}
        for idx in range(12)
    ])
    monkeypatch.setattr(ns, "search_netease_artist_image", lambda artist: None)
    monkeypatch.setattr(
        "app.sources.web_search.search_web_info",
        lambda *args, **kwargs: [{
            "title": "Kanye West 简介",
            "url": "https://example.com/kanye",
            "content": (
                "Kanye West 是美国说唱歌手、制作人和时装设计师，早年以 Roc-A-Fella 的制作人身份崭露头角，"
                "之后凭《The College Dropout》《Late Registration》《Graduation》奠定地位。"
                "他在《808s & Heartbreak》《My Beautiful Dark Twisted Fantasy》《Yeezus》等阶段不断调整声音方向，"
                "既推动主流嘻哈审美变化，也因公开言论和跨界项目长期处于舆论中心。"
            ),
        }],
    )
    monkeypatch.setattr(main_module.agent, "search_web_music", lambda *args, **kwargs: [])

    async def _fail_agenerate(*args, **kwargs):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(main_module.agent.llm_fast, "agenerate", _fail_agenerate)

    resp = TestClient(app).post("/artist/info", json={"artist": "Kanye West"})

    assert resp.status_code == 200
    data = resp.json()
    assert "美国说唱歌手" in data["bio"]
    assert "The College Dropout" in data["bio"]
    assert "Ye, born Kanye" not in data["bio"]
    assert len(data["top_albums"]) == 12


def test_discover_classify_routes_activity_away_from_artist_page():
    client = TestClient(app)

    response = client.post("/discover/classify", json={"query": "跑步时听的电子音乐"})

    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "category"
    assert data["browse_category"] == "scene"
    assert "运动" in data["tags"]["scenario"]
    assert "电子" in data["tags"]["genre"]


def test_discover_classify_treats_bare_artist_name_as_artist():
    client = TestClient(app)

    response = client.post("/discover/classify", json={"query": "周杰伦"})

    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "artist"
    assert data["normalized_query"] == "周杰伦"
    assert data["reason"] == "bare_artist_shape"


def test_discover_classify_keeps_focus_like_terms_off_artist_route():
    client = TestClient(app)

    response = client.post("/discover/classify", json={"query": "专注"})

    assert response.status_code == 200
    data = response.json()
    assert data["kind"] != "artist"


def test_discover_search_returns_local_and_bounded_external_results(monkeypatch):
    from app.api import main as main_module
    from app.models import Asset, ExternalTrack, SearchResponse

    local = Asset(
        asset_id="local-1", source_url="https://example.com/local-1",
        title="Blinding Lights", artist="The Weeknd", duration_seconds=200,
        status="analyzed",
    )
    monkeypatch.setattr(main_module.agent, "search", lambda *args, **kwargs: SearchResponse(
        local=[local], external=[], summary="local first", evidences=[], agent_trace=[],
    ))
    monkeypatch.setattr(main_module.agent, "search_web_music", lambda *args, **kwargs: [
        ExternalTrack(external_id="online-1", title="Blinding Lights", artist="The Weeknd", source="netease")
    ])

    client = TestClient(app)

    # 本地那次：只读曲库，秒级返回，不带在线候选。
    local_resp = client.post("/discover/search", json={
        "user_id": "discover-user", "query": "Blinding Lights",
        "include_external": False, "top_k": 12,
    })
    assert local_resp.status_code == 200
    assert local_resp.json()["local"][0]["title"] == "Blinding Lights"
    assert local_resp.json()["external"] == []

    # 在线那次：external_only，跳过本地，只回在线候选。
    ext_resp = client.post("/discover/search", json={
        "user_id": "discover-user", "query": "Blinding Lights",
        "external_only": True, "top_k": 12,
    })
    assert ext_resp.status_code == 200
    assert ext_resp.json()["local"] == []
    assert ext_resp.json()["external"][0]["source"] == "netease"
    assert "在线找到 1 首" in ext_resp.json()["summary"]


def test_artist_info_rejects_fuzzy_non_artist_match(monkeypatch):
    from app.api import main as main_module
    from app.models import ExternalTrack
    from app.sources import netease as ns

    monkeypatch.setattr("app.api.main.settings.lastfm_api_key", "")
    monkeypatch.setattr(ns, "search_netease_artist_albums", lambda artist, limit=6: [])
    monkeypatch.setattr(ns, "search_netease_artist_image", lambda artist: None)
    monkeypatch.setattr(main_module.agent, "search_web_music", lambda *args, **kwargs: [
        ExternalTrack(external_id="1", title="Focus", artist="Unrelated Artist", source="netease")
    ])

    response = TestClient(app).post("/artist/info", json={"artist": "专注"})

    assert response.status_code == 200
    assert response.json()["matched"] is False


def test_saved_album_save_list_delete_and_isolation():
    """收藏专辑：保存→状态→列表→用户隔离→删除 全链路。"""
    client = TestClient(app)
    uid = "album-saver"
    tracks = [{"external_id": "s1", "title": "T1", "artist": "A", "source": "netease", "playback_url": "https://music.163.com/song?id=1"}]

    save = client.post("/album/save", json={
        "user_id": uid, "album_id": "18893", "name": "Album", "artist": "A",
        "track_count": 1, "tracks": tracks,
    })
    assert save.status_code == 200
    assert save.json()["saved"] is True

    # 状态查询
    assert client.get(f"/album/saved/{uid}/18893").json()["saved"] is True
    assert client.get(f"/album/saved/{uid}/00000").json()["saved"] is False

    # 列表
    lst = client.get(f"/albums/saved/{uid}").json()["albums"]
    assert any(a["album_id"] == "18893" for a in lst)
    saved = next(a for a in lst if a["album_id"] == "18893")
    assert saved["tracks"][0]["title"] == "T1"
    assert saved["user_id"] == uid

    # 用户隔离：别的用户看不到
    other = client.get("/albums/saved/someone-else").json()["albums"]
    assert not any(a["album_id"] == "18893" for a in other)

    # 删除
    dele = client.delete(f"/album/saved/{uid}/18893")
    assert dele.status_code == 200
    assert dele.json()["deleted"] is True
    assert client.get(f"/album/saved/{uid}/18893").json()["saved"] is False
