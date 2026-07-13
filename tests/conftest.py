from __future__ import annotations

import pytest

from tests.offline_fakes import (
    apply_pytest_monkeypatch,
    configure_offline_env,
    seed_random,
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="跑标了 @pytest.mark.network 的用例（真联网集成测试）",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: 依赖真实网络（web_search/musicbrainz/lastfm/discogs），默认 skip",
    )


# Tests must be deterministic even when production behavior is online-first and
# a developer has a real LLM key in .env. Runtime behavior is unchanged outside
# pytest.
configure_offline_env()


@pytest.fixture(autouse=True)
def fake_online_music_search(monkeypatch):
    apply_pytest_monkeypatch(monkeypatch)


@pytest.fixture(autouse=True)
def _reset_netease_album_cache():
    """专辑详情缓存是进程级全局的，测试间共享会互相污染（A 缓存的 "18893" 让 B 拿不到
    它打桩的 urlopen）。每个测试前清空，保证隔离。"""
    from app.sources.netease import clear_album_detail_cache

    clear_album_detail_cache()
    yield


@pytest.fixture(autouse=True)
def _reset_shared_executor():
    """共享线程池跨用例隔离：每用例后关停，避免上一用例超时滞留的 worker 线程漏到下一用例
    （离线 fakes 即时返回，关停近乎零开销；池对象下次按需重建，线程按需创建）。"""
    yield
    try:
        from app.concurrency import shutdown_shared_executor

        shutdown_shared_executor()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _seed_random():
    """全局 random 未 seed 是 flaky 总根因：library.sample_ts_scores 的 betavariate
    （TS 探索，enable_explore 默认开）和 MockLLM 的 random.choice 让推荐排序/mock 输出
    随用例顺序漂移——隔离跑过、合跑挂。每用例固定 seed，结果确定且与顺序无关。"""
    seed_random()


@pytest.fixture(autouse=True)
def _network_tests_opt_in(request, monkeypatch):
    """把真实网络挡在确定性套件之外：
    - 标了 @pytest.mark.network 的用例：没传 --run-network 就 skip；传了才放行打真网络。
    - 其余用例：把 web_search_info 兜底 mock 成空，避免任何意外真网络调用（如知识 agent
      的采样溯源）导致 flaky。用例自己 monkeypatch 的会覆盖这个默认。"""
    if request.node.get_closest_marker("network"):
        if not request.config.getoption("--run-network"):
            pytest.skip("network test (use --run-network)")
        return
    try:
        monkeypatch.setattr("app.knowledge.web_search_source.search_web_info", lambda *a, **k: [])
    except Exception:
        pass
