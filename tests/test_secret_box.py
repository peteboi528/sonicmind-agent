"""凭证静态加密（secret_box）+ netease_auth 加密落盘测试。"""
from __future__ import annotations

import json
import os
import stat

import pytest
from cryptography.fernet import Fernet

from app.config import settings
from app.security import secret_box


@pytest.fixture
def crypto_enabled(monkeypatch):
    """注入独立 Fernet 密钥启用加密，隔离真实库的 data/.secret_key。"""
    secret_box._fernet.cache_clear()
    monkeypatch.setenv("SECRET_STORE_KEY", Fernet.generate_key().decode())
    secret_box._fernet.cache_clear()  # setenv 后再清，确保下次 _fernet() 重算
    yield
    secret_box._fernet.cache_clear()


def test_roundtrip(crypto_enabled):
    assert secret_box.is_enabled()
    plain = "MUSIC_U=long_lived_session_token_123"
    token = secret_box.encrypt(plain)
    assert token != plain
    assert "MUSIC_U" not in token
    assert secret_box.decrypt(token) == plain


def test_disabled_passthrough(monkeypatch):
    monkeypatch.setattr(secret_box, "_fernet", lambda: None)
    assert not secret_box.is_enabled()
    assert secret_box.encrypt("plain") == "plain"
    assert secret_box.decrypt("plain") == "plain"


def test_decrypt_tolerates_plaintext_token(crypto_enabled):
    # 解密一个非法 token（明文旧值）应原样返回，不抛异常——兼容历史明文落盘
    assert secret_box.decrypt("not-a-fernet-token") == "not-a-fernet-token"


def test_migrate_encrypts_plaintext_files(crypto_enabled, tmp_path):
    auth_dir = tmp_path / "netease_auth"
    auth_dir.mkdir()
    legacy = auth_dir / "user1.json"
    legacy.write_text(json.dumps({"cookie": "MUSIC_U=plaintext_secret", "nickname": "u1"}))
    os.chmod(legacy, 0o644)

    result = secret_box.migrate_plaintext_cookies(str(tmp_path))
    assert result == {"scanned": 1, "migrated": 1, "skipped": 0}

    # 主文件已加密：明文消失、_encrypted=True、可解密还原
    payload = json.loads(legacy.read_text())
    assert payload["_encrypted"] is True
    assert "MUSIC_U=plaintext_secret" not in legacy.read_text()
    assert json.loads(secret_box.decrypt(payload["blob"]))["cookie"] == "MUSIC_U=plaintext_secret"
    # 权限收紧到 0600
    assert stat.S_IMODE(os.stat(legacy).st_mode) == 0o600
    # 不遗留明文备份
    backup = auth_dir / "user1.json.bak.plaintext"
    assert not backup.exists()
    # 幂等：已是 _encrypted 的不再动
    again = secret_box.migrate_plaintext_cookies(str(tmp_path))
    assert again["migrated"] == 0


def test_migrate_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(secret_box, "_fernet", lambda: None)
    auth_dir = tmp_path / "netease_auth"
    auth_dir.mkdir()
    (auth_dir / "u.json").write_text(json.dumps({"cookie": "x"}))
    # 未启用加密时 early-return，不扫不写
    assert secret_box.migrate_plaintext_cookies(str(tmp_path)) == {
        "scanned": 0, "migrated": 0, "skipped": 0,
    }
    assert json.loads((auth_dir / "u.json").read_text())["cookie"] == "x"


def test_netease_auth_save_load_roundtrip(crypto_enabled, monkeypatch, tmp_path):
    from app import netease_auth

    monkeypatch.setattr(settings, "store_root", str(tmp_path))
    netease_auth.save_cookie("alice", "MUSIC_U=abc; __csrf=def", nickname="Alice", vip_type=11)

    # 落盘文件不含明文 cookie
    raw = (tmp_path / "netease_auth" / "alice.json").read_text()
    assert "MUSIC_U=abc" not in raw
    # load 透明解密还原
    data = netease_auth.load_cookie("alice")
    assert data["cookie"] == "MUSIC_U=abc; __csrf=def"
    assert data["nickname"] == "Alice"
    assert data["vip_type"] == 11


def test_netease_auth_load_legacy_plaintext(crypto_enabled, monkeypatch, tmp_path):
    """新代码读取旧版明文文件（无 blob 字段）应兼容返回，不破坏存量。"""
    from app import netease_auth

    monkeypatch.setattr(settings, "store_root", str(tmp_path))
    auth_dir = tmp_path / "netease_auth"
    auth_dir.mkdir()
    (auth_dir / "legacy.json").write_text(json.dumps({"cookie": "MUSIC_U=old", "nickname": "L"}))

    data = netease_auth.load_cookie("legacy")
    assert data == {"cookie": "MUSIC_U=old", "nickname": "L"}
