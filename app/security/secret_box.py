"""凭证静态加密（Fernet）。

网易云 MUSIC_U cookie 等长效会话凭证落盘前用 Fernet 加密，避免明文 JSON 被备份
/日志/误提交泄漏（实测历史文件甚至是 0644 全机可读，比代码试图设的 0600 更糟）。
密钥来自 SECRET_STORE_KEY 环境变量；未配置时本地 demo 自动生成并落 data/.secret_key
（仅 dev 兜底，已 gitignore；生产必须显式配置，启动时告警）。

设计取舍：选 Fernet（cryptography 高层 API）而非复用飞书的 AES-CBC——同库不同层，
密钥管理更简单（一行生成）、自带 IV/时间戳/认证，是静态凭证加密的业界标准。
密钥不可得时降级为明文，向后兼容旧文件。

落盘格式统一为 ``{"_encrypted": bool, "blob": str}``，blob 为 JSON 串的加密/明文
token；旧版 ``{"cookie": ...}`` 明文格式（无 blob 字段）由 load 兼容读取。
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fernet() -> Fernet | None:
    """懒加载 Fernet 实例。

    密钥来源优先级：SECRET_STORE_KEY 环境变量 > data/.secret_key（dev 兜底）。
    都不可得时返回 None（降级明文）。懒加载以避免 import 时写盘副作用。
    """
    raw = os.getenv("SECRET_STORE_KEY", "").strip()
    if not raw:
        # dev 兜底：未配置时自动生成一份密钥落盘（data/.secret_key，已 gitignore）。
        # 生产部署必须显式配 SECRET_STORE_KEY，否则凭证等于裸存。
        store_root = Path(os.getenv("STORE_ROOT", "data/store"))
        key_file = store_root.parent / ".secret_key"
        try:
            if key_file.exists():
                raw = key_file.read_text(encoding="utf-8").strip()
            else:
                raw = Fernet.generate_key().decode()
                key_file.parent.mkdir(parents=True, exist_ok=True)
                key_file.write_text(raw, encoding="utf-8")
                try:
                    key_file.chmod(0o600)
                except OSError:
                    pass
                logger.warning(
                    "SECRET_STORE_KEY 未配置，已生成 dev 兜底密钥落 %s。生产部署必须显式设置 "
                    "SECRET_STORE_KEY 环境变量，否则加密等于裸存。", key_file,
                )
        except OSError:
            logger.warning("无法写入 dev 兜底密钥，凭证将明文落盘。生产必须配置 SECRET_STORE_KEY。")
            return None
    try:
        return Fernet(raw.encode())
    except (ValueError, TypeError):
        logger.error("SECRET_STORE_KEY 格式非法（应为 Fernet base64 密钥），凭证将明文落盘。")
        return None


def is_enabled() -> bool:
    """是否启用加密（密钥已配置或 dev 兜底成功）。"""
    return _fernet() is not None


def encrypt(plaintext: str) -> str:
    """加密字符串，返回 base64 token；未启用时原样返回（降级明文）。"""
    f = _fernet()
    if f is None:
        return plaintext
    return f.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    """解密 base64 token；未启用或 token 非法（明文旧值）时原样返回，由调用方兼容处理。"""
    f = _fernet()
    if f is None:
        return token
    try:
        return f.decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError):
        return token


def _needs_migration(data: object) -> bool:
    """该已解析的 JSON 对象是否需要重写为当前加密格式。"""
    if not isinstance(data, dict):
        return False
    if "blob" not in data:
        return True  # 旧版明文 {"cookie": ...}
    if data.get("_encrypted"):
        return False  # 已加密
    return is_enabled()  # 明文 blob 且当前可加密 → 升级


def migrate_plaintext_cookies(store_root: str) -> dict[str, int]:
    """扫描 ``{store_root}/netease_auth/*.json``，把旧明文 / 降级明文重写为加密格式。

    幂等：已是 ``_encrypted=true`` 的跳过。重写时备份原文件为 ``{name}.bak.plaintext``
    并以原子替换写回，避免留下长期明文备份。返回 ``{scanned, migrated, skipped}``。
    即使 is_enabled() 为 False 也安全运行（此时 _needs_migration 仅对旧格式返回 True，
    但重写后仍是明文 blob，等价不动——故直接 early-return 省去无谓 IO）。
    """
    auth_dir = Path(store_root) / "netease_auth"
    if not auth_dir.is_dir() or not is_enabled():
        return {"scanned": 0, "migrated": 0, "skipped": 0}
    scanned = migrated = skipped = 0
    for p in sorted(auth_dir.glob("*.json")):
        scanned += 1
        try:
            raw = p.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            logger.debug("跳过无法解析的 cookie 文件：%s", p, exc_info=True)
            skipped += 1
            continue
        if not _needs_migration(data):
            skipped += 1
            continue
        # 提取明文 JSON 串：旧格式整个 dict 序列化；新降级格式取 blob 字段。
        inner = data["blob"] if "blob" in data else json.dumps(data, ensure_ascii=False)
        new_payload = {"_encrypted": True, "blob": encrypt(inner)}
        try:
            # 先写入同目录临时文件再原子替换，既避免中断时损坏主文件，也不在磁盘上
            # 长期保留含 MUSIC_U 的明文副本。
            tmp = p.with_name(f".{p.name}.tmp")
            tmp.write_text(json.dumps(new_payload, ensure_ascii=False), encoding="utf-8")
            try:
                tmp.chmod(0o600)
            except OSError:
                pass
            os.replace(tmp, p)
            migrated += 1
            logger.warning("已加密明文 cookie 文件 %s。", p.name)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            logger.debug("加密迁移失败：%s", p, exc_info=True)
            skipped += 1
    return {"scanned": scanned, "migrated": migrated, "skipped": skipped}
