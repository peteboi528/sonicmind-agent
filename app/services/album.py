"""AlbumService —— 用户收藏专辑的 CRUD。

从 `AudioVisualAgent` 抽离：save_album / list_saved_albums / delete_saved_album /
is_album_saved。纯 store 读写 + 保存/删除后刷新品味档案。依赖 store/memory +
list_assets 回调；agent 保留同名薄委托。
"""

from __future__ import annotations

import logging
from typing import Callable

from app.memory import MemoryManager
from app.models import Asset, SavedAlbum
from app.storage import JsonStore

logger = logging.getLogger(__name__)


class AlbumService:
    def __init__(
        self,
        store: JsonStore,
        memory: MemoryManager,
        *,
        list_assets: Callable[[], list[Asset]],
    ) -> None:
        self.store = store
        self.memory = memory
        self._list_assets = list_assets

    def save_album(self, user_id: str, album: SavedAlbum) -> SavedAlbum:
        self.store.write_model("saved_albums", f"{user_id}_{album.album_id}", album)
        try:
            self.memory.refresh_taste_profile(user_id, self._list_assets())
        except Exception:
            logger.debug("refresh_taste_profile failed after save_album(%s)", album.album_id, exc_info=True)
        return album

    def list_saved_albums(self, user_id: str) -> list[SavedAlbum]:
        albums: list[SavedAlbum] = []
        for key in self.store.list_keys("saved_albums"):
            if not key.startswith(f"{user_id}_"):
                continue
            try:
                a = self.store.read_model("saved_albums", key, SavedAlbum)
            except Exception:
                logger.warning("Skipping unreadable saved album %s (stale schema?)", key, exc_info=True)
                continue
            if a:
                albums.append(a)
        return albums

    def delete_saved_album(self, user_id: str, album_id: str) -> bool:
        deleted = self.store.delete_key("saved_albums", f"{user_id}_{album_id}")
        if deleted:
            try:
                self.memory.refresh_taste_profile(user_id, self._list_assets())
            except Exception:
                logger.debug("refresh_taste_profile failed after delete_saved_album(%s)", album_id, exc_info=True)
        return deleted

    def is_album_saved(self, user_id: str, album_id: str) -> bool:
        return self.store.read_model("saved_albums", f"{user_id}_{album_id}", SavedAlbum) is not None
