from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.models import Asset
from app.sources import netease as netease_source

logger = logging.getLogger(__name__)


class CatalogService:
    def __init__(
        self,
        *,
        store: Any,
        enrich_asset: Any,
        fetch_video_title: Any,
        sync_recommend_artist_albums: Any,
        search_netease_detail: Any,
        search_bilibili_detail: Any,
        has_reliable_metadata: Any,
        generic_metadata_title: Any,
    ) -> None:
        self.store = store
        self._enrich_asset = enrich_asset
        self._fetch_video_title = fetch_video_title
        self._sync_recommend_artist_albums = sync_recommend_artist_albums
        self._search_netease_detail = search_netease_detail
        self._search_bilibili_detail = search_bilibili_detail
        self._has_reliable_metadata = has_reliable_metadata
        self._generic_metadata_title = generic_metadata_title

    def fetch_track_metadata(
        self,
        asset_id: str | None = None,
        url: str | None = None,
        use_network: bool = True,
    ) -> dict[str, Any]:
        if asset_id:
            asset = self.store.read_model("assets", asset_id, Asset)
            if asset is None:
                return {"found": False, "asset_id": asset_id, "error": "unknown asset"}
            if use_network:
                try:
                    enriched = self._enrich_asset(asset_id, use_network=True)
                    asset = enriched.asset
                except Exception as exc:
                    return {
                        "found": self._has_reliable_metadata(asset),
                        "asset_id": asset_id,
                        "title": asset.title,
                        "artist": asset.artist,
                        "source_url": asset.source_url,
                        "error": str(exc),
                    }
            found = self._has_reliable_metadata(asset)
            return {
                "found": found,
                "asset_id": asset.asset_id,
                "title": asset.title,
                "artist": asset.artist,
                "album": asset.album,
                "genre": asset.genre,
                "mood": asset.mood,
                "source_url": asset.source_url,
            }

        if url:
            title = self._fetch_video_title(url) if use_network else None
            return {
                "found": bool(title and not self._generic_metadata_title(title)),
                "url": url,
                "title": title,
                "mode": "online" if use_network else "offline",
            }

        return {"found": False, "error": "asset_id or url is required"}

    def recommend_artist_albums(self, user_id: str, artist: str, limit: int = 12) -> list[dict[str, Any]]:
        artist = (artist or "").strip()
        if not artist:
            return []
        try:
            return netease_source.search_netease_artist_albums(artist, limit)
        except Exception:
            logger.debug("recommend_artist_albums failed for %s", artist, exc_info=True)
            return []

    async def recommend_artist_albums_async(self, user_id: str, artist: str, limit: int = 12) -> list[dict[str, Any]]:
        artist = (artist or "").strip()
        if not artist:
            return []
        try:
            result = await netease_source.asearch_netease_artist_albums(artist, limit)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("async recommend_artist_albums failed for %s", artist, exc_info=True)
            return await asyncio.to_thread(
                self._sync_recommend_artist_albums,
                user_id=user_id,
                artist=artist,
                limit=limit,
            )
        if result:
            return result
        return await asyncio.to_thread(
            self._sync_recommend_artist_albums,
            user_id=user_id,
            artist=artist,
            limit=limit,
        )
