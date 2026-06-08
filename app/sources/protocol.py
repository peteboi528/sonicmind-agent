from __future__ import annotations

from typing import Protocol

from app.models import ExternalTrack


class ExternalSource(Protocol):
    def search(self, query: str, limit: int = 20) -> list[ExternalTrack]: ...
    def get_track(self, external_id: str) -> ExternalTrack | None: ...
    def get_recommendations(self, seed_genres: list[str], seed_moods: list[str], limit: int = 20) -> list[ExternalTrack]: ...
