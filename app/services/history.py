from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.models import (
    ChatHistoryMessage,
    ChatHistoryThread,
    RecommendationHistoryItem,
    utc_now_iso,
)
from app.storage import JsonStore

CHAT_COLLECTION = "chat_history"
RECOMMENDATION_COLLECTION = "recommendation_history_full"


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _title_from_message(message: str) -> str:
    compact = " ".join((message or "").split())
    return compact[:28] or "新的对话"


class HistoryService:
    """Persist UI-facing chat and recommendation history.

    DialogueState remains the lightweight continuation state for the agent loop;
    this service stores what the user actually saw so the Web demo survives a
    backend/frontend restart.
    """

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def list_chat_threads(self, user_id: str) -> list[ChatHistoryThread]:
        threads = self.store.read_models(CHAT_COLLECTION, user_id, ChatHistoryThread)
        return sorted(threads, key=lambda item: item.updated_at, reverse=True)

    def get_chat_thread(self, user_id: str, thread_id: str) -> ChatHistoryThread | None:
        for thread in self.list_chat_threads(user_id):
            if thread.thread_id == thread_id:
                return thread
        return None

    def append_chat_turn(
        self,
        user_id: str,
        thread_id: str,
        user_message: str,
        assistant_message: str = "",
        cards: list[dict] | None = None,
        trace_summary: dict | None = None,
    ) -> ChatHistoryThread:
        with self.store.lock(CHAT_COLLECTION, user_id):
            threads = self.store.read_models(CHAT_COLLECTION, user_id, ChatHistoryThread)
            by_id = {thread.thread_id: thread for thread in threads}
            now = utc_now_iso()
            thread = by_id.get(thread_id)
            if thread is None:
                thread = ChatHistoryThread(
                    thread_id=thread_id,
                    user_id=user_id,
                    title=_title_from_message(user_message),
                    created_at=now,
                    updated_at=now,
                )
                threads.append(thread)

            if user_message:
                thread.messages.append(ChatHistoryMessage(role="user", content=user_message, created_at=now))
            if assistant_message:
                thread.messages.append(
                    ChatHistoryMessage(
                        role="assistant",
                        content=assistant_message,
                        created_at=now,
                        cards=list(cards or []),
                        trace_summary=dict(trace_summary or {}),
                    )
                )
            thread.messages = thread.messages[-settings.chat_history_max_messages_per_thread :]
            thread.updated_at = now
            if not thread.title:
                thread.title = _title_from_message(user_message)

            kept = sorted(threads, key=lambda item: item.updated_at, reverse=True)[: settings.chat_history_max_threads]
            self.store.write_models(CHAT_COLLECTION, user_id, kept)
            return thread

    def delete_chat_thread(self, user_id: str, thread_id: str) -> bool:
        with self.store.lock(CHAT_COLLECTION, user_id):
            threads = self.store.read_models(CHAT_COLLECTION, user_id, ChatHistoryThread)
            kept = [thread for thread in threads if thread.thread_id != thread_id]
            self.store.write_models(CHAT_COLLECTION, user_id, kept)
            return len(kept) != len(threads)

    def clear_chat_threads(self, user_id: str) -> int:
        with self.store.lock(CHAT_COLLECTION, user_id):
            threads = self.store.read_models(CHAT_COLLECTION, user_id, ChatHistoryThread)
            self.store.write_models(CHAT_COLLECTION, user_id, [])
            return len(threads)

    def list_recommendations(self, user_id: str) -> list[RecommendationHistoryItem]:
        with self.store.lock(RECOMMENDATION_COLLECTION, user_id):
            items = self.store.read_models(RECOMMENDATION_COLLECTION, user_id, RecommendationHistoryItem)
            now = datetime.now(UTC)
            active = [item for item in items if _parse_iso(item.expires_at) > now]
            if len(active) != len(items):
                self.store.write_models(RECOMMENDATION_COLLECTION, user_id, active)
            return sorted(active, key=lambda item: item.created_at, reverse=True)

    def save_recommendation(
        self,
        user_id: str,
        query: str,
        answer: str = "",
        cards: list[dict] | None = None,
        thread_id: str = "",
        ttl_days: int | None = None,
    ) -> RecommendationHistoryItem:
        ttl = ttl_days or settings.recommendation_history_ttl_days
        now = datetime.now(UTC)
        item = RecommendationHistoryItem(
            record_id=uuid.uuid4().hex,
            user_id=user_id,
            thread_id=thread_id,
            query=query,
            answer=answer,
            cards=list(cards or []),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=ttl)).isoformat(),
        )
        with self.store.lock(RECOMMENDATION_COLLECTION, user_id):
            items = self.store.read_models(RECOMMENDATION_COLLECTION, user_id, RecommendationHistoryItem)
            active = [old for old in items if _parse_iso(old.expires_at) > now]
            active.append(item)
            active = sorted(active, key=lambda old: old.created_at, reverse=True)[
                : settings.recommendation_history_max_items
            ]
            self.store.write_models(RECOMMENDATION_COLLECTION, user_id, active)
        return item

    def delete_recommendation(self, user_id: str, record_id: str) -> bool:
        with self.store.lock(RECOMMENDATION_COLLECTION, user_id):
            items = self.store.read_models(RECOMMENDATION_COLLECTION, user_id, RecommendationHistoryItem)
            kept = [item for item in items if item.record_id != record_id]
            self.store.write_models(RECOMMENDATION_COLLECTION, user_id, kept)
            return len(kept) != len(items)
