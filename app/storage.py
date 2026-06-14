from __future__ import annotations

import fcntl
import json
import os
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonStore:
    def __init__(self, root: Path | str = "data/store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # 进程内 per-key 锁：FastAPI 同步端点跑在线程池里，多线程并发 RMW 会丢更新。
        self._thread_locks: dict[str, threading.Lock] = {}
        self._thread_locks_guard = threading.Lock()

    def _thread_lock_for(self, lock_key: str) -> threading.Lock:
        with self._thread_locks_guard:
            lock = self._thread_locks.get(lock_key)
            if lock is None:
                lock = threading.Lock()
                self._thread_locks[lock_key] = lock
            return lock

    @contextmanager
    def lock(self, collection: str, key: str) -> Iterator[None]:
        """包住读-改-写临界区的 per-(collection,key) 锁，防并发丢更新。

        双重锁：进程内 threading.Lock（FastAPI 线程池并发）+ 跨进程 fcntl.flock
        （多 uvicorn worker）。非 POSIX / fcntl 不可用时退化为仅进程内线程锁。
        单次 write_model 仍由 os.replace 保证原子，本锁只防 read-modify-write 竞态。

        用法：with store.lock("memory", user_id): memory=read; modify; write
        """
        tlock = self._thread_lock_for(f"{collection}/{key}")
        tlock.acquire()
        lock_path = self._path(collection, key).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+")
        try:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except (OSError, AttributeError):
                pass  # 非 POSIX 或不支持，退化为进程内线程锁
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except (OSError, AttributeError):
                pass
            fh.close()
            tlock.release()

    def read_model(self, collection: str, key: str, model: type[T]) -> T | None:
        path = self._path(collection, key)
        if not path.exists():
            return None
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    def write_model(self, collection: str, key: str, value: BaseModel) -> None:
        path = self._path(collection, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, value.model_dump_json(indent=2))

    def read_models(self, collection: str, key: str, model: type[T]) -> list[T]:
        path = self._path(collection, key)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [model.model_validate(item) for item in raw]

    def write_models(self, collection: str, key: str, values: list[BaseModel]) -> None:
        path = self._path(collection, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = [value.model_dump(mode="json") for value in values]
        self._atomic_write(path, json.dumps(raw, ensure_ascii=False, indent=2))

    def delete_key(self, collection: str, key: str) -> bool:
        path = self._path(collection, key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def clear_collection(self, collection: str) -> int:
        collection_dir = self.root / collection
        if not collection_dir.exists():
            return 0
        count = len(list(collection_dir.glob("*.json")))
        shutil.rmtree(collection_dir)
        collection_dir.mkdir(parents=True, exist_ok=True)
        return count

    def list_keys(self, collection: str) -> list[str]:
        collection_dir = self.root / collection
        if not collection_dir.exists():
            return []
        return sorted(p.stem for p in collection_dir.glob("*.json"))

    def _path(self, collection: str, key: str) -> Path:
        safe_key = key.replace("/", "_")
        return self.root / collection / f"{safe_key}.json"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
