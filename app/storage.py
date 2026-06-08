from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class JsonStore:
    def __init__(self, root: Path | str = "data/store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

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
