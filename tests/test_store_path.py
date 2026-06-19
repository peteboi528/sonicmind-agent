from pathlib import Path

from app.config import PROJECT_ROOT, Settings


def test_relative_store_path_is_project_relative_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORE_ROOT", "isolated/store")
    monkeypatch.delenv("MEDIA_ROOT", raising=False)
    monkeypatch.delenv("RESOURCE_LIBRARY_PATH", raising=False)

    settings = Settings()

    assert Path(settings.store_root) == (PROJECT_ROOT / "isolated/store").resolve()
    assert Path(settings.resource_library_path) == (PROJECT_ROOT / "isolated/resource_library.sqlite").resolve()
