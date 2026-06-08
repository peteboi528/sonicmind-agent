from __future__ import annotations

import os


# Tests must be deterministic and offline-first even when a developer has a
# real LLM key in .env. Runtime behavior is unchanged outside pytest.
os.environ["LLM_API_KEY"] = ""
os.environ.setdefault("LLM_TIMEOUT_SECONDS", "1")
