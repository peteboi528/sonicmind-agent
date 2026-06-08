# SONICMIND Agent

SONICMIND Agent is an interview-focused music and audiovisual recommendation project. It keeps a product-shaped demo surface, but the core is now an explainable pure-Python agent:

`ingest -> analyze -> memory update -> RAG evidence retrieval -> tool selection -> recommendation / search / playlist / taste analysis`

## What It Demonstrates

- `Memory`: explicit preferences, listening history, ratings, and a derived taste profile.
- `RAG`: segment-level text / vision / audio / summary evidence retrieval.
- `Agent orchestration`: a ReAct-style loop routes chat requests into tools instead of sending everything straight to the LLM.
- `Offline-first execution`: mock source, mock LLM, and demo analyzer keep the default demo stable.
- `Explainability`: recommendations and chat answers expose trace steps and evidence chunks.

## Quick Start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
uvicorn app.api.main:app --reload --port 8000
```

Optional UI:

```bash
streamlit run app/ui/streamlit_app.py --server.port 8501
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- UI: `http://127.0.0.1:8501`

## Main API

- `POST /assets/ingest`
- `POST /assets/{asset_id}/enrich`
- `POST /assets/{asset_id}/analyze`
- `POST /recommend/daily`
- `POST /search`
- `POST /chat`
- `POST /memory/update`
- `POST /listen`
- `POST /rate`
- `POST /playlist/generate`

## Offline-First Behavior

- Ingest does not block on network metadata lookup by default.
- Online title or metadata enrichment is optional through `POST /assets/{asset_id}/enrich`.
- Search, recommendation, chat, and playlist generation all have non-network fallback paths.

## Demo Story

1. Add a few music or video links into the library.
2. Teach the agent a preference such as `我喜欢电子音乐和放松的氛围`.
3. Generate daily recommendations and inspect the trace.
4. Search for a style and inspect evidence chunks.
5. Ask the chat agent to analyze your taste or create a playlist.

## Notes

Read [docs/EXPLAINER.md](/Users/peteboi/Documents/MusicAgent/docs/EXPLAINER.md) for the interview-facing architecture explanation.
