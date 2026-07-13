from __future__ import annotations

from app.agent import CineSonicAgent
from app.models import MemoryUpdateRequest


def main() -> None:
    agent = CineSonicAgent()
    user_id = "demo-user"
    asset = agent.ingest_video("https://example.com/sony-cinematic-demo")
    agent.analyze_media(asset.asset_id)

    print(f"Asset: {asset.asset_id} / {asset.title}")
    print("\nRAG answer")
    answer = agent.answer_with_rag(
        asset_id=asset.asset_id,
        user_id=user_id,
        question="Find the 3 moments best suited for a cinematic trailer climax.",
        top_k=5,
    )
    print(answer.answer)

    print("\nUpdate memory")
    memory, changed = agent.update_memory(
        MemoryUpdateRequest(
            user_id=user_id,
            asset_id=asset.asset_id,
            segment_id=answer.recommended_segments[0].segment_id if answer.recommended_segments else None,
            event="I like fast-paced cinematic moments with strong music.",
        )
    )
    print({"changed": changed, "preferences": memory.preferences})

    print("\nMemory-aware recommendation")
    recommendation = agent.recommend_with_memory(
        asset_id=asset.asset_id,
        user_id=user_id,
        goal="Recommend trailer highlights for a Sony-style audiovisual pitch.",
        top_k=3,
    )
    print(recommendation.answer)


if __name__ == "__main__":
    main()
