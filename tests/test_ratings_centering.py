"""评分相对中心化：选择偏差下的评分膨胀治理（compute_taste_profile 的 ratings 路径）。

用户只入库喜欢的歌 → 评分天然偏高（实测 10×123/8×36/6×2）。旧公式 (score-5)*2 让所有评分
都成正激励、零梯度。改用相对自己均值的 z-score 后：基准分≈0（"喜欢"由入库表达）、低于常态
的真负信号、全员同分自动归零。
"""

from app.models import Asset, AssetStatus, RatingEntry
from app.recommend.engine import compute_taste_profile


def _asset(aid, genre):
    return Asset(
        asset_id=aid,
        source_url=f"x/{aid}",
        title=aid,
        duration_seconds=200,
        status=AssetStatus.ANALYZED,
        genre=genre,
        mood=["放松"],
    )


def _rating(aid, score, genre):
    return RatingEntry(asset_id=aid, score=score, title=aid, artist="", genre=genre, mood=["放松"])


def _weight_of(taste, genre):
    for g, w in taste.top_genres:
        if g == genre:
            return w
    return 0.0


def test_uniform_top_ratings_carry_no_gradient():
    # 两首不同 genre 都评 10：相对均值无偏差 → 评分不再额外放大任一 genre，只剩入库 base
    assets = [_asset("a", ["流行"]), _asset("b", ["摇滚"])]
    ratings = [_rating("a", 10, ["流行"]), _rating("b", 10, ["摇滚"])]
    taste = compute_taste_profile(assets, [], ratings)
    assert abs(_weight_of(taste, "流行") - _weight_of(taste, "摇滚")) < 0.1


def test_below_norm_rating_damps_genre():
    # 流行评 10（基准），摇滚评 6（远低于均值 8）→ 摇滚被压低甚至移出 top_genres
    assets = [_asset("a", ["流行"]), _asset("b", ["摇滚"])]
    ratings = [_rating("a", 10, ["流行"]), _rating("b", 6, ["摇滚"])]
    taste = compute_taste_profile(assets, [], ratings)
    assert _weight_of(taste, "流行") > _weight_of(taste, "摇滚")


def test_all_identical_scores_yield_zero_rating_weight():
    # 全员同分 → std=0 → 评分权重统一归零（诚实：相同评分无区分信息），genre 只剩入库 base
    assets = [_asset("a", ["流行"]), _asset("b", ["流行"])]
    ratings = [_rating("a", 8, ["流行"]), _rating("b", 8, ["流行"])]
    taste = compute_taste_profile(assets, [], ratings)
    # 流行 = 2 首入库 base(1+1) + 评分归零 = 2.0
    assert abs(_weight_of(taste, "流行") - 2.0) < 0.1


def test_above_norm_rating_still_lifts_within_inflated_library():
    # 均值 9 的高分库里：10（基准,+1.6）轻抬、8（低于常态,负）压低 → 仍有相对梯度
    assets = [_asset("a", ["流行"]), _asset("b", ["摇滚"])]
    ratings = [_rating("a", 10, ["流行"]), _rating("b", 8, ["摇滚"])]  # 均值 9
    taste = compute_taste_profile(assets, [], ratings)
    assert _weight_of(taste, "流行") > _weight_of(taste, "摇滚")
