from __future__ import annotations

import hashlib

from app.models import ExternalTrack

CATALOG_DATA: list[tuple[str, str, str, list[str], list[str], int, float]] = [
    # (title, artist, album, genres, moods, bpm, energy)
    ("晴天", "周杰伦", "叶惠美", ["流行"], ["浪漫", "治愈"], 74, 0.5),
    ("七里香", "周杰伦", "七里香", ["流行"], ["浪漫"], 80, 0.45),
    ("夜曲", "周杰伦", "十一月的萧邦", ["流行"], ["伤感", "浪漫"], 72, 0.4),
    ("稻香", "周杰伦", "魔杰座", ["流行"], ["欢快", "治愈"], 100, 0.65),
    ("告白气球", "周杰伦", "周杰伦的床边故事", ["流行"], ["欢快", "浪漫"], 112, 0.6),
    ("富士山下", "陈奕迅", "What's Going On...?", ["流行"], ["伤感"], 68, 0.35),
    ("孤勇者", "陈奕迅", "孤勇者", ["流行", "摇滚"], ["激昂", "热血"], 128, 0.85),
    ("十年", "陈奕迅", "黑白灰", ["流行"], ["伤感"], 72, 0.35),
    ("浮夸", "陈奕迅", "U87", ["流行", "摇滚"], ["激昂"], 92, 0.8),
    ("江南", "林俊杰", "第二天堂", ["流行"], ["伤感", "浪漫"], 76, 0.4),
    ("她说", "林俊杰", "她说", ["流行"], ["伤感"], 70, 0.35),
    ("修炼爱情", "林俊杰", "因你而在", ["流行"], ["伤感"], 68, 0.4),
    ("光年之外", "邓紫棋", "光年之外", ["流行"], ["激昂", "浪漫"], 120, 0.75),
    ("泡沫", "邓紫棋", "Xposed", ["流行"], ["伤感"], 76, 0.4),
    ("句号", "邓紫棋", "摩天动物园", ["流行"], ["伤感"], 80, 0.45),
    ("消愁", "毛不易", "平凡的一天", ["民谣"], ["伤感", "忧郁"], 78, 0.3),
    ("像我这样的人", "毛不易", "平凡的一天", ["民谣"], ["忧郁"], 72, 0.25),
    ("南山南", "马頔", "南山南", ["民谣"], ["忧郁", "伤感"], 80, 0.3),
    ("成都", "赵雷", "无法长大", ["民谣"], ["放松", "浪漫"], 84, 0.4),
    ("平凡之路", "朴树", "猎户星座", ["摇滚", "民谣"], ["激昂", "治愈"], 96, 0.65),
]

CATALOG_DATA_2: list[tuple[str, str, str, list[str], list[str], int, float]] = [
    ("那些花儿", "朴树", "我去2000年", ["民谣"], ["伤感", "治愈"], 88, 0.4),
    ("起风了", "买辣椒也用券", "起风了", ["流行"], ["治愈", "激昂"], 92, 0.6),
    ("海阔天空", "Beyond", "乐与怒", ["摇滚"], ["激昂", "热血"], 132, 0.85),
    ("真的爱你", "Beyond", "Beyond IV", ["摇滚"], ["欢快", "治愈"], 120, 0.7),
    ("光辉岁月", "Beyond", "命运派对", ["摇滚"], ["激昂"], 128, 0.8),
    ("追梦赤子心", "GALA", "追梦痴子心", ["摇滚"], ["激昂", "热血"], 140, 0.9),
    ("New Boy", "朴树", "我去2000年", ["摇滚", "流行"], ["欢快"], 136, 0.75),
    ("Faded", "Alan Walker", "Different World", ["电子"], ["伤感", "梦幻"], 90, 0.6),
    ("Alone", "Alan Walker", "Different World", ["电子"], ["激昂", "梦幻"], 97, 0.7),
    ("The Spectre", "Alan Walker", "The Spectre", ["电子"], ["激昂", "热血"], 128, 0.8),
    ("Something Just Like This", "Coldplay", "A Head Full of Dreams", ["电子", "流行"], ["欢快"], 103, 0.65),
    ("Yellow", "Coldplay", "Parachutes", ["摇滚"], ["浪漫", "治愈"], 88, 0.5),
    ("Viva La Vida", "Coldplay", "Viva la Vida", ["摇滚", "流行"], ["激昂"], 138, 0.8),
    ("Shape of You", "Ed Sheeran", "÷", ["流行"], ["欢快"], 96, 0.7),
    ("Perfect", "Ed Sheeran", "÷", ["流行"], ["浪漫"], 63, 0.35),
    ("Blinding Lights", "The Weeknd", "After Hours", ["流行", "电子"], ["热血"], 171, 0.85),
    ("Starboy", "The Weeknd", "Starboy", ["R&B", "电子"], ["热血"], 186, 0.8),
    ("夜に駆ける", "YOASOBI", "THE BOOK", ["流行", "电子"], ["激昂", "梦幻"], 130, 0.8),
    ("アイドル", "YOASOBI", "THE BOOK 3", ["流行"], ["欢快", "热血"], 166, 0.9),
    ("Lemon", "米津玄师", "BOOTLEG", ["流行"], ["伤感"], 87, 0.45),
    ("打上花火", "米津玄师", "BOOTLEG", ["流行"], ["激昂", "浪漫"], 96, 0.65),
    ("Merry Christmas Mr. Lawrence", "坂本龙一", "音楽図鑑", ["古典"], ["宁静", "伤感"], 72, 0.2),
    ("Summer", "久石让", "菊次郎の夏", ["古典"], ["欢快", "治愈"], 120, 0.55),
    ("天空之城", "久石让", "天空の城ラピュタ", ["古典"], ["宁静", "梦幻"], 80, 0.3),
    ("River Flows in You", "Yiruma", "First Love", ["古典"], ["宁静", "浪漫"], 68, 0.25),
    ("克罗地亚狂想曲", "Maksim", "The Piano Player", ["古典"], ["激昂"], 160, 0.9),
    ("野蜂飞舞", "Maksim", "The Piano Player", ["古典"], ["激昂", "热血"], 180, 0.95),
    ("See You Again", "Wiz Khalifa", "Furious 7", ["说唱", "流行"], ["伤感", "治愈"], 80, 0.5),
    ("Lose Yourself", "Eminem", "8 Mile", ["说唱"], ["激昂", "热血"], 171, 0.9),
    ("Old Town Road", "Lil Nas X", "7", ["说唱", "流行"], ["欢快"], 136, 0.7),
    ("Sunflower", "Post Malone", "Spider-Man: Into the Spider-Verse", ["说唱", "流行"], ["欢快", "放松"], 90, 0.55),
    ("说好不哭", "周杰伦", "说好不哭", ["流行"], ["伤感"], 64, 0.3),
    ("Mojito", "周杰伦", "Mojito", ["流行"], ["欢快", "放松"], 105, 0.6),
    ("等你下课", "周杰伦", "等你下课", ["流行"], ["浪漫"], 76, 0.4),
    ("漠河舞厅", "柳爽", "漠河舞厅", ["民谣"], ["伤感", "忧郁"], 108, 0.35),
    ("错位时空", "艾辰", "错位时空", ["流行"], ["伤感"], 80, 0.4),
    ("孤独摇滚", "结束乐队", "孤独摇滚OST", ["摇滚"], ["激昂", "欢快"], 155, 0.85),
    ("春日影", "结束乐队", "孤独摇滚OST", ["摇滚"], ["激昂"], 148, 0.8),
    ("Bohemian Rhapsody", "Queen", "A Night at the Opera", ["摇滚"], ["激昂"], 72, 0.75),
    ("Hotel California", "Eagles", "Hotel California", ["摇滚"], ["放松"], 75, 0.5),
    ("Yesterday", "The Beatles", "Help!", ["流行", "民谣"], ["伤感"], 96, 0.3),
    ("Let It Be", "The Beatles", "Let It Be", ["摇滚", "流行"], ["治愈"], 76, 0.4),
    ("Fly Me to the Moon", "Frank Sinatra", "It Might as Well Be Swing", ["爵士"], ["浪漫", "放松"], 120, 0.45),
    ("Take Five", "Dave Brubeck", "Time Out", ["爵士"], ["放松"], 176, 0.5),
    ("So What", "Miles Davis", "Kind of Blue", ["爵士"], ["放松", "宁静"], 136, 0.4),
    ("Autumn Leaves", "Bill Evans", "Portrait in Jazz", ["爵士"], ["伤感", "宁静"], 100, 0.3),
]

# R&B / Soul 专区：用户曲库以英文 R&B 为主，单独补一批真实曲目，
# 让推荐在该方向有足够候选，也平衡 catalog 的语言分布（英文占比更高）。
CATALOG_RNB: list[tuple[str, str, str, list[str], list[str], int, float]] = [
    ("Blinding Lights", "The Weeknd", "After Hours", ["R&B", "流行"], ["热血", "浪漫"], 171, 0.85),
    ("Earned It", "The Weeknd", "Fifty Shades of Grey", ["R&B"], ["浪漫", "伤感"], 122, 0.5),
    ("Often", "The Weeknd", "Beauty Behind the Madness", ["R&B"], ["放松", "孤独"], 92, 0.45),
    ("Save Your Tears", "The Weeknd", "After Hours", ["R&B", "流行"], ["伤感", "欢快"], 118, 0.7),
    ("Adorn", "Miguel", "Kaleidoscope Dream", ["R&B"], ["浪漫", "放松"], 110, 0.55),
    ("Sure Thing", "Miguel", "All I Want Is You", ["R&B"], ["浪漫"], 80, 0.45),
    ("Best Part", "Daniel Caesar", "Freudian", ["R&B"], ["浪漫", "治愈"], 70, 0.35),
    ("Get You", "Daniel Caesar", "Freudian", ["R&B"], ["浪漫", "放松"], 110, 0.4),
    ("Nights", "Frank Ocean", "Blonde", ["R&B"], ["孤独", "放松"], 88, 0.5),
    ("Pink + White", "Frank Ocean", "Blonde", ["R&B"], ["治愈", "放松"], 80, 0.45),
    ("Thinkin Bout You", "Frank Ocean", "Channel Orange", ["R&B"], ["伤感", "浪漫"], 132, 0.4),
    ("Snooze", "SZA", "SOS", ["R&B"], ["浪漫", "放松"], 143, 0.5),
    ("Good Days", "SZA", "SOS", ["R&B"], ["治愈", "放松"], 121, 0.45),
    ("Kill Bill", "SZA", "SOS", ["R&B", "流行"], ["伤感", "欢快"], 89, 0.55),
    ("Location", "Khalid", "American Teen", ["R&B"], ["放松", "浪漫"], 89, 0.4),
    ("Redbone", "Childish Gambino", "Awaken, My Love!", ["R&B"], ["放松", "孤独"], 80, 0.45),
    ("Cranes in the Sky", "Solange", "A Seat at the Table", ["R&B"], ["伤感", "宁静"], 100, 0.35),
    ("Passionfruit", "Drake", "More Life", ["R&B", "流行"], ["放松", "浪漫"], 112, 0.5),
    ("Come Through and Chill", "Miguel", "War & Leisure", ["R&B"], ["放松", "浪漫"], 75, 0.4),
    ("Cranes", "Masego", "Lady Lady", ["R&B", "爵士"], ["放松", "欢快"], 95, 0.5),
    ("特别的人", "方大同", "未来", ["R&B"], ["浪漫", "治愈"], 96, 0.55),
    ("Love Song", "方大同", "爱爱爱", ["R&B"], ["浪漫"], 88, 0.5),
    ("小方", "陶喆", "太平盛世", ["R&B"], ["欢快", "治愈"], 100, 0.6),
    ("普通朋友", "陶喆", "I'm OK", ["R&B"], ["伤感"], 84, 0.45),
    ("爱很简单", "陶喆", "David Tao", ["R&B"], ["浪漫", "治愈"], 72, 0.4),
]


def _build_catalog() -> list[ExternalTrack]:
    all_data = CATALOG_DATA + CATALOG_DATA_2 + CATALOG_RNB
    catalog: list[ExternalTrack] = []
    for title, artist, album, genres, moods, bpm, energy in all_data:
        ext_id = hashlib.sha1(f"{title}-{artist}".encode()).hexdigest()[:10]
        catalog.append(
            ExternalTrack(
                external_id=ext_id,
                title=title,
                artist=artist,
                album=album,
                genre=genres,
                mood=moods,
                tempo_bpm=bpm,
                energy_level=energy,
                source="mock",
            )
        )
    return catalog


CATALOG = _build_catalog()


class MockSource:
    def __init__(self) -> None:
        self.catalog = CATALOG

    def search(self, query: str, limit: int = 20) -> list[ExternalTrack]:
        query_lower = query.lower()
        scored: list[tuple[int, ExternalTrack]] = []
        for track in self.catalog:
            score = 0
            searchable = f"{track.title} {track.artist} {' '.join(track.genre)} {' '.join(track.mood)}".lower()
            for term in query_lower.split():
                if term in searchable:
                    score += 1
            if score > 0:
                scored.append((score, track))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:limit]]

    def get_track(self, external_id: str) -> ExternalTrack | None:
        for track in self.catalog:
            if track.external_id == external_id:
                return track
        return None

    def get_recommendations(
        self, seed_genres: list[str], seed_moods: list[str], limit: int = 20
    ) -> list[ExternalTrack]:
        scored: list[tuple[float, ExternalTrack]] = []
        for track in self.catalog:
            genre_match = len(set(track.genre) & set(seed_genres)) / max(len(seed_genres), 1)
            mood_match = len(set(track.mood) & set(seed_moods)) / max(len(seed_moods), 1)
            score = 0.6 * genre_match + 0.4 * mood_match
            if score > 0:
                scored.append((score, track))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:limit]]
