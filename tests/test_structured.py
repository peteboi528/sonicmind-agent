"""structured.py JSON 提取器测试：覆盖嵌套、字符串内括号、多片段、fenced block。"""
from __future__ import annotations

from app.llm.structured import extract_json_dict, extract_json_list


class TestExtractJsonDict:
    def test_plain_object(self):
        assert extract_json_dict('{"intent": "recommend"}') == {"intent": "recommend"}

    def test_object_with_prose_around(self):
        text = '好的，这是规划：{"intent": "search", "n": 3} 以上。'
        assert extract_json_dict(text) == {"intent": "search", "n": 3}

    def test_nested_object(self):
        text = '{"intent": "playlist", "plan": {"use_web": true, "tags": {"mood": "chill"}}}'
        assert extract_json_dict(text) == {
            "intent": "playlist",
            "plan": {"use_web": True, "tags": {"mood": "chill"}},
        }

    def test_object_containing_array(self):
        text = '{"entities": ["周杰伦", "林俊杰"], "n": 2}'
        assert extract_json_dict(text) == {"entities": ["周杰伦", "林俊杰"], "n": 2}

    def test_braces_inside_string_value(self):
        # 字符串内的 } 不能提前收口
        text = '{"reasoning": "用户说 {不要} 抖音", "intent": "recommend"}'
        assert extract_json_dict(text) == {
            "reasoning": "用户说 {不要} 抖音",
            "intent": "recommend",
        }

    def test_escaped_quote_inside_string(self):
        text = r'{"note": "他说\"很好\"", "ok": true}'
        assert extract_json_dict(text) == {"note": '他说"很好"', "ok": True}

    def test_fenced_json_block(self):
        text = '说明如下：\n```json\n{"intent": "discuss", "n": 5}\n```\n完毕'
        assert extract_json_dict(text) == {"intent": "discuss", "n": 5}

    def test_fenced_block_preferred_over_trailing_garbage(self):
        # fenced 内是合法对象，围栏后还有杂散括号，不应污染
        text = '```json\n{"intent": "taste"}\n```\n后面还有 } 杂散字符 {'
        assert extract_json_dict(text) == {"intent": "taste"}

    def test_first_of_multiple_objects(self):
        text = '{"intent": "search"} 然后 {"intent": "recommend"}'
        # 深度扫描在第一个配平处收口，不会贪婪吞到第二个
        assert extract_json_dict(text) == {"intent": "search"}

    def test_no_object_returns_none(self):
        assert extract_json_dict("没有任何 JSON 内容") is None

    def test_array_input_returns_none(self):
        # extract_json_dict 只认对象
        assert extract_json_dict('[1, 2, 3]') is None


class TestExtractJsonList:
    def test_plain_array(self):
        assert extract_json_list('[1, 2, 3]') == [1, 2, 3]

    def test_array_of_objects(self):
        text = '[{"title": "A"}, {"title": "B"}]'
        assert extract_json_list(text) == [{"title": "A"}, {"title": "B"}]

    def test_array_with_nested_brackets_in_string(self):
        text = '["含 [括号] 的标题", "正常"]'
        assert extract_json_list(text) == ["含 [括号] 的标题", "正常"]

    def test_fenced_array(self):
        text = '候选：\n```json\n[{"id": 1}, {"id": 2}]\n```'
        assert extract_json_list(text) == [{"id": 1}, {"id": 2}]

    def test_array_with_prose_around(self):
        text = '结果是 [10, 20] 这些。'
        assert extract_json_list(text) == [10, 20]

    def test_no_array_returns_none(self):
        assert extract_json_list("没有数组") is None
