import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # harness reads data/tools.json relative to cwd

from schema_utils import (  # noqa: E402
    ToolSchemas, extract_json, normalize_call, score_prediction,
    validate_call, validate_schema,
)


def _schemas() -> ToolSchemas:
    return ToolSchemas(json.loads((ROOT / "data" / "tools.json").read_text()))


class TestValidateSchema(unittest.TestCase):
    def test_good_schema(self):
        s = {"name": "f", "parameters": {"type": "object",
             "properties": {"a": {"type": "string"}}, "required": ["a"]}}
        self.assertEqual(validate_schema(s), [])

    def test_bad_schemas(self):
        self.assertTrue(validate_schema({"parameters": {}}))
        self.assertTrue(validate_schema({"name": "f", "parameters": {"type": "object"}}))
        self.assertTrue(validate_schema(
            {"name": "f", "parameters": {"type": "object",
             "properties": {"a": {"type": "weird"}}}}))
        self.assertTrue(validate_schema(
            {"name": "f", "parameters": {"type": "object",
             "properties": {"a": {"type": "string"}}, "required": ["b"]}}))


class TestExtractAndNormalize(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_embedded_json(self):
        self.assertEqual(extract_json('Sure! {"a": 1} done'), {"a": 1})

    def test_garbage(self):
        self.assertIsNone(extract_json("no json here"))
        self.assertIsNone(extract_json(""))

    def test_normalize_shapes(self):
        self.assertEqual(normalize_call({"name": "x", "arguments": {"a": 1}}),
                         {"name": "x", "arguments": {"a": 1}})
        self.assertEqual(normalize_call({"tool": "x", "args": {"a": 1}}),
                         {"name": "x", "arguments": {"a": 1}})
        self.assertEqual(normalize_call({"name": "x", "arguments": '{"a": 1}'}),
                         {"name": "x", "arguments": {"a": 1}})
        self.assertEqual(normalize_call(
            {"function": {"name": "x", "arguments": '{"a": 1}'}}),
            {"name": "x", "arguments": {"a": 1}})
        self.assertIsNone(normalize_call("not a dict"))
        self.assertIsNone(normalize_call({"arguments": {}}))


class TestValidateCall(unittest.TestCase):
    def setUp(self):
        self.s = _schemas()

    def test_ok(self):
        self.assertEqual(validate_call({"name": "get_weather",
                                        "arguments": {"city": "Oslo"}}, self.s), [])

    def test_missing_required(self):
        self.assertTrue(validate_call({"name": "get_weather", "arguments": {}}, self.s))

    def test_unknown_tool(self):
        self.assertTrue(validate_call({"name": "nope", "arguments": {}}, self.s))

    def test_type_mismatch(self):
        self.assertTrue(validate_call(
            {"name": "set_timer", "arguments": {"seconds": "five"}}, self.s))

    def test_bool_is_not_int(self):
        self.assertTrue(validate_call(
            {"name": "set_timer", "arguments": {"seconds": True}}, self.s))


class TestScorePrediction(unittest.TestCase):
    def setUp(self):
        self.s = _schemas()
        self.gold = {"name": "get_weather", "arguments": {"city": "Oslo"}}

    def test_exact(self):
        m = score_prediction('{"name": "get_weather", "arguments": {"city": "oslo"}}',
                             self.gold, self.s)
        for k in ("json_parsed", "normalized", "parse_ok", "tool_correct",
                  "args_correct", "exact_match"):
            self.assertTrue(m[k], msg=k)

    def test_wrong_tool(self):
        m = score_prediction('{"name": "search_web", "arguments": {"query": "Oslo"}}',
                             self.gold, self.s)
        self.assertTrue(m["parse_ok"])
        self.assertFalse(m["tool_correct"])
        self.assertFalse(m["exact_match"])

    def test_wrong_args(self):
        m = score_prediction('{"name": "get_weather", "arguments": {"city": "Lima"}}',
                             self.gold, self.s)
        self.assertTrue(m["tool_correct"])
        self.assertFalse(m["args_correct"])

    def test_unparseable(self):
        m = score_prediction("I think the weather is nice.", self.gold, self.s)
        self.assertFalse(m["parse_ok"])
        self.assertFalse(m["exact_match"])


if __name__ == "__main__":
    unittest.main()