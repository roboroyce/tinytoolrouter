import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import eval_harness as eh  # noqa: E402
from schema_utils import ToolSchemas, score_prediction  # noqa: E402


def _rows(name="test.jsonl", n=200):
    p = ROOT / "data" / name
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()[:n]]


class TestJsonScoring(unittest.TestCase):
    """Regression: ordinary JSON predictions must be PARSED, not called
    unparseable. This is the heuristic/evaluation bug that was fixed."""

    def setUp(self):
        self.s = eh.load_tools()
        self.gold = {"name": "get_weather", "arguments": {"city": "Oslo"}}

    def test_wellformed_json_counts_as_parsed(self):
        m = score_prediction('{"name": "get_weather", "arguments": {"city": "Oslo"}}',
                             self.gold, self.s)
        self.assertTrue(m["json_parsed"])
        self.assertTrue(m["normalized"])
        self.assertTrue(m["parse_ok"])
        self.assertTrue(m["exact_match"])

    def test_json_with_empty_args_is_parsed_but_not_schema_valid(self):
        """The heuristic backend's output: valid JSON, missing required args."""
        m = score_prediction('{"name": "get_weather", "arguments": {}}',
                             self.gold, self.s)
        self.assertTrue(m["json_parsed"], "must be recognised as JSON")
        self.assertTrue(m["normalized"])
        self.assertFalse(m["parse_ok"], "schema-invalid, but parsed")
        self.assertFalse(m["exact_match"])

    def test_prose_is_not_parsed(self):
        m = score_prediction("The weather in Oslo is mild.", self.gold, self.s)
        self.assertFalse(m["json_parsed"])
        self.assertFalse(m["parse_ok"])

    def test_no_tool_prediction_is_valid(self):
        gold = {"name": "no_tool", "arguments": {}}
        m = score_prediction('{"name": "no_tool", "arguments": {}}', gold, self.s)
        self.assertTrue(m["parse_ok"])
        self.assertTrue(m["exact_match"])
        self.assertTrue(m["pred_is_notool"])

    def test_no_tool_with_arguments_is_invalid(self):
        m = score_prediction('{"name": "no_tool", "arguments": {"x": 1}}',
                             {"name": "no_tool", "arguments": {}}, self.s)
        self.assertFalse(m["parse_ok"])

    def test_array_argument_typing(self):
        gold = {"name": "tag_note", "arguments": {"content": "x", "tags": ["a"]}}
        ok = score_prediction('{"name": "tag_note", "arguments": {"content": "x", "tags": ["a"]}}',
                             gold, self.s)
        bad = score_prediction('{"name": "tag_note", "arguments": {"content": "x", "tags": [1]}}',
                              gold, self.s)
        self.assertTrue(ok["parse_ok"])
        self.assertFalse(bad["parse_ok"])


class TestHarness(unittest.TestCase):
    def setUp(self):
        self.rows = _rows()
        if not self.rows:
            self.skipTest("data/test.jsonl not generated yet")

    def test_gold_backend_is_perfect(self):
        rep = eh.evaluate(self.rows, "gold")
        o = rep["overall"]
        self.assertEqual(o["exact_match_accuracy"], 1.0)
        self.assertEqual(o["valid_structured_output_rate"], 1.0)
        self.assertEqual(o["json_parse_rate"], 1.0)

    def test_heuristic_parses_but_scores_low(self):
        """The fix: heuristic emits JSON (parsed) but has no arguments
        (schema-invalid), so json_parse_rate >> valid_structured_output_rate."""
        rep = eh.evaluate(self.rows, "heuristic")
        o = rep["overall"]
        self.assertGreater(o["json_parse_rate"], 0.9)
        self.assertLess(o["valid_structured_output_rate"], o["json_parse_rate"])
        # Heuristic emits empty args, so it can only ever match the no_tool
        # rows by accident; exact match must stay near zero.
        self.assertLess(o["exact_match_accuracy"], 0.2)

    def test_families_present(self):
        rep = eh.evaluate(self.rows, "gold")
        self.assertIn("seen_paraphrase", rep["by_family"])
        self.assertIn("unseen_schema", rep["by_family"])
        self.assertIn("no_tool", rep["by_family"])
        self.assertEqual(rep["unseen_schema_exact_match"], 1.0)

    def test_metrics_in_range(self):
        rep = eh.evaluate(self.rows, "heuristic")
        for k in ("json_parse_rate", "valid_structured_output_rate",
                  "tool_selection_accuracy", "argument_accuracy", "exact_match_accuracy"):
            self.assertGreaterEqual(rep["overall"][k], 0.0)
            self.assertLessEqual(rep["overall"][k], 1.0)

    def test_unknown_backend_raises(self):
        with self.assertRaises(SystemExit):
            eh.evaluate(self.rows, "nope")


if __name__ == "__main__":
    unittest.main()