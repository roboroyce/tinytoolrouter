import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import gen_dataset as gd  # noqa: E402
from schema_utils import ToolSchemas, validate_schema  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_every_schema_valid(self):
        for t in gd.ALL_TOOLS:
            self.assertEqual(validate_schema(t), [], msg=t["name"])

    def test_seen_and_unseen_disjoint(self):
        s = {t["name"] for t in gd.SEEN_TOOLS}
        u = {t["name"] for t in gd.UNSEEN_TOOLS}
        self.assertFalse(s & u)

    def test_type_coverage(self):
        types = set()
        for t in gd.SEEN_TOOLS:
            for pspec in t["parameters"]["properties"].values():
                types.add(pspec["type"])
        self.assertLessEqual({"string", "integer", "number", "boolean", "array"}, types)

    def test_arg_order_covers_tools(self):
        for t in gd.ALL_TOOLS:
            self.assertIn(t["name"], gd.ARG_ORDER)


class TestTemplateIsolation(unittest.TestCase):
    def _template_set(self, rows):
        return {t for t, _ in rows}

    def test_train_vs_heldout_templates_disjoint(self):
        train = self._template_set(gd.TRAIN_TEMPLATES)
        held = self._template_set(gd.TEST_SEEN_TEMPLATES) | \
            self._template_set(gd.TEST_UNSEEN_TEMPLATES)
        self.assertFalse(train & held)

    def test_seen_vs_unseen_templates_disjoint(self):
        self.assertFalse(self._template_set(gd.TEST_SEEN_TEMPLATES) &
                         self._template_set(gd.TEST_UNSEEN_TEMPLATES))

    def test_pools_disjoint(self):
        # Boolean pools necessarily share values; compare non-boolean values.
        def norm(vals):
            out = set()
            for v in vals:
                if isinstance(v, bool):
                    continue
                out.add(json.dumps(v, sort_keys=True))
            return out
        for k, tv in gd.TRAIN_POOLS.items():
            ov = gd.TEST_POOLS.get(k, [])
            self.assertFalse(norm(tv) & norm(ov), msg=f"pool {k} overlaps")

    def test_unseen_templates_only_use_unseen_tools(self):
        unseen = {t["name"] for t in gd.UNSEEN_TOOLS}
        for _, tool in gd.TEST_UNSEEN_TEMPLATES:
            self.assertIn(tool, unseen)

    def test_seen_templates_only_use_seen_tools(self):
        seen = {t["name"] for t in gd.SEEN_TOOLS}
        for _, tool in gd.TRAIN_TEMPLATES + gd.TEST_SEEN_TEMPLATES:
            self.assertIn(tool, seen)


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.schemas = ToolSchemas(gd.ALL_TOOLS)

    def test_gold_is_valid_call(self):
        from schema_utils import normalize_call, validate_call
        for rows in (
            gd._gen(gd.TRAIN_TEMPLATES, gd.TRAIN_POOLS, 30, random.Random(1),
                    self.schemas, "train"),
            gd._gen(gd.TEST_UNSEEN_TEMPLATES, gd.UNSEEN_POOLS, 30, random.Random(1),
                    self.schemas, "unseen_schema"),
            gd._gen_notool(10, random.Random(1), self.schemas),
        ):
            for row in rows:
                gold = row["gold"]
                call = normalize_call(gold)
                self.assertEqual(validate_call(call, self.schemas), [],
                                 msg=f"{row['id']} {gold}")

    def test_assistant_matches_gold(self):
        for row in gd._gen(gd.TRAIN_TEMPLATES, gd.TRAIN_POOLS, 20, random.Random(2),
                           self.schemas, "train"):
            self.assertEqual(json.loads(row["messages"][2]["content"]), row["gold"])

    def test_notool_gold(self):
        for row in gd._gen_notool(5, random.Random(1), self.schemas):
            self.assertEqual(row["gold"], {"name": "no_tool", "arguments": {}})

    def test_determinism(self):
        a = gd._gen(gd.TRAIN_TEMPLATES, gd.TRAIN_POOLS, 40, random.Random(7),
                    self.schemas, "train")
        b = gd._gen(gd.TRAIN_TEMPLATES, gd.TRAIN_POOLS, 40, random.Random(7),
                    self.schemas, "train")
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])

    def test_heldout_tool_flag(self):
        seen = gd._gen(gd.TEST_SEEN_TEMPLATES, gd.TEST_POOLS, 20, random.Random(3),
                       self.schemas, "seen_paraphrase")
        unseen = gd._gen(gd.TEST_UNSEEN_TEMPLATES, gd.UNSEEN_POOLS, 20, random.Random(3),
                         self.schemas, "unseen_schema")
        self.assertTrue(all(not r["held_out_tool"] for r in seen))
        self.assertTrue(all(r["held_out_tool"] for r in unseen))


class TestCLI(unittest.TestCase):
    def test_cli_no_leakage_and_200_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "data"
            r = subprocess.run(
                [sys.executable, "src/gen_dataset.py", "--out", str(out),
                 "--seed", "3", "--n-train", "400", "--n-seen", "140",
                 "--n-unseen", "60", "--n-notool", "40"],
                capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            m = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(m["leakage_check"], "pass")
            self.assertGreaterEqual(m["n_heldout"], 200)
            self.assertGreaterEqual(m["n_unseen_schema"], 20)
            self.assertGreaterEqual(m["n_no_tool"], 20)
            train = {json.loads(l)["id"] for l in (out / "train.jsonl").read_text().splitlines()}
            held = {json.loads(l)["id"] for l in (out / "test.jsonl").read_text().splitlines()}
            self.assertFalse(train & held)

    def test_release_artifacts_consistent(self):
        d = ROOT / "data"
        if not (d / "MANIFEST.json").exists():
            self.skipTest("data/ not generated yet")
        m = json.loads((d / "MANIFEST.json").read_text())
        self.assertEqual(m["leakage_check"], "pass")
        self.assertGreaterEqual(m["n_heldout"], 200)
        self.assertTrue((d / "test_v1_original.jsonl").exists())


if __name__ == "__main__":
    unittest.main()