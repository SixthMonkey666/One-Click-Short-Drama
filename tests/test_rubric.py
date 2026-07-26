from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.rubric import Rubric, RubricDimension, _built_in_default_rubric, load_default_rubric
from tests import _test_env  # noqa: F401


class TestRubricDimension(unittest.TestCase):
    def test_anchor_text(self):
        d = RubricDimension("instruction_following", "指令遵循", anchors={1: "bad", 5: "good"})
        self.assertEqual(d.anchor_text(1), "bad")
        self.assertEqual(d.anchor_text(5), "good")
        self.assertEqual(d.anchor_text(3), "")

    def test_to_dict_and_from_dict(self):
        d = RubricDimension("test_dim", "测试维度", weight=0.5, required=False, anchors={1: "a", 3: "b"})
        d2 = RubricDimension.from_dict(d.to_dict())
        self.assertEqual(d2.key, "test_dim")
        self.assertEqual(d2.label, "测试维度")
        self.assertEqual(d2.weight, 0.5)
        self.assertFalse(d2.required)
        self.assertEqual(d2.anchors[1], "a")
        self.assertEqual(d2.anchors[3], "b")


class TestRubric(unittest.TestCase):
    def test_default_rubric_loads(self):
        r = load_default_rubric()
        self.assertIsInstance(r, Rubric)
        self.assertTrue(len(r.dimensions) >= 5)
        self.assertTrue(len(r.bad_case_tags) >= 10)

    def test_total_weight(self):
        r = _built_in_default_rubric()
        self.assertGreater(r.total_weight, 0)

    def test_required_and_optional(self):
        r = load_default_rubric()
        required = r.required_dimensions
        optional = r.optional_dimensions
        self.assertTrue(len(required) >= 1)
        total = len(required) + len(optional)
        self.assertEqual(total, len(r.dimensions))
        for d in required:
            self.assertTrue(d.required)
        for d in optional:
            self.assertFalse(d.required)

    def test_get_dimension(self):
        r = _built_in_default_rubric()
        d = r.get_dimension("instruction_following")
        self.assertIsNotNone(d)
        self.assertEqual(d.label, "指令遵循")
        self.assertIsNone(r.get_dimension("nonexistent"))

    def test_json_roundtrip(self):
        r = load_default_rubric()
        j = r.to_json()
        r2 = Rubric.from_json(j)
        self.assertEqual(r.version, r2.version)
        self.assertEqual(r.name, r2.name)
        self.assertEqual(len(r.dimensions), len(r2.dimensions))
        self.assertEqual(r.bad_case_tags, r2.bad_case_tags)

    def test_builtin_fallback(self):
        r = _built_in_default_rubric()
        self.assertIn("instruction_following", [d.key for d in r.dimensions])
        self.assertIn("构图异常", r.bad_case_tags)


if __name__ == "__main__":
    unittest.main()
