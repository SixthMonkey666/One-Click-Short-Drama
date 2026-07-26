from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.rubric import load_default_rubric
from app.evaluation.scoring import (
    SCORE_LABELS,
    calculate_weighted_total,
    filter_items,
    score_color,
    summarize_items,
    validate_scores,
)
from tests import _test_env  # noqa: F401


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.rubric = load_default_rubric()

    def test_score_labels_five_levels(self):
        self.assertEqual(len(SCORE_LABELS), 5)
        for i in range(1, 6):
            self.assertIn(i, SCORE_LABELS)

    def test_perfect_scores(self):
        scores = {d.key: 5 for d in self.rubric.dimensions}
        total = calculate_weighted_total(scores, self.rubric)
        self.assertAlmostEqual(total, 5.0, places=1)
        ok, errs = validate_scores(scores, self.rubric)
        self.assertTrue(ok)
        self.assertEqual(errs, [])

    def test_all_ones(self):
        scores = {d.key: 1 for d in self.rubric.dimensions}
        total = calculate_weighted_total(scores, self.rubric)
        self.assertAlmostEqual(total, 1.0, places=1)

    def test_missing_required_dimension_fails(self):
        scores = {}
        for d in self.rubric.dimensions[1:]:
            scores[d.key] = 3
        total = calculate_weighted_total(scores, self.rubric)
        self.assertEqual(total, 0.0)
        ok, errs = validate_scores(scores, self.rubric)
        self.assertFalse(ok)
        self.assertTrue(len(errs) >= 1)

    def test_missing_optional_dimension_ok(self):
        scores = {}
        for d in self.rubric.dimensions:
            if d.required:
                scores[d.key] = 4
        ok, errs = validate_scores(scores, self.rubric)
        self.assertTrue(ok, f"Should pass with required dims only: {errs}")
        total = calculate_weighted_total(scores, self.rubric)
        self.assertGreater(total, 0)

    def test_invalid_score_value(self):
        scores = {d.key: 3 for d in self.rubric.dimensions}
        first_key = self.rubric.dimensions[0].key
        scores[first_key] = 6
        ok, errs = validate_scores(scores, self.rubric)
        self.assertFalse(ok)

    def test_weighted_average(self):
        d_high = self.rubric.dimensions[0]
        scores = {d.key: 3 for d in self.rubric.dimensions}
        scores[d_high.key] = 5
        total = calculate_weighted_total(scores, self.rubric)
        self.assertGreater(total, 3.0)
        self.assertLess(total, 5.0)

    def test_score_color_ranges(self):
        self.assertEqual(score_color(5.0), "#22c55e")
        self.assertEqual(score_color(4.0), "#22c55e")
        self.assertEqual(score_color(3.5), "#eab308")
        self.assertEqual(score_color(3.0), "#eab308")
        self.assertEqual(score_color(2.5), "#f97316")
        self.assertEqual(score_color(1.5), "#ef4444")

    def test_summarize_empty(self):
        summary = summarize_items([], self.rubric)
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["submitted"], 0)
        self.assertEqual(summary["avg_total"], 0.0)

    def test_summarize_items(self):
        items = [
            {
                "id": "a", "status": "submitted", "total_score": 4.0,
                "scores": {d.key: {"score": 4} for d in self.rubric.dimensions},
                "bad_case_tags": [],
            },
            {
                "id": "b", "status": "submitted", "total_score": 3.0,
                "scores": {d.key: {"score": 3} for d in self.rubric.dimensions},
                "bad_case_tags": [{"tag": "构图异常"}],
            },
            {
                "id": "c", "status": "pending", "total_score": None,
                "scores": {}, "bad_case_tags": [],
            },
        ]
        summary = summarize_items(items, self.rubric)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["submitted"], 2)
        self.assertAlmostEqual(summary["avg_total"], 3.5, places=1)
        self.assertEqual(summary["tag_counts"].get("构图异常"), 1)
        self.assertEqual(sum(summary["distribution"].values()), 2)

    def test_filter_items_combines_dashboard_filters(self):
        items = [
            {"id": "a", "status": "submitted", "asset_type": "keyframe", "model_provider": "mock"},
            {"id": "b", "status": "pending", "asset_type": "video", "model_provider": "qwen"},
            {"id": "c", "status": "submitted", "asset_type": "video", "model_provider": "qwen"},
        ]
        filtered = filter_items(
            items,
            statuses={"submitted"},
            asset_types={"video"},
            model_providers={"qwen"},
        )
        self.assertEqual([item["id"] for item in filtered], ["c"])
        self.assertEqual(filter_items(items), items)


if __name__ == "__main__":
    unittest.main()
