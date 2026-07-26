from __future__ import annotations

import csv
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.export import export_run_to_csv, export_run_to_json
from app.evaluation.repository import analyze_agreement
from app.evaluation.rubric import load_default_rubric
from tests import _test_env  # noqa: F401


class TestAgreementAndExport(unittest.TestCase):
    def setUp(self):
        self.rubric = load_default_rubric()
        keys = [dimension.key for dimension in self.rubric.dimensions]
        self.items = [
            {
                "id": "one",
                "status": "submitted",
                "total_score": 2.0,
                "auto_total_score": 2.5,
                "scores": {key: {"score": 2} for key in keys},
                "auto_scores": {key: {"score": 3} for key in keys},
                "bad_case_tags": [{"tag": "构图异常"}],
                "auto_bad_case_tags": [{"tag": "画面瑕疵"}],
                "auto_judge_provider": "mock",
                "auto_confidence": 0.8,
                "auto_explanation": "自动评分说明",
            },
            {
                "id": "two",
                "status": "submitted",
                "total_score": 4.0,
                "auto_total_score": 4.5,
                "scores": {key: {"score": 4} for key in keys},
                "auto_scores": {key: {"score": 5} for key in keys},
                "bad_case_tags": [],
                "auto_bad_case_tags": [],
            },
            {
                "id": "human-only",
                "status": "submitted",
                "total_score": 3.0,
                "auto_total_score": None,
                "scores": {},
                "auto_scores": {},
            },
        ]

    def test_agreement_uses_all_eligible_items(self):
        result = analyze_agreement(self.items)

        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["pearson_r"], 1.0)
        self.assertEqual(result["mae"], 0.5)
        self.assertEqual(result["within_one_rate"], 1.0)

    def test_json_and_csv_include_auto_judge_fields(self):
        run = {
            "id": "run",
            "name": "export",
            "rubric": self.rubric,
            "source_run_ids": ["run-a", "run-b"],
        }

        payload = json.loads(export_run_to_json(run, self.items[:1]))
        exported_item = payload["items"][0]
        self.assertEqual(payload["run"]["source_run_ids"], ["run-a", "run-b"])
        self.assertEqual(exported_item["auto_total_score"], 2.5)
        self.assertEqual(exported_item["auto_judge_provider"], "mock")
        self.assertEqual(exported_item["auto_bad_case_tags"], "画面瑕疵")

        rows = list(csv.DictReader(io.StringIO(export_run_to_csv(run, self.items[:1]))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["auto_total_score"], "2.5")
        self.assertEqual(rows[0]["auto_bad_case_tags"], "画面瑕疵")


if __name__ == "__main__":
    unittest.main()
