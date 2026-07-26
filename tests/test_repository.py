from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app.config import ensure_directories
from app.evaluation import repository as eval_repo
from app.evaluation.rubric import Rubric, RubricDimension, load_default_rubric
from tests import _test_env  # noqa: F401


class TestEvaluationRepository(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_directories()
        cls.tmpdir = tempfile.mkdtemp()
        cls.fake_img = os.path.join(cls.tmpdir, "test.png")
        Image.new("RGB", (200, 200), "blue").save(cls.fake_img)
        cls._test_run_ids: list[str] = []

    @classmethod
    def tearDownClass(cls):
        import shutil
        for rid in cls._test_run_ids:
            try:
                eval_repo.delete_run(rid)
            except Exception:
                pass
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_default_rubric_seeded(self):
        r = eval_repo.get_or_create_default_rubric()
        self.assertIsNotNone(r)
        self.assertIn("rubric", r)
        self.assertEqual(r["is_default"], 1)

    def test_create_run_and_items(self):
        run = eval_repo.create_evaluation_run("Repo测试任务", "keyframe", judge_type="human")
        self.__class__._test_run_ids.append(run["id"])
        self.assertIn("id", run)
        self.assertEqual(run["name"], "Repo测试任务")
        self.assertEqual(run["status"], "pending")

        item = eval_repo.add_evaluation_item(
            run_id=run["id"],
            asset_path=self.fake_img,
            asset_type="keyframe",
            prompt="test prompt",
            model_provider="mock",
            order_index=0,
        )
        self.assertEqual(item["status"], "pending")

        run2 = eval_repo.get_run(run["id"])
        self.assertEqual(run2["status"], "in_progress")

        items = eval_repo.list_run_items(run["id"])
        self.assertEqual(len(items), 1)

        progress = eval_repo.get_run_progress(run["id"])
        self.assertEqual(progress["total"], 1)
        self.assertEqual(progress["submitted"], 0)

    def test_save_scores_and_progress(self):
        run = eval_repo.create_evaluation_run("评分测试", "keyframe")
        item = eval_repo.add_evaluation_item(
            run_id=run["id"], asset_path=self.fake_img, asset_type="keyframe",
            prompt="test", model_provider="mock", order_index=0,
        )
        rubric = load_default_rubric()
        scores = {d.key: 4 for d in rubric.dimensions}
        saved = eval_repo.save_item_scores(
            item_id=item["id"],
            scores=scores,
            bad_case_tags=[{"tag": "构图异常"}, {"tag": "自定义问题"}],
            note="测试备注",
            total_score=4.0,
            submit=True,
        )
        self.assertEqual(saved["status"], "submitted")
        self.assertEqual(len(saved["scores"]), len(rubric.dimensions))
        self.assertEqual(
            {entry["tag"] for entry in saved["bad_case_tags"]},
            {"构图异常", "自定义问题"},
        )

        progress = eval_repo.get_run_progress(run["id"])
        self.assertEqual(progress["submitted"], 1)
        self.assertEqual(progress["progress_pct"], 100.0)

        run_done = eval_repo.get_run(run["id"])
        self.assertEqual(run_done["status"], "completed")

        stats = eval_repo.get_run_stats(run["id"])
        self.assertEqual(stats["total_items"], 1)
        self.assertEqual(stats["submitted_items"], 1)
        self.assertAlmostEqual(stats["avg_total_score"], 4.0, places=1)
        self.assertIn("构图异常", stats["tag_counts"])
        self.assertIn("自定义问题", stats["tag_counts"])

        eval_repo.delete_run(run["id"])

    def test_list_runs(self):
        runs_before = len(eval_repo.list_runs())
        run = eval_repo.create_evaluation_run("列表测试", "character")
        runs_after = len(eval_repo.list_runs())
        self.assertEqual(runs_after, runs_before + 1)
        eval_repo.delete_run(run["id"])

    def test_auto_scores_are_validated_and_replace_previous_provider(self):
        run = eval_repo.create_evaluation_run("机评保存测试", "keyframe")
        item = eval_repo.add_evaluation_item(
            run_id=run["id"],
            asset_path=self.fake_img,
            asset_type="keyframe",
        )
        rubric = load_default_rubric()
        scores = {
            dimension.key: 3
            for dimension in rubric.required_dimensions
        }

        eval_repo.save_auto_scores(
            item["id"],
            scores,
            [],
            1.0,
            "Mock 说明",
            0.8,
            provider_name="mock",
        )
        eval_repo.save_auto_scores(
            item["id"],
            {key: 4 for key in scores},
            ["构图异常"],
            1.0,
            "Qwen 说明",
            0.9,
            provider_name="qwen3_vl",
        )

        saved = eval_repo.get_item(item["id"])
        self.assertEqual(saved["auto_judge_provider"], "qwen3_vl")
        self.assertEqual(saved["auto_total_score"], 4.0)
        self.assertEqual(len(saved["auto_scores"]), len(scores))
        self.assertEqual(
            [entry["tag"] for entry in saved["auto_bad_case_tags"]],
            ["构图异常"],
        )

        with self.assertRaises(ValueError):
            eval_repo.save_auto_scores(
                item["id"],
                {},
                [],
                0.0,
                "",
                0.0,
                provider_name="mock",
            )
        eval_repo.delete_run(run["id"])

    def test_create_run_with_selected_rubric_version(self):
        rubric = Rubric(
            version="test-v2",
            name="测试 Rubric",
            task_type="keyframe",
            dimensions=[RubricDimension("quality", "质量")],
            bad_case_tags=["测试问题"],
        )
        stored = eval_repo.create_rubric_version(rubric)
        same_version = eval_repo.create_rubric_version(rubric)
        self.assertEqual(stored["id"], same_version["id"])

        run = eval_repo.create_evaluation_run(
            "指定 Rubric",
            "keyframe",
            rubric_version_id=stored["id"],
        )
        self.assertEqual(run["rubric_version_id"], stored["id"])
        loaded = eval_repo.get_run(run["id"])
        self.assertEqual(loaded["rubric"].version, "test-v2")
        eval_repo.delete_run(run["id"])

    def test_create_run_rejects_unknown_rubric(self):
        with self.assertRaises(ValueError):
            eval_repo.create_evaluation_run(
                "无效 Rubric",
                "keyframe",
                rubric_version_id="missing-rubric",
            )

    def test_discover_project_assets_no_project(self):
        project = {"id": None, "shots": [], "characters": [], "analysis": None}
        assets = eval_repo.discover_project_assets(project)
        self.assertEqual(assets, [])


if __name__ == "__main__":
    unittest.main()
