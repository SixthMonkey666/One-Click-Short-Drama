from __future__ import annotations

import threading
import time
import unittest

from app.task_manager import (
    GenerationTaskManager,
    TaskBusyError,
    get_active_task,
    get_task,
    submit_task,
    wait_for_task,
)
from tests import _test_env  # noqa: F401


class TestGenerationTaskManager(unittest.TestCase):
    def tearDown(self) -> None:
        active = get_active_task()
        if active:
            wait_for_task(active["id"], timeout=2)

    def test_only_one_task_can_run_across_manager_instances(self):
        started = threading.Event()
        release = threading.Event()

        def slow_runner(emit):
            emit({"type": "status", "message": "慢任务运行中"})
            started.set()
            release.wait(timeout=2)
            return {"ok": True}

        first = submit_task("text", "第一个任务", slow_runner, "project-a")
        self.assertTrue(started.wait(timeout=1))

        second_manager = GenerationTaskManager()
        snapshot = second_manager.active()
        self.assertEqual(snapshot["id"], first["id"])
        self.assertEqual(snapshot["message"], "慢任务运行中")
        with self.assertRaises(TaskBusyError):
            second_manager.submit("asset", "不应启动", lambda _emit: None, "project-b")

        release.set()
        completed = wait_for_task(first["id"], timeout=2)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["result"], {"ok": True})

    def test_failure_releases_global_slot_for_next_task(self):
        failed = submit_task(
            "storyboard",
            "失败任务",
            lambda _emit: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        failed_snapshot = wait_for_task(failed["id"], timeout=2)
        self.assertEqual(failed_snapshot["status"], "failed")
        self.assertIn("boom", failed_snapshot["message"])

        following = submit_task("idea", "后续任务", lambda emit: emit({
            "type": "status", "message": "后续任务执行",
        }))
        following_snapshot = wait_for_task(following["id"], timeout=2)
        self.assertEqual(following_snapshot["status"], "completed")
        persisted = get_task(following["id"])
        self.assertTrue(persisted["events"])

    def test_status_snapshot_is_updated_while_page_is_not_consuming_events(self):
        def runner(emit):
            emit({
                "type": "asset_start",
                "asset_name": "流浪橘猫",
                "view_label": "正面全身",
            })
            emit({"type": "generation_phase", "phase": "model_loading", "message": "加载模型"})
            time.sleep(0.03)

        task = submit_task("asset", "后台资产生成", runner, "project-progress")
        snapshot = wait_for_task(task["id"], timeout=2)
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["current_asset"], "流浪橘猫")
        self.assertEqual(snapshot["current_view"], "正面全身")
        self.assertTrue(any(
            event.get("phase") == "model_loading"
            for event in snapshot["events"]
        ))

    def test_task_context_is_persisted_for_dynamic_page_routing(self):
        context = {
            "app_view": "evaluation",
            "eval_view": "auto",
            "eval_run_id": "run-routing",
        }
        task = submit_task(
            "auto_judge",
            "自动机评路由测试",
            lambda emit: emit({"type": "judge_batch_complete", "total": 1}),
            "project-routing",
            context=context,
        )
        snapshot = wait_for_task(task["id"], timeout=2)
        self.assertEqual(snapshot["context"], context)


if __name__ == "__main__":
    unittest.main()
