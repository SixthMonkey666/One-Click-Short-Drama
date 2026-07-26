from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.rubric import load_default_rubric
from app.local_models import LocalModelError
from app.providers import get_image_provider, get_judge_provider, get_llm_provider
from app.providers.judge_base import (
    JudgeProvider,
    JudgeResult,
    JudgeValidationError,
    validate_judge_result,
)
from app.providers.qwen_judge import QwenVLJudgeProvider
from tests import _test_env  # noqa: F401


class _RetryJudge(JudgeProvider):
    name = "retry-test"

    def __init__(self, valid_result: JudgeResult) -> None:
        self.calls = 0
        self.valid_result = valid_result

    def evaluate_image(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
    ) -> JudgeResult:
        self.calls += 1
        if self.calls == 1:
            return JudgeResult({}, 0, [], "invalid", 0.5)
        return self.valid_result


class TestJudgeValidation(unittest.TestCase):
    def setUp(self):
        self.rubric = load_default_rubric().to_dict()
        self.scores = {
            dimension["key"]: 4
            for dimension in self.rubric["dimensions"]
            if dimension.get("required", True)
        }
        self.valid = JudgeResult(
            self.scores,
            1.0,
            ["构图异常"],
            "评分结果说明",
            0.8,
            "raw",
        )

    def test_valid_result_is_normalized_and_recalculates_total(self):
        result = validate_judge_result(self.valid, self.rubric)
        self.assertEqual(result.total_score, 4.0)
        self.assertEqual(result.raw_response, "raw")

    def test_rejects_missing_required_dimension(self):
        invalid = JudgeResult({}, 0.0, [], "说明", 0.5)
        with self.assertRaises(JudgeValidationError):
            validate_judge_result(invalid, self.rubric)

    def test_rejects_unknown_dimension_and_tag(self):
        scores = dict(self.scores)
        scores["invented"] = 3
        invalid = JudgeResult(scores, 3.0, ["自创标签"], "说明", 0.5)
        with self.assertRaises(JudgeValidationError):
            validate_judge_result(invalid, self.rubric)

    def test_provider_retries_then_returns_validated_result(self):
        provider = _RetryJudge(self.valid)
        result = provider.evaluate_validated(
            Path("/tmp/test.png"),
            "prompt",
            "keyframe",
            self.rubric,
            max_retries=1,
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.total_score, 4.0)

    def test_explicit_provider_selection(self):
        self.assertEqual(get_judge_provider("mock").name, "mock")
        with self.assertRaises(ValueError):
            get_judge_provider("does-not-exist")

    def test_auto_provider_never_falls_back_to_mock(self):
        with (
            patch("app.providers._LLM_PROVIDER_NAME", "auto"),
            patch("app.providers.QwenVLProvider.is_available", return_value=False),
        ):
            get_llm_provider.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "qwen3_vl"):
                get_llm_provider()

        with (
            patch("app.providers._IMAGE_PROVIDER_NAME", "auto"),
            patch("app.providers.FluxProvider.is_available", return_value=False),
        ):
            get_image_provider.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "flux"):
                get_image_provider()

        with patch("app.providers.QwenVLJudgeProvider.is_available", return_value=False):
            get_judge_provider.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "qwen3_vl"):
                get_judge_provider("auto")

    def test_qwen_video_frames_are_cleaned_after_evaluation(self):
        captured_frames: list[Path] = []

        def fake_extract(_video, output_dir, max_frames=6):
            frame = output_dir / "frame_0001.jpg"
            frame.write_bytes(b"frame")
            captured_frames.append(frame)
            return [frame]

        def fake_stream(image_paths, _prompt, _task_type, _rubric):
            self.assertEqual(image_paths, captured_frames)
            yield {"type": "status"}
            return {
                "dimension_scores": self.scores,
                "bad_case_tags": [],
                "explanation": "视频评分说明",
                "confidence": 0.8,
                "raw_response": "raw",
            }

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video")
            with (
                patch(
                    "app.providers.qwen_judge.extract_video_frames",
                    side_effect=fake_extract,
                ),
                patch(
                    "app.providers.qwen_judge.qwen_vlm_judge_stream",
                    side_effect=fake_stream,
                ),
            ):
                result = QwenVLJudgeProvider().evaluate_validated(
                    video,
                    "prompt",
                    "video",
                    self.rubric,
                )

        self.assertEqual(result.total_score, 4.0)
        self.assertTrue(captured_frames)
        self.assertTrue(all(not frame.exists() for frame in captured_frames))

    def test_qwen_judge_mps_oom_falls_back_to_cpu_once(self):
        calls = []

        def fake_stream(_images, _prompt, _task_type, _rubric, **kwargs):
            calls.append(kwargs.get("device_override"))
            if kwargs.get("device_override") != "cpu":
                raise LocalModelError("VLM推理失败：MPS backend out of memory")
            if False:
                yield {}
            return {
                "dimension_scores": self.scores,
                "bad_case_tags": [],
                "explanation": "CPU保守模式评分完成",
                "confidence": 0.8,
                "raw_response": "raw",
            }

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"image")
            with (
                patch(
                    "app.providers.qwen_judge.qwen_vlm_judge_stream",
                    side_effect=fake_stream,
                ),
                patch("app.providers.qwen_judge.release_qwen") as release,
            ):
                provider = QwenVLJudgeProvider()
                result = provider.evaluate_validated(
                    image, "prompt", "keyframe", self.rubric,
                )

        self.assertEqual(calls, [None, "cpu"])
        self.assertEqual(provider._device_override, "cpu")
        self.assertEqual(result.total_score, 4.0)
        release.assert_called_once()

    def test_batch_judge_releases_after_every_item_and_continues_after_failure(self):
        from app.evaluation.ui_auto_judge import _run_batch_judge

        class BatchJudge:
            name = "batch"

            def __init__(self):
                self.calls = 0
                self.releases = 0

            def evaluate_validated(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first failed")
                return self.valid_result

            def release(self):
                self.releases += 1

        judge = BatchJudge()
        judge.valid_result = self.valid
        events = []
        items = [
            {
                "id": "item-1", "asset_path": "/tmp/one.png",
                "asset_type": "keyframe", "prompt": "", "reference_images": [],
            },
            {
                "id": "item-2", "asset_path": "/tmp/two.png",
                "asset_type": "keyframe", "prompt": "", "reference_images": [],
            },
        ]
        with (
            patch("app.providers.get_judge_provider", return_value=judge),
            patch("app.evaluation.repository.save_auto_scores"),
            patch("app.config.QWEN_JUDGE_RELEASE_EACH_ITEM", True),
        ):
            summary = _run_batch_judge(items, self.rubric, "batch", events.append)

        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(judge.calls, 2)
        self.assertEqual(judge.releases, 2)
        self.assertTrue(any(event["type"] == "judge_item_failed" for event in events))
        self.assertTrue(any(event["type"] == "judge_item_done" for event in events))


if __name__ == "__main__":
    unittest.main()
