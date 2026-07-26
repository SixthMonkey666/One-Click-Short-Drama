from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.adapters import _generate_image
from app.local_models import LocalModelError, ModelQueueStatus, qwen_vlm_judge_stream
from tests import _test_env  # noqa: F401


class _FailingImageProvider:
    name = "flux"

    def __init__(self) -> None:
        self.released = False

    def generate_image(self, *args, **kwargs) -> None:
        raise RuntimeError("inference failed")

    def release(self) -> None:
        self.released = True


class _SmallImageProvider(_FailingImageProvider):
    def generate_image(self, _prompt, output_path, **kwargs) -> None:
        output_path.write_bytes(b"invalid")


class _QueueProbe:
    def __init__(self) -> None:
        self.released = False

    def wait(self, _model_name):
        yield ModelQueueStatus("queued", 1)

    def release(self) -> None:
        self.released = True


class TestProjectCompliance(unittest.TestCase):
    """Regression tests for non-negotiable production architecture rules."""

    def test_default_model_paths_use_home_ai_models(self):
        with patch.object(config.Path, "home", return_value=Path("/Users/tester")):
            self.assertEqual(
                config._default_qwen_model_dir(),
                Path("/Users/tester/ai_models/Qwen3-VL-4B-Instruct"),
            )
            self.assertEqual(
                config._default_flux_model_dir(),
                Path("/Users/tester/ai_models/FLUX.2-klein-4B"),
            )

    def test_runtime_modules_do_not_read_environment_directly(self):
        violations: list[str] = []
        for path in Path("app").rglob("*.py"):
            if path.name == "config.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Call, ast.Subscript)):
                    continue
                text = ast.unparse(node)
                if text.startswith(("os.getenv(", "os.environ[", "os.environ.get(")):
                    violations.append(f"{path}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_image_provider_failure_never_creates_placeholder(self):
        provider = _FailingImageProvider()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.png"
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                _generate_image(provider, "prompt", target, 512, 288)
            self.assertFalse(target.exists())
        self.assertTrue(provider.released)

    def test_invalid_image_output_is_deleted_and_reported(self):
        provider = _SmallImageProvider()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.png"
            with self.assertRaisesRegex(RuntimeError, "未生成有效文件"):
                _generate_image(provider, "prompt", target, 512, 288)
            self.assertFalse(target.exists())
        self.assertTrue(provider.released)

    def test_vlm_queue_slot_released_when_model_load_fails(self):
        queue = _QueueProbe()

        def no_memory_wait(*_args, **_kwargs):
            if False:
                yield None

        with (
            patch("app.local_models._INFERENCE_QUEUE", queue),
            patch("app.local_models._cleanup_gpu_memory"),
            patch("app.local_models._wait_for_available_memory", no_memory_wait),
            patch(
                "app.local_models._load_qwen",
                side_effect=LocalModelError("model unavailable"),
            ),
        ):
            generator = qwen_vlm_judge_stream([], "prompt", "image", {})
            self.assertEqual(next(generator)["message"], "queued")
            with self.assertRaisesRegex(LocalModelError, "model unavailable"):
                next(generator)
        self.assertTrue(queue.released)


if __name__ == "__main__":
    unittest.main()
