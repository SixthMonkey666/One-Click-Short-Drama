from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.adapters import _generate_image_stream
from app.agent import clear_agent_thread, stream_agent
from app.repository import create_project
from tests import _test_env  # noqa: F401


class _SlowImageProvider:
    name = "slow-test"

    def __init__(self) -> None:
        self.released = False

    def generate_image(
        self,
        _prompt,
        output_path,
        *,
        progress_callback=None,
        status_callback=None,
        **_kwargs,
    ) -> None:
        status_callback({"phase": "model_loading", "message": "正在加载测试模型"})
        time.sleep(0.04)
        progress_callback(1, 2)
        time.sleep(0.01)
        progress_callback(2, 2)
        output_path.write_bytes(b"x" * 20_000)

    def release(self) -> None:
        self.released = True


class TestProgressObservability(unittest.TestCase):
    def test_blocking_image_generation_emits_phase_progress_and_heartbeat(self):
        provider = _SlowImageProvider()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "image.png"
            with patch("app.adapters.GENERATION_HEARTBEAT_SECONDS", 0.005):
                generator = _generate_image_stream(
                    provider,
                    "prompt",
                    target,
                    64,
                    64,
                    "asset_progress",
                    {"asset_name": "测试角色"},
                )
                events = []
                while True:
                    try:
                        events.append(next(generator))
                    except StopIteration as stop:
                        provider_name = stop.value
                        break

        self.assertEqual(provider_name, "slow-test")
        self.assertTrue(provider.released)
        self.assertIn("generation_phase", {event["type"] for event in events})
        self.assertIn("generation_heartbeat", {event["type"] for event in events})
        progress = [event for event in events if event["type"] == "asset_progress"]
        self.assertEqual(progress[-1]["step"], 2)

    def test_agent_stream_contains_custom_node_and_pipeline_events(self):
        project = create_project("progress-stream", "雨夜里的橘猫")
        clear_agent_thread(project["id"])
        with patch("app.providers.mock.time.sleep"):
            events = list(stream_agent(project["id"]))

        custom = [
            item["data"]
            for item in events
            if item["stream_mode"] == "custom"
        ]
        self.assertTrue(any(item.get("type") == "node_started" for item in custom))
        self.assertTrue(any(item.get("type") == "node_completed" for item in custom))
        self.assertTrue(any(item.get("type") == "pipeline_event" for item in custom))
        self.assertTrue(any(item["stream_mode"] == "updates" for item in events))


if __name__ == "__main__":
    unittest.main()
