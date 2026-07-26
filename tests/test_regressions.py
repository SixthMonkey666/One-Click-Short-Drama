from __future__ import annotations

import socket
import ssl
import subprocess
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.local_models import (
    _check_network_available,
    _is_mps_oom_error,
    _make_ssl_context,
    _parse_analysis_response,
    generate_flux_image,
)
from app.providers import get_configured_provider_status
from app.providers.mock import FFmpegVideoProvider
from app.repository import character_asset_directory
from app.workflow import generate_assets_stream
from tests import _test_env  # noqa: F401


class TestSecurityRegressions(unittest.TestCase):
    def test_asset_ids_cannot_escape_asset_root(self):
        with self.assertRaises(ValueError):
            character_asset_directory("project-1", "../../../../outside")

    def test_analysis_replaces_model_supplied_asset_ids(self):
        analysis = _parse_analysis_response(
            {
                "characters": [{"id": "../../../../tmp", "name": "角色"}],
                "props": [{"id": "../prop", "name": "道具"}],
                "environments": [{"id": "/tmp/env", "name": "场景"}],
            }
        )
        self.assertEqual(analysis["characters"][0]["id"], "char-01")
        self.assertEqual(analysis["props"][0]["id"], "prop-01")
        self.assertEqual(analysis["environments"][0]["id"], "env-01")

    def test_hot_topic_ssl_context_verifies_certificates(self):
        context = _make_ssl_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_network_probe_does_not_change_global_socket_timeout(self):
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(None)
        try:
            with patch(
                "app.local_models.socket.create_connection",
                side_effect=OSError("offline"),
            ):
                self.assertFalse(_check_network_available())
            self.assertIsNone(socket.getdefaulttimeout())
        finally:
            socket.setdefaulttimeout(previous_timeout)

    def test_provider_status_does_not_raise_when_model_is_unavailable(self):
        with (
            patch("app.providers._LLM_PROVIDER_NAME", "qwen3_vl"),
            patch(
                "app.providers.QwenVLProvider.is_available",
                return_value=False,
            ),
        ):
            status = get_configured_provider_status()
        self.assertIn("llm", status)
        self.assertFalse(status["llm"]["available"])
        self.assertIn("error", status["llm"])


class TestWorkflowRegressions(unittest.TestCase):
    def test_character_view_progress_reaches_total(self):
        project = {
            "id": "project-1",
            "status": "new",
            "analysis": {
                "characters": [
                    {
                        "id": "char-01",
                        "name": "角色",
                        "reference_images": [],
                    }
                ],
                "props": [],
                "environments": [],
            },
            "characters": [],
            "shots": [],
        }

        def fake_character_stream(project_id, character):
            images = []
            for view in ("front", "side", "back", "face"):
                image = {"view": view, "view_label": view}
                images.append(image)
                yield {"type": "asset_done", "image": image}
            return images

        with (
            patch("app.workflow.get_project", return_value=project),
            patch("app.workflow.save_project", side_effect=lambda value: value),
            patch(
                "app.workflow.generate_character_images_stream",
                side_effect=fake_character_stream,
            ),
            patch("app.workflow._release_models"),
        ):
            events = list(generate_assets_stream(project["id"]))

        progress = [
            event
            for event in events
            if event.get("type") == "asset_progress_overall"
        ]
        self.assertEqual(progress[-1]["done"], progress[-1]["total"])
        self.assertEqual(progress[-1]["total"], 4)

    def test_ffmpeg_failure_is_not_reported_as_completed(self):
        image_path = Path(_test_env.TEST_DATA_DIR) / "frame.png"
        image_path.write_bytes(b"fake png")
        shots = [
            {
                "number": 1,
                "duration_seconds": 2,
                "selected_image": {"path": str(image_path)},
                "video_prompt": None,
            }
        ]
        failure = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr=b"encoder failed",
        )
        with (
            patch("app.providers.mock.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("app.providers.mock.subprocess.run", side_effect=failure),
        ):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                FFmpegVideoProvider().generate_video("project-1", shots)

    def test_flux_mps_oom_retries_with_sequential_offload(self):
        class FakeGenerator:
            def manual_seed(self, _seed):
                return self

        class FakeTorch:
            @staticmethod
            def Generator(device=None):
                return FakeGenerator()

            @staticmethod
            def inference_mode():
                return nullcontext()

        class FakeImage:
            def save(self, path, format=None):
                Path(path).write_bytes(b"valid generated image")

        first_pipe = unittest.mock.Mock(
            side_effect=RuntimeError("MPS backend out of memory")
        )
        second_pipe = unittest.mock.Mock(
            return_value=SimpleNamespace(images=[FakeImage()])
        )
        output = Path(_test_env.TEST_DATA_DIR) / "flux-retry.png"

        with (
            patch.dict(sys.modules, {"torch": FakeTorch}),
            patch(
                "app.local_models._load_flux",
                side_effect=[(first_pipe, "mps"), (second_pipe, "mps")],
            ) as load_flux,
            patch("app.local_models.release_qwen"),
            patch("app.local_models.release_flux"),
            patch("app.local_models._cleanup_gpu_memory"),
            patch("app.local_models._wait_for_available_memory", return_value=iter(())),
            patch("app.local_models.FLUX_MPS_OFFLOAD_MODE", "model"),
            patch("app.local_models.FLUX_MPS_OOM_RETRIES", 1),
        ):
            generate_flux_image("prompt", output, width=512, height=288)

        self.assertTrue(output.exists())
        self.assertEqual(
            [call.args[0] for call in load_flux.call_args_list],
            ["model", "sequential"],
        )
        self.assertTrue(_is_mps_oom_error(RuntimeError("MPS backend out of memory")))

    def test_flux_sequential_mode_is_applied_before_mps_inference(self):
        from app.local_models import _load_flux

        class FakeVae:
            def __init__(self):
                self.tiling_enabled = False

            def enable_tiling(self):
                self.tiling_enabled = True

        class FakePipe:
            def __init__(self):
                self.vae = FakeVae()
                self.sequential_device = None

            def enable_sequential_cpu_offload(self, device):
                self.sequential_device = device

        fake_pipe = FakePipe()

        class FakePipelineClass:
            _flux_mps_patched = True

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return fake_pipe

        fake_torch = ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float32 = "float32"
        fake_diffusers = ModuleType("diffusers")
        fake_diffusers.Flux2KleinPipeline = FakePipelineClass

        _load_flux.cache_clear()
        try:
            with (
                patch.dict(
                    sys.modules,
                    {"torch": fake_torch, "diffusers": fake_diffusers},
                ),
                patch("app.local_models._enable_native_packages"),
                patch("app.local_models.release_qwen"),
                patch("app.local_models._select_device", return_value="mps"),
                patch("app.local_models.LOCAL_FLUX_MODEL_DIR", Path(_test_env.TEST_DATA_DIR)),
            ):
                loaded, device = _load_flux("sequential")
        finally:
            _load_flux.cache_clear()

        self.assertIs(loaded, fake_pipe)
        self.assertEqual(device, "mps")
        self.assertEqual(fake_pipe.sequential_device, "mps")
        self.assertTrue(fake_pipe.vae.tiling_enabled)


if __name__ == "__main__":
    unittest.main()
