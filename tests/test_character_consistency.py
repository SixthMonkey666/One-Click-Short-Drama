from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app.adapters import (
    CHARACTER_CONSISTENCY_SYSTEM_PROMPT,
    build_character_identity_prompt,
    generate_character_images_stream,
    generate_keyframe_image_stream,
)
from app.local_models import generate_flux_image
from app.repository import create_project
from tests import _test_env  # noqa: F401


def _drain(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return stop.value, events


class _CapturingImageProvider:
    name = "capture"
    default_steps = 2

    def __init__(self):
        self.calls = []

    def generate_image(
        self,
        prompt,
        output_path,
        *,
        seed=None,
        reference_images=None,
        progress_callback=None,
        status_callback=None,
        **kwargs,
    ):
        self.calls.append({
            "prompt": prompt,
            "seed": seed,
            "reference_images": list(reference_images or []),
            "kwargs": kwargs,
        })
        if status_callback:
            status_callback({"phase": "reference_conditioning", "message": "测试身份参考"})
        if progress_callback:
            progress_callback(2, 2)
        Path(output_path).write_bytes(b"consistent-image" * 100)

    def release(self):
        pass


class TestCharacterConsistency(unittest.TestCase):
    def test_identity_prompt_contains_immutable_character_anchors(self):
        prompt = build_character_identity_prompt({
            "id": "char-01",
            "name": "流浪橘猫",
            "type": "动物",
            "appearance": "orange tabby, golden eyes, torn left ear, red collar",
        })
        self.assertIn("CANONICAL IDENTITY [CHAR_01]", prompt)
        self.assertIn("torn left ear", prompt)
        self.assertIn("immutable", prompt)

    def test_character_views_share_seed_and_use_front_reference(self):
        project = create_project("consistency", "test")
        character = {
            "id": "char-01",
            "name": "流浪橘猫",
            "type": "动物",
            "appearance": "orange tabby, golden eyes, torn left ear, red collar",
            "reference_images": [],
        }
        provider = _CapturingImageProvider()
        with patch("app.adapters.get_image_provider", return_value=provider):
            images, _events = _drain(
                generate_character_images_stream(project["id"], character)
            )

        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(len({call["seed"] for call in provider.calls}), 1)
        self.assertEqual(provider.calls[0]["reference_images"], [])
        self.assertEqual(len(provider.calls[1]["reference_images"]), 1)
        self.assertTrue(all(
            CHARACTER_CONSISTENCY_SYSTEM_PROMPT in call["prompt"]
            for call in provider.calls
        ))
        self.assertTrue(all(image.get("identity_seed") for image in images))

        character["reference_images"] = images
        regeneration_provider = _CapturingImageProvider()
        with patch("app.adapters.get_image_provider", return_value=regeneration_provider):
            _drain(generate_character_images_stream(project["id"], character))
        self.assertTrue(all(
            call["reference_images"] == [Path(images[0]["path"])]
            for call in regeneration_provider.calls
        ))

    def test_keyframe_receives_identity_prompt_and_canonical_reference(self):
        project = create_project("keyframe-consistency", "test")
        reference = Path(_test_env.TEST_DATA_DIR) / "canonical-front.png"
        reference.write_bytes(b"reference")
        character = {
            "id": "char-01",
            "name": "下班女孩",
            "type": "人物",
            "appearance": "short black bob, green eyes, beige trench coat, red scarf",
            "reference_images": [{"view": "front", "path": str(reference)}],
        }
        analysis = {
            "visual_style": "cinematic rainy night",
            "characters": [character],
        }
        shot = {
            "id": "shot-01",
            "number": 1,
            "title": "女孩出现",
            "prompt": "woman walking under an umbrella",
            "character_ids": ["char-01"],
            "image_versions": [],
        }
        provider = _CapturingImageProvider()
        with patch("app.adapters.get_image_provider", return_value=provider):
            asset, _events = _drain(
                generate_keyframe_image_stream(project["id"], shot, analysis)
            )

        call = provider.calls[0]
        self.assertEqual(call["reference_images"], [reference])
        self.assertIn("CANONICAL IDENTITY [CHAR_01]", call["prompt"])
        self.assertIn("REFERENCE IMAGE 1", call["prompt"])
        self.assertEqual(asset["reference_conditioning"], [str(reference)])
        self.assertEqual(asset["character_consistency"][0]["character_id"], "char-01")

    def test_flux_runtime_passes_reference_image_and_requested_seed(self):
        reference = Path(_test_env.TEST_DATA_DIR) / "valid-reference.png"
        Image.new("RGB", (32, 32), "orange").save(reference)
        output = Path(_test_env.TEST_DATA_DIR) / "conditioned-output.png"
        captured = {}

        class FakeGenerator:
            def manual_seed(self, value):
                captured["seed"] = value
                return self

        class FakeTorch:
            @staticmethod
            def Generator(device=None):
                captured["device"] = device
                return FakeGenerator()

            @staticmethod
            def inference_mode():
                return nullcontext()

        class FakeOutputImage:
            def save(self, path, format=None):
                Path(path).write_bytes(b"conditioned" * 100)

        def fake_pipe(**kwargs):
            captured["image"] = kwargs.get("image")
            return SimpleNamespace(images=[FakeOutputImage()])

        with (
            patch.dict(sys.modules, {"torch": FakeTorch}),
            patch("app.local_models._load_flux", return_value=(fake_pipe, "mps")),
            patch("app.local_models.release_qwen"),
            patch("app.local_models._cleanup_gpu_memory"),
            patch("app.local_models._wait_for_available_memory", return_value=iter(())),
            patch("app.local_models.FLUX_MPS_OFFLOAD_MODE", "sequential"),
        ):
            generate_flux_image(
                "same character",
                output,
                width=64,
                height=64,
                seed=4242,
                reference_images=[reference],
            )

        self.assertEqual(captured["seed"], 4242)
        self.assertEqual(len(captured["image"]), 1)
        self.assertEqual(captured["image"][0].mode, "RGB")
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
