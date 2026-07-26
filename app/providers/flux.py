"""FLUX.2-klein-4B image generation provider."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.local_models import (
    FluxProgressCallback,
    generate_flux_image,
    release_flux,
)
from app.providers.base import ImageProvider, StatusCallback


class FluxProvider(ImageProvider):
    """Local FLUX.2-klein-4B image generation provider.

    Black Forest Labs' distilled 4B-parameter model that generates images in 4 steps.
    Runs on MPS (Apple Silicon), CUDA, or CPU.
    """

    name = "flux"
    display_name = "FLUX.2-klein-4B (本地)"
    default_steps = 4
    supports_reference_images = True

    def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 576,
        progress_callback: Callable[[int, int], None] | None = None,
        status_callback: StatusCallback | None = None,
        seed: int | None = None,
        reference_images: list[Path] | None = None,
    ) -> None:
        cb = None
        if progress_callback is not None:
            cb = FluxProgressCallback(progress_callback, status_callback)
            progress_callback(0, self.default_steps)

        generate_flux_image(
            prompt,
            output_path,
            width=width,
            height=height,
            progress_callback=cb,
            status_callback=status_callback,
            seed=seed,
            reference_images=reference_images,
        )

        if progress_callback is not None and cb is not None:
            progress_callback(cb.step, self.default_steps)

    def release(self) -> None:
        release_flux()

    @classmethod
    def is_available(cls) -> bool:
        from app.config import LOCAL_FLUX_MODEL_DIR
        return LOCAL_FLUX_MODEL_DIR.exists()
