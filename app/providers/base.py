"""Provider base classes defining the interface for LLM, Image, and Video generation backends.

To add a new model backend:
1. Subclass LLMProvider, ImageProvider, or VideoProvider in a new file under providers/
2. Implement all abstract methods
3. Register it in the factory in __init__.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generator

ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[dict[str, Any]], None]


class LLMProvider(ABC):
    """Abstract base class for large language model providers (text generation).

    All LLM providers must implement these methods. The streaming methods yield
    typed event dicts consumed by the UI layer, and return parsed results via
    StopIteration.
    """

    name: str = "base"
    display_name: str = "Base LLM"

    @abstractmethod
    def generate_idea_stream(
        self,
    ) -> Generator[dict[str, Any] | str, None, list[dict[str, Any]]]:
        """Stream idea events and return selectable idea dictionaries."""
        ...

    @abstractmethod
    def generate_analysis_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        """Streaming creative analysis/understanding of the source idea.

        Analyzes the idea to extract: title, genre, theme, tone, visual_style,
        total_duration_seconds, narrative_summary, characters, props, environments.

        Yields events: {"type": "status", "message": str}, {"type": "text_start"},
        {"type": "text_delta", "delta": str}, {"type": "text_end"}
        Returns: analysis dict with structured creative understanding.
        """
        ...

    @abstractmethod
    def generate_characters_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        """Extract structured character identities from source text."""
        ...

    @abstractmethod
    def generate_storyboard_stream(
        self, source_text: str, analysis: dict[str, Any]
    ) -> Generator[dict[str, Any], None, tuple[str, list[dict[str, Any]]]]:
        """Streaming storyboard table generation based on creative analysis.

        Each shot contains: id, number, title, description, duration_seconds,
        camera_angle, camera_movement, character_ids, prop_ids, environment_id, prompt.

        Yields events: status, text_start, text_delta, text_end
        Returns: (script_text, list_of_shots)
        """
        ...

    @abstractmethod
    def generate_video_prompts_stream(
        self,
        source_text: str,
        script: str,
        shots: list[dict[str, Any]],
        characters: list[dict[str, Any]] | None = None,
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        """Streaming video motion prompt generation for I2V.

        Yields events: status, text_start, text_delta, text_end
        Returns: list of video prompt dicts with keys:
          shot_number, motion_prompt_en, motion_prompt_cn, camera_movement
        """
        ...

    def release(self) -> None:
        """Release model from GPU memory. Override if the provider loads models."""
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if this provider is available (model files exist, dependencies installed, etc.)."""
        return True


class ImageProvider(ABC):
    """Abstract base class for text-to-image generation providers.

    Image providers handle the core inference: taking a prompt and producing a PNG file.
    File management, output validation, asset metadata, and UI event formatting
    are handled by the adapters layer. Production errors must propagate.
    """

    name: str = "base"
    display_name: str = "Base Image Model"
    default_steps: int = 4
    supports_reference_images: bool = False

    @abstractmethod
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
        """Generate an image from a text prompt and save it to output_path.

        Args:
            prompt: English text prompt
            output_path: Path to save the PNG
            width: Image width in pixels
            height: Image height in pixels
            progress_callback: Optional callback(step, total_steps) for progress tracking
            status_callback: Optional callback receiving structured runtime phase events
            seed: Optional deterministic seed used to stabilize an identity or shot
            reference_images: Optional canonical images used for visual conditioning
        """
        ...

    def release(self) -> None:
        """Release model from GPU memory."""
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if this provider is available."""
        return True


class VideoProvider(ABC):
    """Abstract base class for image-to-video / video generation providers.

    Video providers take keyframe images and generate a video file.
    Supports both local FFmpeg (mock/slideshow) and future remote API I2V models.
    """

    name: str = "base"
    display_name: str = "Base Video"

    @abstractmethod
    def generate_video(
        self, project_id: str, shots: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate a video from keyframe shots.

        Args:
            project_id: Project identifier for asset paths
            shots: List of shot dicts with selected_image, duration_seconds, video_prompt

        Returns: video info dict with keys: provider, status, path, manifest
        """
        ...

    def release(self) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        return True
