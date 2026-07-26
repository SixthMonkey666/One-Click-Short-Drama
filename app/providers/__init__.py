"""Provider factory — resolves which LLM/Image/Video backend to use based on config.

To register a new provider:
1. Create a new file in providers/ implementing the appropriate ABC from base.py
2. Import it here and add it to _LLM_REGISTRY / _IMAGE_REGISTRY / _VIDEO_REGISTRY
3. Set LLM_PROVIDER / IMAGE_PROVIDER / VIDEO_PROVIDER in .env or config

Usage in adapters or other layers:
    from app.providers import get_llm_provider, get_image_provider, get_video_provider
    llm = get_llm_provider()
    img = get_image_provider()
    vid = get_video_provider()
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from app.config import CUSTOM_IMAGE_PROVIDER_CLASS, CUSTOM_LLM_PROVIDER_CLASS
from app.config import IMAGE_PROVIDER as _IMAGE_PROVIDER_NAME
from app.config import JUDGE_PROVIDER as _JUDGE_PROVIDER_NAME
from app.config import LLM_PROVIDER as _LLM_PROVIDER_NAME
from app.config import VIDEO_PROVIDER as _VIDEO_PROVIDER_NAME
from app.providers.base import ImageProvider, LLMProvider, VideoProvider
from app.providers.flux import FluxProvider
from app.providers.judge_base import JudgeProvider
from app.providers.mock import FFmpegVideoProvider, MockImageProvider, MockLLMProvider, MockVideoProvider
from app.providers.mock_judge import MockJudgeProvider
from app.providers.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.openai_image import OpenAICompatibleImageProvider
from app.providers.qwen import QwenVLProvider
from app.providers.qwen_judge import QwenVLJudgeProvider

_LLM_REGISTRY: dict[str, type[LLMProvider]] = {
    "qwen3_vl": QwenVLProvider,
    "openai_compatible": OpenAICompatibleLLMProvider,
    "mock": MockLLMProvider,
}

_IMAGE_REGISTRY: dict[str, type[ImageProvider]] = {
    "flux": FluxProvider,
    "openai_compatible": OpenAICompatibleImageProvider,
    "mock": MockImageProvider,
}

_VIDEO_REGISTRY: dict[str, type[VideoProvider]] = {
    "ffmpeg": FFmpegVideoProvider,
    "mock": MockVideoProvider,
}

_JUDGE_REGISTRY: dict[str, type[JudgeProvider]] = {
    "mock": MockJudgeProvider,
    "qwen3_vl": QwenVLJudgeProvider,
}


def _load_custom_provider(
    entrypoint: str,
    expected_base: type[Any],
    kind: str,
) -> type[Any]:
    """Load ``module:Class`` after explicit user selection."""
    if not entrypoint or ":" not in entrypoint:
        raise ValueError(
            f"CUSTOM_{kind.upper()}_PROVIDER_CLASS 必须使用 module:Class 格式"
        )
    module_name, class_name = entrypoint.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
        provider_class = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"无法加载自定义 {kind} Provider：{entrypoint}"
        ) from exc
    if not isinstance(provider_class, type) or not issubclass(
        provider_class, expected_base
    ):
        raise TypeError(
            f"{entrypoint} 必须继承 {expected_base.__name__}"
        )
    return provider_class


def _llm_registry(
    include_custom: bool | None = None,
) -> dict[str, type[LLMProvider]]:
    registry = dict(_LLM_REGISTRY)
    should_load = (
        _LLM_PROVIDER_NAME == "custom"
        if include_custom is None
        else include_custom
    )
    if should_load and CUSTOM_LLM_PROVIDER_CLASS:
        registry["custom"] = _load_custom_provider(
            CUSTOM_LLM_PROVIDER_CLASS, LLMProvider, "llm"
        )
    return registry


def _image_registry(
    include_custom: bool | None = None,
) -> dict[str, type[ImageProvider]]:
    registry = dict(_IMAGE_REGISTRY)
    should_load = (
        _IMAGE_PROVIDER_NAME == "custom"
        if include_custom is None
        else include_custom
    )
    if should_load and CUSTOM_IMAGE_PROVIDER_CLASS:
        registry["custom"] = _load_custom_provider(
            CUSTOM_IMAGE_PROVIDER_CLASS, ImageProvider, "image"
        )
    return registry


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider instance.

    ``auto`` resolves to the real local provider. Mock providers must be
    selected explicitly for tests.
    """
    name = _LLM_PROVIDER_NAME
    if name == "auto":
        name = "qwen3_vl"
    registry = _llm_registry(include_custom=name == "custom")
    if name not in registry:
        raise ValueError(f"Unknown LLM provider: {name}")
    cls = registry[name]
    if not cls.is_available():
        raise RuntimeError(f"LLM provider is unavailable: {name}")
    return cls()


@lru_cache(maxsize=1)
def get_image_provider() -> ImageProvider:
    """Get the configured image generation provider instance.

    ``auto`` resolves to the real local provider. Mock providers must be
    selected explicitly for tests.
    """
    name = _IMAGE_PROVIDER_NAME
    if name == "auto":
        name = "flux"
    registry = _image_registry(include_custom=name == "custom")
    if name not in registry:
        raise ValueError(f"Unknown image provider: {name}")
    cls = registry[name]
    if not cls.is_available():
        raise RuntimeError(f"Image provider is unavailable: {name}")
    return cls()


@lru_cache(maxsize=1)
def get_video_provider() -> VideoProvider:
    """Get the configured video generation provider instance."""
    name = _VIDEO_PROVIDER_NAME
    if name == "auto":
        name = "ffmpeg"
    elif name not in _VIDEO_REGISTRY:
        raise ValueError(f"Unknown video provider: {name}")
    cls = _VIDEO_REGISTRY[name]
    if not cls.is_available():
        raise RuntimeError(f"Video provider is unavailable: {name}")
    return cls()


@lru_cache(maxsize=None)
def get_judge_provider(name: str | None = None) -> JudgeProvider:
    """Get the configured judge/evaluator provider instance.

    ``auto`` resolves to the real local provider. Mock providers must be
    selected explicitly for tests.
    """
    resolved_name = name or _JUDGE_PROVIDER_NAME
    if resolved_name == "auto":
        resolved_name = "qwen3_vl"
    elif resolved_name not in _JUDGE_REGISTRY:
        raise ValueError(f"Unknown judge provider: {resolved_name}")
    cls = _JUDGE_REGISTRY[resolved_name]
    if not cls.is_available():
        raise RuntimeError(f"Judge provider is unavailable: {resolved_name}")
    return cls()


def release_all() -> None:
    """Release all cached provider models from memory."""
    for getter in (
        get_llm_provider,
        get_image_provider,
        get_video_provider,
        get_judge_provider,
    ):
        try:
            getter().release()
        except Exception:
            pass
    get_llm_provider.cache_clear()
    get_image_provider.cache_clear()
    get_video_provider.cache_clear()
    get_judge_provider.cache_clear()
    from app.local_models import _cleanup_gpu_memory, release_flux, release_qwen
    release_qwen()
    release_flux()
    _cleanup_gpu_memory()


def list_available_providers() -> dict[str, list[dict[str, Any]]]:
    """List all registered providers and their availability status.
    Useful for UI settings/debug panels.
    """
    result: dict[str, list[dict[str, Any]]] = {"llm": [], "image": [], "video": [], "judge": []}
    for name, cls in _llm_registry().items():
        result["llm"].append({"name": name, "display_name": cls.display_name, "available": cls.is_available()})
    for name, cls in _image_registry().items():
        result["image"].append({"name": name, "display_name": cls.display_name, "available": cls.is_available()})
    for name, cls in _VIDEO_REGISTRY.items():
        result["video"].append({"name": name, "display_name": getattr(cls, "display_name", name), "available": cls.is_available()})
    for name, cls in _JUDGE_REGISTRY.items():
        result["judge"].append({"name": name, "display_name": getattr(cls, "display_name", name), "available": cls.is_available()})
    return result


def get_configured_provider_status() -> dict[str, dict[str, Any]]:
    """Describe configured providers without instantiating or loading them."""
    configured = {
        "llm": (_LLM_PROVIDER_NAME, _llm_registry(), "qwen3_vl"),
        "image": (_IMAGE_PROVIDER_NAME, _image_registry(), "flux"),
        "video": (_VIDEO_PROVIDER_NAME, _VIDEO_REGISTRY, "ffmpeg"),
        "judge": (_JUDGE_PROVIDER_NAME, _JUDGE_REGISTRY, "qwen3_vl"),
    }
    result: dict[str, dict[str, Any]] = {}
    for kind, (configured_name, registry, auto_name) in configured.items():
        name = auto_name if configured_name == "auto" else configured_name
        provider_class = registry.get(name)
        if provider_class is None:
            result[kind] = {
                "name": name,
                "display_name": name,
                "available": False,
                "error": f"Unknown {kind} provider: {name}",
            }
            continue
        try:
            available = bool(provider_class.is_available())
            error = None if available else f"{provider_class.display_name} 的模型或运行依赖不可用"
        except Exception as exc:
            available = False
            error = f"{type(exc).__name__}: {exc}"
        result[kind] = {
            "name": name,
            "display_name": provider_class.display_name,
            "available": available,
            "error": error,
        }
    return result
