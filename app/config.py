from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    # Keeps the mock-only workflow usable for an offline static smoke test.
    # Normal local/Docker runs install python-dotenv from requirements.txt.
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]


def _env(name: str, default: Any) -> str:
    """Read one environment setting as text."""
    return os.getenv(name, str(default))


def _env_int(name: str, default: int) -> int:
    """Read an integer setting and fail with a precise configuration error."""
    try:
        return int(_env(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _env_float(name: str, default: float) -> float:
    """Read a floating-point setting and fail with a precise configuration error."""
    try:
        return float(_env(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc


def _env_bool(name: str, default: bool) -> bool:
    """Read a conventional boolean environment setting."""
    value = _env(name, "true" if default else "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true 或 false")


APP_PORT = _env_int("APP_PORT", 8503)
DATA_DIR = Path(_env("APP_DATA_DIR", ROOT_DIR / "data")).expanduser().resolve()
ASSETS_DIR = DATA_DIR / "assets"
PROJECT_DB = DATA_DIR / "projects.sqlite"
CHECKPOINT_DB = DATA_DIR / "langgraph_checkpoints.sqlite"
RUBRICS_DIR = ROOT_DIR / "app" / "rubrics"
EVALUATION_DB = PROJECT_DB
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", "qwen3_vl").lower()
JUDGE_MAX_RETRIES = _env_int("JUDGE_MAX_RETRIES", 2)
JUDGE_PROMPT_VERSION = _env("JUDGE_PROMPT_VERSION", "v1.0")
AGENT_MAX_STORYBOARD_RETRIES = _env_int("AGENT_MAX_STORYBOARD_RETRIES", 2)
AGENT_MAX_ASSET_RETRIES = _env_int("AGENT_MAX_ASSET_RETRIES", 2)
AGENT_MAX_VIDEO_PROMPT_RETRIES = _env_int("AGENT_MAX_VIDEO_PROMPT_RETRIES", 2)
AGENT_STORYBOARD_PASS_SCORE = _env_float("AGENT_STORYBOARD_PASS_SCORE", 3.5)


def _default_qwen_model_dir() -> Path:
    """Return the documented Qwen3 model location under the user's home."""
    return Path.home() / "ai_models" / "Qwen3-VL-4B-Instruct"


def _default_flux_model_dir() -> Path:
    """Return the documented FLUX model location under the user's home."""
    return Path.home() / "ai_models" / "FLUX.2-klein-4B"


LOCAL_QWEN_MODEL_DIR = Path(
    _env("LOCAL_QWEN_MODEL_DIR", _default_qwen_model_dir())
).expanduser()
QWEN_LOAD_IN_8BIT = _env("QWEN_LOAD_IN_8BIT", "auto").lower()
LOCAL_FLUX_MODEL_DIR = Path(
    _env("LOCAL_FLUX_MODEL_DIR", _default_flux_model_dir())
).expanduser()
NATIVE_USER_SITE_PACKAGES = Path(
    _env("LOCAL_PYTHON_SITE_PACKAGES", site.getusersitepackages())
)


def _default_system_site_packages() -> str:
    framework_path = Path("/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages")
    if framework_path.exists():
        return str(framework_path)
    for p in site.getsitepackages():
        pp = Path(p)
        if pp.exists() and ".venv" not in str(pp) and "virtualenv" not in str(pp):
            return p
    return str(framework_path)


NATIVE_SYSTEM_SITE_PACKAGES = Path(
    _env("LOCAL_SYSTEM_SITE_PACKAGES", _default_system_site_packages())
)

LLM_PROVIDER = _env("LLM_PROVIDER", "qwen3_vl").lower()
IMAGE_PROVIDER = _env("IMAGE_PROVIDER", "flux").lower()
VIDEO_PROVIDER = _env("VIDEO_PROVIDER", "ffmpeg").lower()

# OpenAI-compatible endpoints work with hosted APIs and local servers such as
# Ollama, vLLM, LM Studio and LocalAI. Keys may be empty for trusted local
# endpoints. Provider-specific aliases keep text and image services independent.
OPENAI_COMPATIBLE_API_KEY = _env("OPENAI_COMPATIBLE_API_KEY", "")
OPENAI_COMPATIBLE_LLM_BASE_URL = _env(
    "OPENAI_COMPATIBLE_LLM_BASE_URL", "http://127.0.0.1:11434/v1"
).rstrip("/")
OPENAI_COMPATIBLE_LLM_API_KEY = _env(
    "OPENAI_COMPATIBLE_LLM_API_KEY", OPENAI_COMPATIBLE_API_KEY
)
OPENAI_COMPATIBLE_LLM_MODEL = _env("OPENAI_COMPATIBLE_LLM_MODEL", "")
OPENAI_COMPATIBLE_LLM_TIMEOUT_SECONDS = _env_float(
    "OPENAI_COMPATIBLE_LLM_TIMEOUT_SECONDS", 300.0
)
OPENAI_COMPATIBLE_LLM_SYSTEM_PROMPT = _env(
    "OPENAI_COMPATIBLE_LLM_SYSTEM_PROMPT",
    "You are the structured planning model for an AI storyboard workflow. "
    "Follow the requested language and JSON schema exactly.",
)

OPENAI_COMPATIBLE_IMAGE_BASE_URL = _env(
    "OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://127.0.0.1:8080/v1"
).rstrip("/")
OPENAI_COMPATIBLE_IMAGE_API_KEY = _env(
    "OPENAI_COMPATIBLE_IMAGE_API_KEY", OPENAI_COMPATIBLE_API_KEY
)
OPENAI_COMPATIBLE_IMAGE_MODEL = _env("OPENAI_COMPATIBLE_IMAGE_MODEL", "")
OPENAI_COMPATIBLE_IMAGE_TIMEOUT_SECONDS = _env_float(
    "OPENAI_COMPATIBLE_IMAGE_TIMEOUT_SECONDS", 600.0
)
OPENAI_COMPATIBLE_IMAGE_RESPONSE_FORMAT = _env(
    "OPENAI_COMPATIBLE_IMAGE_RESPONSE_FORMAT", "b64_json"
).lower()
if OPENAI_COMPATIBLE_IMAGE_RESPONSE_FORMAT not in {"b64_json", "url"}:
    raise ValueError(
        "OPENAI_COMPATIBLE_IMAGE_RESPONSE_FORMAT 必须是 b64_json 或 url"
    )
OPENAI_COMPATIBLE_IMAGE_MAX_DOWNLOAD_MB = _env_int(
    "OPENAI_COMPATIBLE_IMAGE_MAX_DOWNLOAD_MB", 30
)

# Optional Python plugin entry points, for example:
#   CUSTOM_LLM_PROVIDER_CLASS=my_backends.llm:MyLLMProvider
# The module is imported only when the corresponding provider is selected.
CUSTOM_LLM_PROVIDER_CLASS = _env("CUSTOM_LLM_PROVIDER_CLASS", "")
CUSTOM_IMAGE_PROVIDER_CLASS = _env("CUSTOM_IMAGE_PROVIDER_CLASS", "")

MPS_ENABLE_FALLBACK = _env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
MPS_HIGH_WATERMARK_RATIO = _env("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
MPS_LOW_WATERMARK_RATIO = _env("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", MPS_ENABLE_FALLBACK)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", MPS_HIGH_WATERMARK_RATIO)
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", MPS_LOW_WATERMARK_RATIO)

QWEN_MIN_AVAILABLE_GB = _env_float("QWEN_MIN_AVAILABLE_GB", 3.0)
FLUX_MIN_AVAILABLE_GB = _env_float("FLUX_MIN_AVAILABLE_GB", 4.0)
FLUX_MPS_OFFLOAD_MODE = _env("FLUX_MPS_OFFLOAD_MODE", "model").lower()
if FLUX_MPS_OFFLOAD_MODE not in {"none", "model", "sequential"}:
    raise ValueError("FLUX_MPS_OFFLOAD_MODE 必须是 none、model 或 sequential")
FLUX_MPS_OOM_RETRIES = _env_int("FLUX_MPS_OOM_RETRIES", 1)
if FLUX_MPS_OOM_RETRIES < 0:
    raise ValueError("FLUX_MPS_OOM_RETRIES 不能小于 0")
FLUX_ENABLE_VAE_TILING = _env_bool("FLUX_ENABLE_VAE_TILING", True)
MODEL_QUEUE_WAIT_SECONDS = _env_float("MODEL_QUEUE_WAIT_SECONDS", 2.0)
MODEL_MEMORY_POLL_SECONDS = _env_float("MODEL_MEMORY_POLL_SECONDS", 2.0)
MODEL_STREAM_TIMEOUT_SECONDS = _env_float("MODEL_STREAM_TIMEOUT_SECONDS", 300.0)
MODEL_THREAD_JOIN_TIMEOUT_SECONDS = _env_float(
    "MODEL_THREAD_JOIN_TIMEOUT_SECONDS", 5.0
)
MODEL_CLEANUP_DELAY_SECONDS = _env_float("MODEL_CLEANUP_DELAY_SECONDS", 0.2)

QWEN_DEFAULT_MAX_NEW_TOKENS = _env_int("QWEN_DEFAULT_MAX_NEW_TOKENS", 800)
QWEN_ANALYSIS_MAX_NEW_TOKENS = _env_int("QWEN_ANALYSIS_MAX_NEW_TOKENS", 2000)
QWEN_IDEA_MAX_NEW_TOKENS = _env_int("QWEN_IDEA_MAX_NEW_TOKENS", 220)
QWEN_IDEA_TEMPERATURE = _env_float("QWEN_IDEA_TEMPERATURE", 0.95)
QWEN_IDEA_TOP_P = _env_float("QWEN_IDEA_TOP_P", 0.92)
QWEN_CHARACTER_MAX_NEW_TOKENS = _env_int("QWEN_CHARACTER_MAX_NEW_TOKENS", 600)
QWEN_VIDEO_PROMPT_MAX_NEW_TOKENS = _env_int(
    "QWEN_VIDEO_PROMPT_MAX_NEW_TOKENS", 1500
)
QWEN_JUDGE_MAX_NEW_TOKENS = _env_int("QWEN_JUDGE_MAX_NEW_TOKENS", 1024)
QWEN_JUDGE_MIN_AVAILABLE_GB = _env_float("QWEN_JUDGE_MIN_AVAILABLE_GB", 4.0)
QWEN_JUDGE_MAX_PIXELS = _env_int("QWEN_JUDGE_MAX_PIXELS", 384 * 384)
QWEN_JUDGE_MIN_PIXELS = _env_int("QWEN_JUDGE_MIN_PIXELS", 224 * 224)
QWEN_JUDGE_MAX_INPUT_IMAGES = _env_int("QWEN_JUDGE_MAX_INPUT_IMAGES", 3)
QWEN_JUDGE_CPU_FALLBACK = _env_bool("QWEN_JUDGE_CPU_FALLBACK", True)
QWEN_JUDGE_RELEASE_EACH_ITEM = _env_bool("QWEN_JUDGE_RELEASE_EACH_ITEM", True)
if QWEN_JUDGE_MIN_PIXELS <= 0 or QWEN_JUDGE_MAX_PIXELS < QWEN_JUDGE_MIN_PIXELS:
    raise ValueError("QWEN_JUDGE_MAX_PIXELS 必须大于等于正数 QWEN_JUDGE_MIN_PIXELS")
if QWEN_JUDGE_MAX_INPUT_IMAGES < 1:
    raise ValueError("QWEN_JUDGE_MAX_INPUT_IMAGES 必须至少为 1")

FLUX_GUIDANCE_SCALE = _env_float("FLUX_GUIDANCE_SCALE", 1.0)
FLUX_INFERENCE_STEPS = _env_int("FLUX_INFERENCE_STEPS", 4)
FLUX_MAX_REFERENCE_IMAGES = _env_int("FLUX_MAX_REFERENCE_IMAGES", 2)
if FLUX_MAX_REFERENCE_IMAGES < 0:
    raise ValueError("FLUX_MAX_REFERENCE_IMAGES 不能小于 0")
JUDGE_MAX_VIDEO_FRAMES = _env_int("JUDGE_MAX_VIDEO_FRAMES", 6)
ELO_INITIAL_RATING = _env_float("ELO_INITIAL_RATING", 1000.0)
ELO_K_FACTOR = _env_float("ELO_K_FACTOR", 32.0)

HOT_TOPIC_SEARCH_ENABLED = _env_bool("HOT_TOPIC_SEARCH_ENABLED", True)
HOT_TOPIC_FETCH_TIMEOUT_SECONDS = _env_float(
    "HOT_TOPIC_FETCH_TIMEOUT_SECONDS", 5.0
)
HOT_TOPIC_CACHE_TTL_SECONDS = _env_int("HOT_TOPIC_CACHE_TTL_SECONDS", 300)

LANGFUSE_PUBLIC_KEY = _env("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = _env("LANGFUSE_SECRET_KEY", "")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

IMAGE_RESOLUTION_PRESETS: dict[str, dict[str, int | str]] = {
    "512x288 (极速草稿)": {"width": 512, "height": 288, "label": "极速草稿", "est_time": "~2分钟/张"},
    "768x432 (快速预览)": {"width": 768, "height": 432, "label": "快速预览", "est_time": "~3分钟/张"},
    "1024x576 (标准)": {"width": 1024, "height": 576, "label": "标准", "est_time": "~5分钟/张"},
    "1280x720 (高清)": {"width": 1280, "height": 720, "label": "高清", "est_time": "~8分钟/张"},
}

DEFAULT_IMAGE_RESOLUTION = _env(
    "DEFAULT_IMAGE_RESOLUTION", "768x432 (快速预览)"
)
if DEFAULT_IMAGE_RESOLUTION not in IMAGE_RESOLUTION_PRESETS:
    raise ValueError(
        "DEFAULT_IMAGE_RESOLUTION 必须是 IMAGE_RESOLUTION_PRESETS 中的预设名称"
    )

CHARACTER_REF_SIZE = (512, 768)
CHARACTER_FACE_SIZE = (768, 768)
PROP_REF_SIZE = (768, 768)
ENVIRONMENT_REF_SIZE = (1280, 720)
GENERATED_IMAGE_MIN_BYTES = _env_int("GENERATED_IMAGE_MIN_BYTES", 500)

CHARACTER_VIEWS = [
    {"key": "front", "label": "正面全身", "prompt_suffix": "full body front view, standing, facing camera, T-pose, white background, character design sheet, consistent lighting", "size": CHARACTER_REF_SIZE},
    {"key": "side", "label": "侧面全身", "prompt_suffix": "full body side view, profile, standing, white background, character design sheet, consistent lighting", "size": CHARACTER_REF_SIZE},
    {"key": "back", "label": "背面全身", "prompt_suffix": "full body back view, standing, facing away, white background, character design sheet, consistent lighting", "size": CHARACTER_REF_SIZE},
    {"key": "face", "label": "面部特写", "prompt_suffix": "face close-up portrait, detailed facial features, looking at camera, white background, character design sheet", "size": CHARACTER_FACE_SIZE},
]


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    RUBRICS_DIR.mkdir(parents=True, exist_ok=True)
