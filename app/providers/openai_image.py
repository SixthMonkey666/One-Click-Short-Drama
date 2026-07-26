"""OpenAI-compatible image provider for local servers and hosted APIs."""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from app.config import (
    OPENAI_COMPATIBLE_IMAGE_API_KEY,
    OPENAI_COMPATIBLE_IMAGE_BASE_URL,
    OPENAI_COMPATIBLE_IMAGE_MAX_DOWNLOAD_MB,
    OPENAI_COMPATIBLE_IMAGE_MODEL,
    OPENAI_COMPATIBLE_IMAGE_RESPONSE_FORMAT,
    OPENAI_COMPATIBLE_IMAGE_TIMEOUT_SECONDS,
)
from app.providers.base import ImageProvider, StatusCallback


class OpenAICompatibleImageError(RuntimeError):
    """Raised when an image endpoint fails or returns an invalid image."""


class OpenAICompatibleImageProvider(ImageProvider):
    """Use an images/generations and images/edits compatible endpoint."""

    name = "openai_compatible"
    display_name = "OpenAI-compatible 图片模型"
    default_steps = 1
    supports_reference_images = True

    @classmethod
    def is_available(cls) -> bool:
        return bool(
            OPENAI_COMPATIBLE_IMAGE_BASE_URL
            and OPENAI_COMPATIBLE_IMAGE_MODEL
        )

    @staticmethod
    def _headers() -> dict[str, str]:
        if not OPENAI_COMPATIBLE_IMAGE_API_KEY:
            return {}
        return {
            "Authorization": f"Bearer {OPENAI_COMPATIBLE_IMAGE_API_KEY}"
        }

    @staticmethod
    def _status(
        callback: StatusCallback | None,
        phase: str,
        message: str,
    ) -> None:
        if callback:
            callback({"phase": phase, "message": message})

    def _request(
        self,
        prompt: str,
        width: int,
        height: int,
        reference_images: list[Path],
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise OpenAICompatibleImageError(
                "OpenAI-compatible Provider 需要安装 httpx。"
            ) from exc

        common = {
            "model": OPENAI_COMPATIBLE_IMAGE_MODEL,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "response_format": OPENAI_COMPATIBLE_IMAGE_RESPONSE_FORMAT,
        }
        try:
            with httpx.Client(
                timeout=OPENAI_COMPATIBLE_IMAGE_TIMEOUT_SECONDS
            ) as client:
                if reference_images:
                    endpoint = (
                        f"{OPENAI_COMPATIBLE_IMAGE_BASE_URL}/images/edits"
                    )
                    files = [
                        (
                            "image",
                            (
                                image.name,
                                image.read_bytes(),
                                "image/png",
                            ),
                        )
                        for image in reference_images
                        if image.is_file()
                    ]
                    response = client.post(
                        endpoint,
                        headers=self._headers(),
                        data={key: str(value) for key, value in common.items()},
                        files=files,
                    )
                else:
                    endpoint = (
                        f"{OPENAI_COMPATIBLE_IMAGE_BASE_URL}"
                        "/images/generations"
                    )
                    response = client.post(
                        endpoint,
                        headers={
                            **self._headers(),
                            "Content-Type": "application/json",
                        },
                        json=common,
                    )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:500]
            raise OpenAICompatibleImageError(
                f"图片 API 请求失败（HTTP {status}）：{detail}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise OpenAICompatibleImageError(
                f"图片 API 响应无效：{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise OpenAICompatibleImageError("图片 API 未返回 JSON 对象。")
        return payload

    def _decode_image(self, payload: dict[str, Any]) -> bytes:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise OpenAICompatibleImageError("图片 API 返回缺少 data[0]。")
        item = data[0]
        if not isinstance(item, dict):
            raise OpenAICompatibleImageError("图片 API 返回格式无效。")
        encoded = item.get("b64_json")
        if encoded:
            try:
                content = base64.b64decode(str(encoded), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise OpenAICompatibleImageError(
                    "图片 API 返回了无效的 base64。"
                ) from exc
            return content
        url = item.get("url")
        if not url:
            raise OpenAICompatibleImageError(
                "图片 API 未返回 b64_json 或 url。"
            )
        try:
            import httpx

            limit = OPENAI_COMPATIBLE_IMAGE_MAX_DOWNLOAD_MB * 1024 * 1024
            with httpx.Client(
                timeout=OPENAI_COMPATIBLE_IMAGE_TIMEOUT_SECONDS
            ) as client:
                with client.stream("GET", str(url)) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        content.extend(chunk)
                        if len(content) > limit:
                            raise OpenAICompatibleImageError(
                                "图片 API 下载结果超过大小限制。"
                            )
            return bytes(content)
        except httpx.HTTPError as exc:
            raise OpenAICompatibleImageError(
                f"无法下载图片 API 结果：{exc}"
            ) from exc

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
        del seed  # OpenAI's standard image schema has no portable seed field.
        references = [
            path for path in (reference_images or []) if path.is_file()
        ]
        if progress_callback:
            progress_callback(0, 1)
        phase = "api_image_edit" if references else "api_image_generation"
        self._status(
            status_callback,
            phase,
            "正在调用图片 API（含角色参考图）..."
            if references
            else "正在调用图片生成 API...",
        )
        payload = self._request(prompt, width, height, references)
        content = self._decode_image(payload)
        try:
            with Image.open(io.BytesIO(content)) as image:
                rendered = image.convert("RGB")
                if rendered.size != (width, height):
                    rendered = rendered.resize(
                        (width, height), Image.Resampling.LANCZOS
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                rendered.save(output_path, format="PNG")
        except (OSError, ValueError) as exc:
            raise OpenAICompatibleImageError(
                "图片 API 返回的内容不是有效图像。"
            ) from exc
        if progress_callback:
            progress_callback(1, 1)
        self._status(status_callback, "completed", "图片 API 生成完成")
