"""OpenAI-compatible text provider for hosted APIs and local model servers."""

from __future__ import annotations

import json
from typing import Any, Generator

from app.config import (
    OPENAI_COMPATIBLE_LLM_API_KEY,
    OPENAI_COMPATIBLE_LLM_BASE_URL,
    OPENAI_COMPATIBLE_LLM_MODEL,
    OPENAI_COMPATIBLE_LLM_SYSTEM_PROMPT,
    OPENAI_COMPATIBLE_LLM_TIMEOUT_SECONDS,
    QWEN_ANALYSIS_MAX_NEW_TOKENS,
    QWEN_CHARACTER_MAX_NEW_TOKENS,
    QWEN_IDEA_MAX_NEW_TOKENS,
    QWEN_IDEA_TEMPERATURE,
    QWEN_IDEA_TOP_P,
    QWEN_VIDEO_PROMPT_MAX_NEW_TOKENS,
)
from app.local_models import (
    _ANALYSIS_PROMPT_TEMPLATE,
    _CHARACTERS_PROMPT_TEMPLATE,
    _VIDEO_PROMPT_TEMPLATE,
    _build_random_seed_prompt,
    _build_storyboard_prompt,
    _extract_json,
    _parse_analysis_response,
    _parse_characters_response,
    _parse_storyboard_response,
    _parse_video_prompts_response,
)
from app.providers.base import LLMProvider


class OpenAICompatibleError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint returns an invalid response."""


class OpenAICompatibleLLMProvider(LLMProvider):
    """Use any chat-completions compatible local server or hosted API."""

    name = "openai_compatible"
    display_name = "OpenAI-compatible 文本模型"

    @classmethod
    def is_available(cls) -> bool:
        return bool(
            OPENAI_COMPATIBLE_LLM_BASE_URL and OPENAI_COMPATIBLE_LLM_MODEL
        )

    @staticmethod
    def _headers() -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if OPENAI_COMPATIBLE_LLM_API_KEY:
            headers["Authorization"] = (
                f"Bearer {OPENAI_COMPATIBLE_LLM_API_KEY}"
            )
        return headers

    def _stream_text(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, str]:
        try:
            import httpx
        except ImportError as exc:
            raise OpenAICompatibleError(
                "OpenAI-compatible Provider 需要安装 httpx。"
            ) from exc

        payload = {
            "model": OPENAI_COMPATIBLE_LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": OPENAI_COMPATIBLE_LLM_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        full_text = ""
        endpoint = (
            f"{OPENAI_COMPATIBLE_LLM_BASE_URL}/chat/completions"
        )
        try:
            with httpx.Client(
                timeout=OPENAI_COMPATIBLE_LLM_TIMEOUT_SECONDS
            ) as client:
                with client.stream(
                    "POST",
                    endpoint,
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            delta = (
                                event.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content")
                            )
                        except (IndexError, TypeError, json.JSONDecodeError):
                            continue
                        if delta:
                            text = str(delta)
                            full_text += text
                            yield text
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:500]
            raise OpenAICompatibleError(
                f"文本 API 请求失败（HTTP {status}）：{detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenAICompatibleError(
                f"无法连接文本模型 API：{type(exc).__name__}: {exc}"
            ) from exc
        if not full_text.strip():
            raise OpenAICompatibleError("文本模型 API 返回了空内容。")
        return full_text

    def generate_idea_stream(
        self,
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        ideas: list[dict[str, Any]] = []
        yield {"type": "status", "message": "正在通过文本模型构思 3 个方案..."}
        for index in range(3):
            yield {"type": "idea_start", "index": index}
            yield {"type": "idea_topics", "index": index, "topics": []}
            prompt = _build_random_seed_prompt()
            full_text = ""
            for delta in self._stream_text(
                prompt,
                max_tokens=QWEN_IDEA_MAX_NEW_TOKENS,
                temperature=QWEN_IDEA_TEMPERATURE,
                top_p=QWEN_IDEA_TOP_P,
            ):
                full_text += delta
                yield {
                    "type": "text_delta",
                    "index": index,
                    "delta": delta,
                }
            ideas.append({"text": full_text.strip(), "topics": []})
            yield {"type": "idea_end", "index": index}
        yield {"type": "status", "message": "已生成 3 个创意方案，请选择一个"}
        return ideas

    def _structured_stream(
        self,
        prompt: str,
        *,
        status: str,
        max_tokens: int,
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        yield {"type": "status", "message": status}
        yield {"type": "text_start"}
        response = ""
        for delta in self._stream_text(prompt, max_tokens=max_tokens):
            response += delta
            yield {"type": "text_delta", "delta": delta}
        yield {"type": "text_end"}
        return _extract_json(response)

    def generate_analysis_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        payload = yield from self._structured_stream(
            _ANALYSIS_PROMPT_TEMPLATE.format(source_text=source_text),
            status="正在通过文本模型分析创意...",
            max_tokens=QWEN_ANALYSIS_MAX_NEW_TOKENS,
        )
        result = _parse_analysis_response(payload)
        yield {"type": "status", "message": f"创意分析完成：{result['title']}"}
        return result

    def generate_characters_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        payload = yield from self._structured_stream(
            _CHARACTERS_PROMPT_TEMPLATE.format(source_text=source_text),
            status="正在通过文本模型识别角色...",
            max_tokens=QWEN_CHARACTER_MAX_NEW_TOKENS,
        )
        result = _parse_characters_response(payload)
        yield {"type": "status", "message": f"识别到 {len(result)} 个角色"}
        for character in result:
            yield {"type": "character_found", "character": character}
        return result

    def generate_storyboard_stream(
        self, source_text: str, analysis: dict[str, Any]
    ) -> Generator[
        dict[str, Any], None, tuple[str, list[dict[str, Any]]]
    ]:
        payload = yield from self._structured_stream(
            _build_storyboard_prompt(analysis),
            status="正在通过文本模型生成分镜表...",
            max_tokens=QWEN_ANALYSIS_MAX_NEW_TOKENS,
        )
        result = _parse_storyboard_response(payload, analysis)
        yield {
            "type": "status",
            "message": f"分镜表已生成，共 {len(result[1])} 个镜头",
        }
        return result

    def generate_video_prompts_stream(
        self,
        source_text: str,
        script: str,
        shots: list[dict[str, Any]],
        characters: list[dict[str, Any]] | None = None,
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        shots_context = "\n".join(
            f"镜头{s['number']}：{s['title']}（{s['duration_seconds']}秒）"
            f"- {s['description']}\n  画面Prompt：{s['prompt']}"
            for s in shots
        )
        chars_context = "\n".join(
            f"- {c['name']}：{c['description']}"
            for c in (characters or [])
        ) or "无特定角色"
        duration_text = "、".join(
            f"镜头{s['number']}{s['duration_seconds']}秒" for s in shots
        )
        prompt = _VIDEO_PROMPT_TEMPLATE.format(
            source_text=source_text,
            script=script,
            shot_count=len(shots),
            shots_context=shots_context,
            characters_context=chars_context,
            duration_text=duration_text,
        )
        payload = yield from self._structured_stream(
            prompt,
            status="正在通过文本模型生成视频运动提示词...",
            max_tokens=QWEN_VIDEO_PROMPT_MAX_NEW_TOKENS,
        )
        result = _parse_video_prompts_response(payload, shots)
        yield {
            "type": "status",
            "message": f"视频提示词已生成，共 {len(result)} 个镜头",
        }
        return result
