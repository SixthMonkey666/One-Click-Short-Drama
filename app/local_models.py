"""Local model runtime for the storyboard-writing step with streaming support."""

from __future__ import annotations

import json
import random as _rng
import re
import site
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator

from app.config import (
    FLUX_ENABLE_VAE_TILING,
    FLUX_GUIDANCE_SCALE,
    FLUX_INFERENCE_STEPS,
    FLUX_MAX_REFERENCE_IMAGES,
    FLUX_MIN_AVAILABLE_GB,
    FLUX_MPS_OFFLOAD_MODE,
    FLUX_MPS_OOM_RETRIES,
    HOT_TOPIC_CACHE_TTL_SECONDS,
    HOT_TOPIC_FETCH_TIMEOUT_SECONDS,
    HOT_TOPIC_SEARCH_ENABLED,
    LOCAL_FLUX_MODEL_DIR,
    LOCAL_QWEN_MODEL_DIR,
    MODEL_CLEANUP_DELAY_SECONDS,
    MODEL_MEMORY_POLL_SECONDS,
    MODEL_QUEUE_WAIT_SECONDS,
    MODEL_STREAM_TIMEOUT_SECONDS,
    MODEL_THREAD_JOIN_TIMEOUT_SECONDS,
    NATIVE_SYSTEM_SITE_PACKAGES,
    NATIVE_USER_SITE_PACKAGES,
    QWEN_ANALYSIS_MAX_NEW_TOKENS,
    QWEN_CHARACTER_MAX_NEW_TOKENS,
    QWEN_DEFAULT_MAX_NEW_TOKENS,
    QWEN_IDEA_MAX_NEW_TOKENS,
    QWEN_IDEA_TEMPERATURE,
    QWEN_IDEA_TOP_P,
    QWEN_JUDGE_MAX_NEW_TOKENS,
    QWEN_JUDGE_MAX_PIXELS,
    QWEN_JUDGE_MIN_AVAILABLE_GB,
    QWEN_JUDGE_MIN_PIXELS,
    QWEN_LOAD_IN_8BIT,
    QWEN_MIN_AVAILABLE_GB,
    QWEN_VIDEO_PROMPT_MAX_NEW_TOKENS,
)


class LocalModelError(RuntimeError):
    """Raised when the configured local storyboard model cannot be used."""


_MODEL_LOCK = threading.RLock()


class ModelQueueStatus:
    """Progress event emitted while a real model task waits for capacity."""

    def __init__(self, message: str, position: int = 0):
        self.message = message
        self.position = position


class _InferenceTaskQueue:
    """FIFO queue that guarantees only one heavyweight model task runs at once."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting: deque[object] = deque()
        self._active = False

    def wait(self, model_name: str) -> Generator[ModelQueueStatus, None, None]:
        ticket = object()
        acquired = False
        with self._condition:
            self._waiting.append(ticket)
            self._condition.notify_all()

        try:
            while True:
                with self._condition:
                    position = list(self._waiting).index(ticket)
                    if position == 0 and not self._active:
                        self._waiting.popleft()
                        self._active = True
                        acquired = True
                        self._condition.notify_all()
                        return
                    tasks_ahead = position + (1 if self._active else 0)

                yield ModelQueueStatus(
                    f"{model_name} 任务排队中，前方还有 {tasks_ahead} 个任务",
                    position=tasks_ahead,
                )
                with self._condition:
                    self._condition.wait(timeout=MODEL_QUEUE_WAIT_SECONDS)
        finally:
            if not acquired:
                with self._condition:
                    try:
                        self._waiting.remove(ticket)
                    except ValueError:
                        pass
                    self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            self._active = False
            self._condition.notify_all()


_INFERENCE_QUEUE = _InferenceTaskQueue()


def _available_memory_bytes() -> int | None:
    """Best-effort estimate of reclaimable memory without extra dependencies."""
    if sys.platform != "darwin":
        return None
    try:
        import subprocess

        output = subprocess.check_output(["vm_stat"], text=True, timeout=2)
        page_size_match = re.search(r"page size of (\d+) bytes", output)
        page_size = int(page_size_match.group(1)) if page_size_match else 4096
        reclaimable = 0
        for label in (
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        ):
            match = re.search(rf"^{label}:\s+(\d+)", output, flags=re.MULTILINE)
            if match:
                reclaimable += int(match.group(1))
        return reclaimable * page_size
    except Exception:
        return None


def _wait_for_available_memory(
    model_name: str, minimum_gb: float
) -> Generator[ModelQueueStatus, None, None]:
    minimum_bytes = minimum_gb * 1024**3
    while True:
        available = _available_memory_bytes()
        if available is None or available >= minimum_bytes:
            return
        current_gb = available / 1024**3
        yield ModelQueueStatus(
            f"内存不足，任务已保留并排队等待：可用 {current_gb:.1f}GB，"
            f"{model_name} 需要至少 {minimum_gb:.1f}GB"
        )
        time.sleep(MODEL_MEMORY_POLL_SECONDS)


def _cleanup_gpu_memory() -> None:
    import gc
    import time

    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    time.sleep(MODEL_CLEANUP_DELAY_SECONDS)


def release_qwen() -> None:
    with _MODEL_LOCK:
        _load_qwen.cache_clear()
    _cleanup_gpu_memory()


def release_flux() -> None:
    with _MODEL_LOCK:
        _load_flux.cache_clear()
    _cleanup_gpu_memory()


def _enable_native_packages() -> None:
    for package_dir in (NATIVE_SYSTEM_SITE_PACKAGES, NATIVE_USER_SITE_PACKAGES):
        if package_dir.exists():
            sp = str(package_dir)
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)
            site.addsitedir(sp)


def _select_device(torch: Any) -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=2)
def _load_qwen(device_override: str | None = None) -> tuple[Any, Any, Any, str]:
    with _MODEL_LOCK:
        release_flux()
        if not LOCAL_QWEN_MODEL_DIR.exists():
            raise LocalModelError(f"找不到本地 Qwen 模型：{LOCAL_QWEN_MODEL_DIR}")
        _enable_native_packages()
        try:
            import torch
            from transformers import AutoProcessor
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError as exc:
            raise LocalModelError(
                "无法加载本地模型依赖。请确保已安装 torch>=2.5 和 transformers>=5.0。"
            ) from exc

        device = device_override or _select_device(torch)
        if device not in {"mps", "cuda", "cpu"}:
            raise ValueError(f"不支持的 Qwen 设备：{device}")

        use_8bit = QWEN_LOAD_IN_8BIT in ("1", "true", "yes")
        if QWEN_LOAD_IN_8BIT == "auto":
            import subprocess
            try:
                mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
                total_gb = mem_bytes / (1024**3)
                use_8bit = total_gb <= 20
            except Exception:
                use_8bit = True

        if device == "mps":
            dtype = torch.float16
        elif device == "cuda" and use_8bit:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        processor = AutoProcessor.from_pretrained(str(LOCAL_QWEN_MODEL_DIR), local_files_only=True)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }

        if device == "mps":
            load_kwargs["attn_implementation"] = "eager"
            load_kwargs["device_map"] = "mps"
        elif device == "cuda" and use_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device

        model = ModelClass.from_pretrained(str(LOCAL_QWEN_MODEL_DIR), **load_kwargs)
        model.eval()
        return torch, processor, model, device


def _extract_json(response: str) -> dict[str, Any]:
    think_match = re.search(r"</think>\s*", response)
    if think_match:
        response = response[think_match.end():]
    candidate = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        raise LocalModelError("Qwen 未按要求返回 JSON，请重试。")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LocalModelError("Qwen 返回的分镜 JSON 无法解析，请重试。") from exc


def _apply_qwen_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        try:
            return processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False}
            )
        except TypeError:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _qwen_generate_stream(
    prompt: str,
    max_new_tokens: int = QWEN_DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Generator[str | ModelQueueStatus, None, str]:
    """Streaming text generation from Qwen. Yields text deltas, returns full response.

    When temperature <= 0, uses greedy decoding (deterministic, good for JSON).
    Set temperature > 0 for creative/varied output.
    """
    inputs = None
    generation_kwargs = None
    thread = None
    slot_acquired = False
    stop_event = threading.Event()
    generated_text = ""
    try:
        for queue_status in _INFERENCE_QUEUE.wait("Qwen"):
            yield queue_status
        slot_acquired = True

        release_flux()
        for memory_status in _wait_for_available_memory("Qwen", QWEN_MIN_AVAILABLE_GB):
            yield memory_status

        torch, processor, model, device = _load_qwen()
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class _StopOnInterrupt(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return stop_event.is_set()

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        rendered_prompt = _apply_qwen_chat_template(processor, messages)
        inputs = processor(text=[rendered_prompt], return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=MODEL_STREAM_TIMEOUT_SECONDS,
        )

        do_sample = temperature > 0
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_StopOnInterrupt()]),
        )
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        generation_error: list[Exception] = []

        def run_generate() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**generation_kwargs)
            except Exception as exc:
                generation_error.append(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=run_generate, name="qwen-generation")
        thread.start()

        in_think_block = False
        think_buffer = ""
        for new_text in streamer:
            if generation_error:
                raise LocalModelError(f"模型生成失败：{generation_error[0]}")
            if "<think>" in new_text:
                in_think_block = True
                parts = new_text.split("<think>", 1)
                before_think = parts[0]
                if before_think:
                    generated_text += before_think
                    yield before_think
                think_buffer = parts[1] if len(parts) > 1 else ""
                continue
            if in_think_block:
                think_buffer += new_text
                if "</think>" in think_buffer:
                    in_think_block = False
                    after_think = think_buffer.split("</think>", 1)[1]
                    think_buffer = ""
                    if after_think:
                        generated_text += after_think
                        yield after_think
                continue
            generated_text += new_text
            yield new_text

        if generation_error:
            raise LocalModelError(f"模型生成失败：{generation_error[0]}")
        return generated_text
    finally:
        stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join()
        if generation_kwargs is not None:
            generation_kwargs.clear()
        if inputs is not None:
            inputs.clear()
        _cleanup_gpu_memory()
        if slot_acquired:
            _INFERENCE_QUEUE.release()


_CHARACTERS_PROMPT_TEMPLATE = """你是专业的角色设计师。从用户的创意描述中，识别出1-2个核心主体（人物或动物）。
为每个主体生成详细的外观描述，用于后续AI图像生成时保持角色一致性。

用户创意：{source_text}

只返回 JSON，不要 Markdown，不要解释。格式必须是：
{{
  "characters": [
    {{
      "name": "角色名称（中文）",
      "type": "人物" 或 "动物",
      "description": "详细中文外观描述：年龄、性别、体型、发型发色、五官特征、服装颜色款式、配饰、显著特征",
      "prompt": "detailed English description for image generation: age, gender, body type, hair, face, clothing colors and style, accessories, distinguishing features, highly detailed"
    }}
  ]
}}

注意：
- 如果创意中没有明确的人物/动物主体，根据创意推断一个合理的主体
- 英文 prompt 要非常详细具体，包含颜色、材质、风格，确保不同镜头间角色外观一致
- 最多返回2个角色，最少1个"""


_ANALYSIS_PROMPT_TEMPLATE = """你是专业的短视频创意分析师。请深度理解用户的创意描述，输出结构化的创意分析，为后续分镜脚本和资产生成提供基础。

用户创意：{source_text}

请分析并提取以下内容，只返回 JSON，不要 Markdown，不要解释：
{{
  "title": "简短有力的作品标题（中文，4-10字）",
  "genre": "类型标签（如：温馨治愈/幽默搞笑/奇幻冒险/悬疑反转/日常感动/科幻未来/古风唯美/赛博朋克/童话奇幻）",
  "theme": "核心主题（一句话概括故事想表达什么）",
  "tone": "情感基调（如：温暖/紧张/欢快/忧伤/神秘/热血/治愈/震撼）",
  "visual_style": "视觉风格描述（如：日系动画风/电影写实/赛博朋克霓虹/水彩童话/复古胶片/国风水墨），用于统一全片画面风格",
  "total_duration_seconds": 建议总时长（秒）,
  "narrative_summary": "故事概要（100-200字，描述起承转合，包含关键情节点）",
  "characters": [
    {{
      "id": "char-01",
      "name": "角色名称（中文）",
      "type": "人物" 或 "动物",
      "description": "中文角色描述：身份、性格、在故事中的作用",
      "appearance": "detailed English visual description for consistent image generation: age, gender, body type, hair color/style, facial features, clothing with specific colors and style, accessories, distinguishing marks, highly detailed character design"
    }}
  ],
  "props": [
    {{
      "id": "prop-01",
      "name": "道具名称（中文）",
      "description": "中文道具描述：外观、在故事中的作用",
      "visual_prompt": "detailed English visual description of the prop: material, color, shape, style, condition"
    }}
  ],
  "environments": [
    {{
      "id": "env-01",
      "name": "场景名称（中文）",
      "description": "中文场景描述：时间、地点、氛围、光线",
      "visual_prompt": "detailed English environment description: location, time of day, lighting, atmosphere, architectural style, weather, color palette, cinematic establishing shot"
    }}
  ]
}}

分析规则：
1. 角色：识别1-3个核心角色，每个人物/动物都要有详细的英文外观描述（appearance），确保后续生图一致性
2. 道具：识别故事中出现的关键道具（0-5个），如信件、武器、魔法物品、交通工具等。没有重要道具可以返回空数组
3. 环境/场景：识别故事发生的主要场景（1-4个），每个场景要有明确的光线和氛围描述
4. 总时长：根据故事复杂度智能建议，简单故事10-20秒，中等20-40秒，复杂叙事30-60秒。最长不超过60秒
5. visual_style 要具体，包含美术风格、色调倾向、光影特征
6. 所有英文描述要足够详细（30词以上），包含颜色、材质、光影、风格关键词"""


_STORYBOARD_PROMPT_TEMPLATE = """你是专业短视频分镜导演。根据已有的创意分析，生成分镜脚本表格。

创意标题：{title}
类型：{genre}
主题：{theme}
基调：{tone}
视觉风格：{visual_style}
建议总时长：{total_duration}秒
故事概要：{narrative_summary}

角色列表：
{characters_list}

道具列表：
{props_list}

场景列表：
{environments_list}

请根据以上分析，设计{shot_count_hint}个分镜镜头，生成完整的分镜表。

镜头设计规则：
1. 镜头数量：{shot_count_hint}个镜头（根据故事复杂度决定，简单故事2-4个，中等4-6个，复杂6-10个）
2. 单镜头时长：2-8秒，开场建立2-3秒，关键动作3-6秒，情绪特写2-4秒，结尾留白2-3秒
3. 所有镜头时长之和应约等于建议总时长（{total_duration}秒），误差不超过5秒
4. 每个镜头必须明确标注：出场角色（用角色ID）、使用道具（用道具ID）、所在场景（用场景ID）
5. 镜头要有视觉变化：景别交替（特写→中景→全景等）、运镜方式变化
6. 英文 prompt 必须非常详细，包含角色外观、道具细节、场景描述、机位、光线、视觉风格关键词
7. camera_angle 取值：特写/近景/中景/全景/远景
8. camera_movement 取值：固定/推/拉/摇左/摇右/上移/下移/跟拍/环绕

只返回 JSON，不要 Markdown，不要解释。格式：
{{
  "script": "中文分镜脚本全文，按镜头顺序描述故事发展（200字以内）",
  "shots": [
    {{
      "number": 1,
      "title": "镜头标题",
      "description": "详细中文画面描述：人物动作、表情、机位、情绪、光线变化",
      "duration_seconds": 3,
      "camera_angle": "中景",
      "camera_movement": "推",
      "character_ids": ["char-01"],
      "prop_ids": ["prop-01"],
      "environment_id": "env-01",
      "prompt": "detailed English image prompt: [character appearance descriptions], [prop details], [environment description], camera angle, camera movement, lighting, [visual style], cinematic film still, high quality"
    }}
  ]
}}"""


def _parse_characters_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_chars = payload.get("characters")
    if not isinstance(raw_chars, list) or len(raw_chars) < 1:
        raise LocalModelError("Qwen 未识别出有效角色，请重试。")
    characters: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_chars[:2], start=1):
        if not isinstance(raw, dict):
            continue
        char_type = str(raw.get("type") or "人物").strip()
        if char_type not in ("人物", "动物"):
            char_type = "人物"
        characters.append(
            {
                "id": f"char-{idx:02d}",
                "name": str(raw.get("name") or f"角色{idx}").strip(),
                "type": char_type,
                "description": str(raw.get("description") or "待补充角色描述。").strip(),
                "prompt": str(raw.get("prompt") or "detailed character").strip(),
                "reference_images": [],
            }
        )
    if not characters:
        raise LocalModelError("Qwen 未生成有效角色，请重试。")
    return characters


def _parse_analysis_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse the creative analysis JSON response from the LLM."""
    title = str(payload.get("title") or "未命名作品").strip()
    genre = str(payload.get("genre") or "日常").strip()
    theme = str(payload.get("theme") or "").strip()
    tone = str(payload.get("tone") or "温暖").strip()
    visual_style = str(payload.get("visual_style") or "cinematic film style").strip()
    try:
        total_dur = int(payload.get("total_duration_seconds", 20))
    except (TypeError, ValueError):
        total_dur = 20
    total_dur = min(60, max(10, total_dur))
    narrative_summary = str(payload.get("narrative_summary") or "").strip()

    def _parse_assets(raw_list, id_prefix, name_key, desc_key, prompt_key, default_prompt):
        items = []
        if not isinstance(raw_list, list):
            return items
        for idx, raw in enumerate(raw_list, start=1):
            if not isinstance(raw, dict):
                continue
            # Model output is untrusted and must never become a filesystem path.
            aid = f"{id_prefix}-{idx:02d}"
            name = str(raw.get(name_key) or f"元素{idx}").strip()
            desc = str(raw.get(desc_key) or "").strip()
            prompt_val = str(raw.get(prompt_key) or default_prompt).strip()
            items.append({
                "id": aid,
                "name": name,
                "description": desc,
                "prompt": prompt_val if prompt_key != "visual_prompt" else prompt_val,
                "visual_prompt": prompt_val if prompt_key == "visual_prompt" else prompt_val,
                "appearance": prompt_val if "appearance" in raw else prompt_val,
                "reference_images": [],
            })
        return items

    raw_chars = payload.get("characters", [])
    characters = []
    if isinstance(raw_chars, list):
        for idx, raw in enumerate(raw_chars[:3], start=1):
            if not isinstance(raw, dict):
                continue
            cid = f"char-{idx:02d}"
            ctype_raw = str(raw.get("type") or "人物").strip()
            ctype = ctype_raw if ctype_raw in ("人物", "动物") else "人物"
            characters.append({
                "id": cid,
                "name": str(raw.get("name") or f"角色{idx}").strip(),
                "type": ctype,
                "description": str(raw.get("description") or "").strip(),
                "appearance": str(raw.get("appearance") or raw.get("prompt") or "detailed character").strip(),
                "reference_images": [],
            })

    props = _parse_assets(
        payload.get("props", []), "prop", "name", "description", "visual_prompt", "detailed prop"
    )
    environments = _parse_assets(
        payload.get("environments", []), "env", "name", "description", "visual_prompt", "detailed environment"
    )

    if not characters:
        characters = [{
            "id": "char-01", "name": "主角", "type": "人物",
            "description": "故事主角", "appearance": "a person, detailed character design",
            "reference_images": [],
        }]
    if not environments:
        environments = [{
            "id": "env-01", "name": "默认场景", "description": "故事发生的场景",
            "visual_prompt": "cinematic environment, detailed setting",
            "prompt": "cinematic environment, detailed setting",
            "appearance": "cinematic environment, detailed setting",
            "reference_images": [],
        }]

    return {
        "title": title,
        "genre": genre,
        "theme": theme,
        "tone": tone,
        "visual_style": visual_style,
        "total_duration_seconds": total_dur,
        "narrative_summary": narrative_summary,
        "characters": characters,
        "props": props,
        "environments": environments,
    }


def _parse_storyboard_response(
    payload: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Parse the storyboard JSON, using analysis data for asset references."""
    raw_shots = payload.get("shots")
    if not isinstance(raw_shots, list) or len(raw_shots) < 2 or len(raw_shots) > 10:
        raise LocalModelError(f"AI 返回的镜头数量异常（{len(raw_shots) if isinstance(raw_shots, list) else 0}个），请重试。")

    char_ids = {c["id"] for c in analysis.get("characters", [])}
    prop_ids = {p["id"] for p in analysis.get("props", [])}
    env_ids = {e["id"] for e in analysis.get("environments", [])}
    default_env_id = analysis.get("environments", [{}])[0].get("id", "env-01") if analysis.get("environments") else "env-01"

    shots: list[dict[str, Any]] = []
    total_dur = 0
    for number, raw_shot in enumerate(raw_shots, start=1):
        if not isinstance(raw_shot, dict):
            raise LocalModelError("AI 返回了无效的镜头数据，请重试。")
        duration = raw_shot.get("duration_seconds", 3)
        try:
            duration = min(8, max(2, int(duration)))
        except (TypeError, ValueError):
            duration = 3
        total_dur += duration

        raw_cids = raw_shot.get("character_ids", [])
        cids = [cid for cid in (raw_cids if isinstance(raw_cids, list) else []) if cid in char_ids]
        if not cids and analysis.get("characters"):
            cids = [analysis["characters"][0]["id"]]

        raw_pids = raw_shot.get("prop_ids", [])
        pids = [pid for pid in (raw_pids if isinstance(raw_pids, list) else []) if pid in prop_ids]

        env_id = raw_shot.get("environment_id", default_env_id)
        if env_id not in env_ids:
            env_id = default_env_id

        shot_prompt = str(raw_shot.get("prompt") or "cinematic film still").strip()
        cam_angle = str(raw_shot.get("camera_angle") or "中景").strip()
        cam_move = str(raw_shot.get("camera_movement") or "固定").strip()

        shots.append({
            "id": f"shot-{number:02d}",
            "number": number,
            "title": str(raw_shot.get("title") or f"镜头 {number}").strip(),
            "description": str(raw_shot.get("description") or "待补充分镜说明。").strip(),
            "duration_seconds": duration,
            "camera_angle": cam_angle,
            "camera_movement": cam_move,
            "character_ids": cids,
            "prop_ids": pids,
            "environment_id": env_id,
            "prompt": shot_prompt,
            "image_versions": [],
            "selected_image": None,
            "review_status": "pending",
            "video_prompt": None,
        })

    script = str(payload.get("script") or "").strip()
    return script, shots


def generate_qwen_analysis_stream(source_text: str) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Streaming creative analysis. Yields events, returns parsed analysis dict."""
    yield {"type": "status", "message": "正在深度分析创意..."}
    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(source_text=source_text)
    full_response = ""
    yield {"type": "text_start"}
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_ANALYSIS_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            yield {"type": "status", "message": delta.message}
            continue
        full_response += delta
        yield {"type": "text_delta", "delta": delta}
    yield {"type": "text_end"}
    payload = _extract_json(full_response)
    analysis = _parse_analysis_response(payload)
    yield {"type": "status", "message": f"创意分析完成：{analysis['title']}"}
    return analysis


def generate_qwen_analysis(source_text: str) -> dict[str, Any]:
    """Non-streaming creative analysis."""
    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(source_text=source_text)
    full_response = ""
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_ANALYSIS_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            continue
        full_response += delta
    payload = _extract_json(full_response)
    return _parse_analysis_response(payload)


def generate_qwen_storyboard(
    source_text: str, analysis: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Non-streaming storyboard generation from analysis."""
    prompt = _build_storyboard_prompt(analysis)
    full_response = ""
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_ANALYSIS_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            continue
        full_response += delta
    payload = _extract_json(full_response)
    return _parse_storyboard_response(payload, analysis)


def generate_qwen_storyboard_stream(
    source_text: str, analysis: dict[str, Any]
) -> Generator[dict[str, Any], None, tuple[str, list[dict[str, Any]]]]:
    """Streaming storyboard generation from analysis. Yields events, returns (script, shots)."""
    yield {"type": "status", "message": "正在生成分镜表..."}
    prompt = _build_storyboard_prompt(analysis)
    full_response = ""
    yield {"type": "text_start"}
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_ANALYSIS_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            yield {"type": "status", "message": delta.message}
            continue
        full_response += delta
        yield {"type": "text_delta", "delta": delta}
    yield {"type": "text_end"}
    payload = _extract_json(full_response)
    script, shots = _parse_storyboard_response(payload, analysis)
    total_dur = sum(s["duration_seconds"] for s in shots)
    yield {"type": "status", "message": f"分镜表已生成，共 {len(shots)} 个镜头，全片约 {total_dur} 秒"}
    return script, shots


def _build_storyboard_prompt(analysis: dict[str, Any]) -> str:
    """Build the storyboard prompt from analysis data."""
    chars = analysis.get("characters", [])
    props = analysis.get("props", [])
    envs = analysis.get("environments", [])

    chars_list = "\n".join(
        f"- {c['id']}: {c['name']}（{c['type']}）- {c['description']} | 外观: {c.get('appearance', c.get('prompt', ''))}"
        for c in chars
    ) if chars else "- char-01: 主角（人物）"

    props_list = "\n".join(
        f"- {p['id']}: {p['name']} - {p['description']}"
        for p in props
    ) if props else "- 无关键道具"

    envs_list = "\n".join(
        f"- {e['id']}: {e['name']} - {e['description']} | 视觉: {e.get('visual_prompt', e.get('prompt', ''))}"
        for e in envs
    ) if envs else "- env-01: 默认场景"

    total_dur = analysis.get("total_duration_seconds", 20)
    if total_dur <= 20:
        shot_hint = "3-4"
    elif total_dur <= 40:
        shot_hint = "4-6"
    else:
        shot_hint = "6-8"

    return _STORYBOARD_PROMPT_TEMPLATE.format(
        title=analysis.get("title", "未命名"),
        genre=analysis.get("genre", "日常"),
        theme=analysis.get("theme", ""),
        tone=analysis.get("tone", "温暖"),
        visual_style=analysis.get("visual_style", "cinematic"),
        total_duration=total_dur,
        narrative_summary=analysis.get("narrative_summary", ""),
        characters_list=chars_list,
        props_list=props_list,
        environments_list=envs_list,
        shot_count_hint=shot_hint,
    )


_hot_topics_cache: dict[str, Any] = {"topics": [], "timestamp": 0}


def _make_ssl_context() -> Any:
    import ssl
    return ssl.create_default_context()


_SSL_CONTEXT = None


def _get_ssl_context() -> Any:
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        _SSL_CONTEXT = _make_ssl_context()
    return _SSL_CONTEXT


def _check_network_available() -> bool:
    if not HOT_TOPIC_SEARCH_ENABLED:
        return False
    try:
        test_hosts = [
            ("114.114.114.114", 53),
            ("223.5.5.5", 53),
        ]
        for host, port in test_hosts:
            try:
                with socket.create_connection((host, port), timeout=2):
                    pass
                return True
            except OSError:
                continue
        return False
    except Exception:
        return False


def _fetch_url_json(url: str, timeout: int | None = None, headers: dict[str, str] | None = None) -> Any:
    timeout = timeout or HOT_TOPIC_FETCH_TIMEOUT_SECONDS
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    ctx = _get_ssl_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_weibo_hot() -> list[str]:
    data = _fetch_url_json(
        "https://weibo.com/ajax/side/hotSearch",
        headers={"Referer": "https://weibo.com/"},
    )
    realtime = data.get("data", {}).get("realtime", [])
    topics = []
    for item in realtime[:20]:
        word = str(item.get("word", "")).strip()
        if word and not item.get("is_ad"):
            topics.append(word)
    return topics


def _fetch_toutiao_hot() -> list[str]:
    data = _fetch_url_json("https://www.toutiao.com/hot-event/hot-board/?origin=toutiao_pc")
    topics = []
    for item in data.get("data", [])[:20]:
        title = str(item.get("Title", "")).strip()
        if title:
            topics.append(title)
    return topics


def _fetch_bilibili_hot() -> list[str]:
    data = _fetch_url_json("https://api.bilibili.com/x/web-interface/search/square?limit=20")
    topics = []
    trending = data.get("data", {}).get("trending", {})
    for item in trending.get("list", [])[:20]:
        kw = str(item.get("keyword") or item.get("show_name") or "").strip()
        if kw:
            topics.append(kw)
    if not topics:
        top_list = trending.get("top_list", [])
        for item in top_list[:20]:
            kw = str(item.get("keyword") or item.get("show_name") or "").strip()
            if kw:
                topics.append(kw)
    return topics


def _fetch_zhihu_hot() -> list[str]:
    data = _fetch_url_json("https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit=20")
    topics = []
    for item in data.get("data", [])[:20]:
        target = item.get("target", {})
        title = str(target.get("title", "")).strip()
        if title:
            topics.append(title)
    return topics


def _fetch_hot_topics(force_refresh: bool = False) -> list[str]:
    import time as _time
    now = _time.time()
    if (
        not force_refresh
        and _hot_topics_cache["topics"]
        and now - _hot_topics_cache["timestamp"] < HOT_TOPIC_CACHE_TTL_SECONDS
    ):
        return list(_hot_topics_cache["topics"])

    if not _check_network_available():
        return []

    fetchers = [
        _fetch_weibo_hot,
        _fetch_toutiao_hot,
        _fetch_bilibili_hot,
        _fetch_zhihu_hot,
    ]
    for fetcher in fetchers:
        try:
            topics = fetcher()
            if topics:
                seen: set[str] = set()
                unique: list[str] = []
                for t in topics:
                    if t not in seen:
                        seen.add(t)
                        unique.append(t)
                _hot_topics_cache["topics"] = unique
                _hot_topics_cache["timestamp"] = now
                return unique
        except Exception:
            continue

    return []


_IDEA_PROMPT_TEMPLATE = """你是一个富有创意的短视频编剧。请构思一个有趣、有画面感、适合10-15秒短视频的创意点子。

创意要求：
1. 题材多样，可以是温馨治愈、幽默搞笑、奇幻冒险、悬疑反转、日常感动、科幻未来、古风唯美等任何类型
2. 要有明确的主角（人或动物）和核心冲突/情感
3. 描述要有画面感，包含场景、动作、情绪，方便后续生成分镜
4. 长度在80-150字之间，用中文描述
5. 只输出创意描述本身，不要加标题、编号、解释或其他内容
6. 每次生成都要完全不同，避免重复常见套路

{inspiration_hint}

请直接给出创意："""


_IDEA_RANDOM_PROMPT_TEMPLATE = """你是一个富有创意的短视频编剧。请随机构思一个有趣、有画面感、适合10-15秒短视频的创意点子。

创意要求：
1. 题材多样，可以是温馨治愈、幽默搞笑、奇幻冒险、悬疑反转、日常感动、科幻未来、古风唯美等任何类型
2. 要有明确的主角（人或动物）和核心冲突/情感
3. 描述要有画面感，包含场景、动作、情绪，方便后续生成分镜
4. 长度在80-150字之间，用中文描述
5. 只输出创意描述本身，不要加标题、编号、解释或其他内容
6. 每次生成都要完全不同，避免重复常见套路

请直接给出创意："""


def _build_random_seed_prompt() -> str:
    seed_genres = [
        "温馨治愈", "幽默搞笑", "奇幻冒险", "悬疑反转", "日常感动",
        "科幻未来", "古风唯美", "都市温情", "童话奇幻", "热血励志",
    ]
    seed_subjects = [
        "一个普通人", "一只小动物", "一位老人", "一个孩子", "一对情侣",
        "一位独居者", "一个旅行者", "一位手艺人", "一只流浪猫", "一个机器人",
    ]
    seed_twists = [
        "意外发现", "久别重逢", "默默守护", "勇敢抉择", "温柔告别",
        "奇妙相遇", "自我发现", "微小善意", "时间错位", "梦境与现实",
    ]
    genre = _rng.choice(seed_genres)
    subject = _rng.choice(seed_subjects)
    twist = _rng.choice(seed_twists)
    hint = f"本次请构思一个「{genre}」风格的故事，主角是{subject}，核心情感/转折围绕「{twist}」展开。"
    return _IDEA_RANDOM_PROMPT_TEMPLATE + "\n\n" + hint


def _build_idea_prompt() -> tuple[str, bool]:
    hot_topics = _fetch_hot_topics()
    if hot_topics:
        selected = _rng.sample(hot_topics, min(3, len(hot_topics)))
        inspiration = "当前网络热门话题参考（可从中汲取灵感或自由发挥）：" + "、".join(selected)
        return _IDEA_PROMPT_TEMPLATE.format(inspiration_hint=inspiration), True
    return _build_random_seed_prompt(), False


_IDEA_COUNT = 3


def _generate_single_idea(prompt: str) -> str:
    full_text = ""
    for delta in _qwen_generate_stream(
        prompt,
        max_new_tokens=QWEN_IDEA_MAX_NEW_TOKENS,
        temperature=QWEN_IDEA_TEMPERATURE,
        top_p=QWEN_IDEA_TOP_P,
    ):
        if isinstance(delta, ModelQueueStatus):
            continue
        cleaned = delta
        if "<think>" in cleaned or "</think>" in cleaned:
            cleaned = ""
        if cleaned:
            full_text += cleaned
    return full_text.strip()


def generate_random_idea_stream() -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
    """Streaming random idea generation producing 3 candidate ideas.

    Yields events:
      - {"type": "status", "message": "..."}     progress updates
      - {"type": "hot_topics", "topics": [...]}   all fetched hot topics
      - {"type": "idea_start", "index": n}        nth idea starting (0-based)
      - {"type": "idea_topics", "index": n, "topics": [...]}  inspiration topics for idea n
      - {"type": "text_delta", "index": n, "delta": "..."}    text chunk for idea n
      - {"type": "idea_end", "index": n}          nth idea finished
    Returns a list of {"text": str, "topics": list[str]} dicts.
    """
    yield {"type": "status", "message": "正在检查网络连接..."}
    net_available = _check_network_available()

    hot_topics: list[str] = []
    used_hot_topics = False
    if net_available:
        yield {"type": "status", "message": "正在联网搜索热门话题..."}
        hot_topics = _fetch_hot_topics()
        if hot_topics:
            used_hot_topics = True
            yield {"type": "hot_topics", "topics": hot_topics[:10]}
            yield {"type": "status", "message": f"已获取{len(hot_topics[:10])}个热点，正在构思3个创意方案..."}
        else:
            yield {"type": "status", "message": "网络获取失败，正在随机构思3个创意方案..."}
    else:
        yield {"type": "status", "message": "当前离线，正在随机构思3个创意方案..."}

    ideas: list[dict[str, Any]] = []
    used_topics_set: set[str] = set()
    topics_pool = list(hot_topics) if hot_topics else []

    for idx in range(_IDEA_COUNT):
        yield {"type": "idea_start", "index": idx}

        idea_topics: list[str] = []
        if used_hot_topics and topics_pool:
            available = [t for t in topics_pool if t not in used_topics_set]
            if not available:
                used_topics_set.clear()
                available = list(topics_pool)
            pick_count = min(2, len(available))
            if pick_count > 0:
                if idx == 0 and len(available) >= 2:
                    idea_topics = available[:2]
                else:
                    idea_topics = _rng.sample(available, pick_count)
                for t in idea_topics:
                    used_topics_set.add(t)
            if idea_topics:
                inspiration = "当前网络热门话题参考（请围绕以下话题构思一个短视频创意，可以结合话题或自由发挥）：" + "、".join(idea_topics)
            else:
                inspiration = ""
            prompt = _IDEA_PROMPT_TEMPLATE.format(inspiration_hint=inspiration)
            yield {"type": "idea_topics", "index": idx, "topics": idea_topics}
        else:
            prompt = _build_random_seed_prompt()
            idea_topics = []
            yield {"type": "idea_topics", "index": idx, "topics": []}

        yield {"type": "status", "message": f"正在生成方案 {idx + 1}/{_IDEA_COUNT}..."}

        full_text = ""
        for delta in _qwen_generate_stream(
            prompt,
            max_new_tokens=QWEN_IDEA_MAX_NEW_TOKENS,
            temperature=QWEN_IDEA_TEMPERATURE,
            top_p=QWEN_IDEA_TOP_P,
        ):
            if isinstance(delta, ModelQueueStatus):
                yield {"type": "status", "message": delta.message}
                continue
            cleaned = delta
            if "<think>" in cleaned or "</think>" in cleaned:
                cleaned = ""
            if cleaned:
                full_text += cleaned
                yield {"type": "text_delta", "index": idx, "delta": cleaned}

        result = full_text.strip()
        ideas.append({"text": result, "topics": idea_topics})
        yield {"type": "idea_end", "index": idx}

    if used_hot_topics:
        yield {"type": "status", "message": f"已生成{_IDEA_COUNT}个创意方案（灵感来自网络热点），请选择一个"}
    else:
        yield {"type": "status", "message": f"已生成{_IDEA_COUNT}个创意方案，请选择一个"}

    return ideas


def generate_random_idea() -> str:
    """Non-streaming random idea generation (returns first idea for backward compat)."""
    gen = generate_random_idea_stream()
    ideas = None
    try:
        while True:
            next(gen)
    except StopIteration as e:
        ideas = e.value
    if ideas and ideas[0].get("text"):
        return ideas[0]["text"]
    return ""


def generate_qwen_characters(source_text: str) -> list[dict[str, Any]]:
    """Non-streaming character extraction (collects all output, returns parsed result)."""
    prompt = _CHARACTERS_PROMPT_TEMPLATE.format(source_text=source_text)
    full_response = ""
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_CHARACTER_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            continue
        full_response += delta
    payload = _extract_json(full_response)
    return _parse_characters_response(payload)


def generate_qwen_characters_stream(source_text: str) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
    """Streaming character extraction. Yields progress events, returns parsed characters."""
    yield {"type": "status", "message": "正在分析创意，识别角色..."}
    prompt = _CHARACTERS_PROMPT_TEMPLATE.format(source_text=source_text)
    full_response = ""
    yield {"type": "text_start"}
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_CHARACTER_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            yield {"type": "status", "message": delta.message}
            continue
        full_response += delta
        yield {"type": "text_delta", "delta": delta}
    yield {"type": "text_end"}
    payload = _extract_json(full_response)
    characters = _parse_characters_response(payload)
    yield {"type": "status", "message": f"识别到 {len(characters)} 个角色"}
    for c in characters:
        yield {"type": "character_found", "character": c}
    return characters


_VIDEO_PROMPT_TEMPLATE = """你是专业的图生视频（Image-to-Video）提示词工程师。根据分镜脚本和关键帧描述，为每个镜头生成一段视频动画提示词，用于指导 AI 将静态关键帧图片转换为动态视频片段。

创意来源：{source_text}

分镜脚本：
{script}

镜头列表（共 {shot_count} 个）：
{shots_context}

角色信息：
{characters_context}

要求：
1. 为每个镜头生成一段英文 motion prompt（30-60词），描述该镜头从静止画面开始的动态变化
2. 提示词应包含：镜头运动（pan/zoom/dolly/tracking）、人物动作、表情变化、环境变化、氛围光效
3. 动作要自然流畅，符合镜头的画面描述和时长（{duration_text}）
4. 镜头之间要有连贯性，上一个镜头的结束状态自然过渡到下一个镜头
5. 动作幅度适中，适合短片段视频，不要过于剧烈
6. 严格输出JSON格式，不要输出任何其他文字

输出JSON格式：
```json
{{
  "video_prompts": [
    {{
      "shot_number": 1,
      "motion_prompt_en": "英文运动提示词",
      "motion_prompt_cn": "中文运动描述（简要说明这个镜头怎么动）",
      "camera_movement": "镜头运动类型，如 slow zoom in / pan left to right / static / dolly forward / tracking shot"
    }}
  ]
}}
```"""


def _parse_video_prompts_response(payload: Any, shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    video_prompts = payload.get("video_prompts", []) if isinstance(payload, dict) else []
    result = []
    default_camera = "subtle zoom in"
    default_motions = [
        "The character gently moves, creating a natural living moment with subtle environmental details coming alive.",
        "A smooth camera movement reveals more of the scene while characters perform their natural actions.",
        "The scene flows naturally with ambient motion, light shifts gently, and characters move with purpose.",
    ]
    for shot in shots:
        vp = None
        for item in video_prompts:
            if isinstance(item, dict) and item.get("shot_number") == shot["number"]:
                vp = item
                break
        if not vp and isinstance(video_prompts, list) and len(video_prompts) >= shot["number"]:
            vp = video_prompts[shot["number"] - 1]
        if vp and isinstance(vp, dict):
            result.append({
                "shot_number": shot["number"],
                "motion_prompt_en": str(vp.get("motion_prompt_en") or default_motions[shot["number"] % len(default_motions)]).strip(),
                "motion_prompt_cn": str(vp.get("motion_prompt_cn") or "自然微动").strip(),
                "camera_movement": str(vp.get("camera_movement") or default_camera).strip(),
            })
        else:
            result.append({
                "shot_number": shot["number"],
                "motion_prompt_en": default_motions[shot["number"] % len(default_motions)],
                "motion_prompt_cn": "自然微动",
                "camera_movement": default_camera,
            })
    return result


def generate_qwen_video_prompts(
    source_text: str,
    script: str,
    shots: list[dict[str, Any]],
    characters: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Non-streaming video prompt generation."""
    shots_context = "\n".join(
        f"镜头{s['number']}：{s['title']}（{s['duration_seconds']}秒）- {s['description']}\n  画面Prompt：{s['prompt']}"
        for s in shots
    )
    chars_context = "\n".join(f"- {c['name']}：{c['description']}" for c in (characters or [])) or "无特定角色"
    duration_text = "、".join(f"镜头{s['number']}{s['duration_seconds']}秒" for s in shots)
    prompt = _VIDEO_PROMPT_TEMPLATE.format(
        source_text=source_text,
        script=script,
        shot_count=len(shots),
        shots_context=shots_context,
        characters_context=chars_context,
        duration_text=duration_text,
    )
    full_response = ""
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_VIDEO_PROMPT_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            continue
        full_response += delta
    payload = _extract_json(full_response)
    return _parse_video_prompts_response(payload, shots)


def generate_qwen_video_prompts_stream(
    source_text: str,
    script: str,
    shots: list[dict[str, Any]],
    characters: list[dict[str, Any]] | None = None,
) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
    """Streaming video prompt generation. Yields progress events, returns video prompts list."""
    yield {"type": "status", "message": "正在生成视频运动提示词..."}
    shots_context = "\n".join(
        f"镜头{s['number']}：{s['title']}（{s['duration_seconds']}秒）- {s['description']}\n  画面Prompt：{s['prompt']}"
        for s in shots
    )
    chars_context = "\n".join(f"- {c['name']}：{c['description']}" for c in (characters or [])) or "无特定角色"
    duration_text = "、".join(f"镜头{s['number']}{s['duration_seconds']}秒" for s in shots)
    prompt = _VIDEO_PROMPT_TEMPLATE.format(
        source_text=source_text,
        script=script,
        shot_count=len(shots),
        shots_context=shots_context,
        characters_context=chars_context,
        duration_text=duration_text,
    )
    full_response = ""
    yield {"type": "text_start"}
    for delta in _qwen_generate_stream(
        prompt, max_new_tokens=QWEN_VIDEO_PROMPT_MAX_NEW_TOKENS
    ):
        if isinstance(delta, ModelQueueStatus):
            yield {"type": "status", "message": delta.message}
            continue
        full_response += delta
        yield {"type": "text_delta", "delta": delta}
    yield {"type": "text_end"}
    payload = _extract_json(full_response)
    video_prompts = _parse_video_prompts_response(payload, shots)
    yield {"type": "status", "message": f"视频提示词已生成，共 {len(video_prompts)} 个镜头"}
    return video_prompts


@lru_cache(maxsize=2)
def _load_flux(offload_mode: str | None = None) -> tuple[Any, str]:
    with _MODEL_LOCK:
        release_qwen()
        if not LOCAL_FLUX_MODEL_DIR.exists():
            raise LocalModelError(f"找不到本地 FLUX 模型：{LOCAL_FLUX_MODEL_DIR}")
        _enable_native_packages()
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise LocalModelError(
                "无法加载 FLUX 模型依赖。请确保已安装 diffusers (git+https://github.com/huggingface/diffusers.git)、torch、transformers。"
            ) from exc

        device = _select_device(torch)

        if device == "mps":
            dtype = torch.float16
        elif device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        _patch_flux_mps_compat(Flux2KleinPipeline, device)
        pipe = Flux2KleinPipeline.from_pretrained(
            str(LOCAL_FLUX_MODEL_DIR),
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

        if FLUX_ENABLE_VAE_TILING and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        if device == "mps":
            resolved_offload_mode = offload_mode or FLUX_MPS_OFFLOAD_MODE
            if resolved_offload_mode == "sequential":
                pipe.enable_sequential_cpu_offload(device="mps")
            elif resolved_offload_mode == "model":
                pipe.enable_model_cpu_offload(device="mps")
            else:
                pipe = pipe.to("mps")
        elif device == "cuda":
            pipe = pipe.to("cuda")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cpu")

        return pipe, device


def _patch_flux_mps_compat(pipeline_cls: Any, device: Any) -> None:
    if device != "mps":
        return
    if getattr(pipeline_cls, "_flux_mps_patched", False):
        return
    original_unpack = pipeline_cls._unpack_latents_with_ids

    @staticmethod
    def _patched_unpack(x, x_ids, height=None, width=None):
        original_device = x.device
        x_cpu = x.cpu()
        x_ids_cpu = x_ids.cpu()
        result = original_unpack(x_cpu, x_ids_cpu, height, width)
        return result.to(original_device)

    pipeline_cls._unpack_latents_with_ids = _patched_unpack
    pipeline_cls._flux_mps_patched = True


class FluxProgressCallback:
    """Callback hook for FLUX pipeline to report denoising step progress."""
    def __init__(self, progress_callback: Any = None, status_callback: Any = None):
        self.step = 0
        self.total_steps = FLUX_INFERENCE_STEPS
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        self.step = step_index + 1
        if self.progress_callback is not None:
            self.progress_callback(self.step, self.total_steps)
        _emit_generation_status(
            self.status_callback,
            "denoising",
            f"正在执行去噪步骤 {self.step}/{self.total_steps}",
            step=self.step,
            total_steps=self.total_steps,
        )
        return callback_kwargs


def generate_flux_image(
    prompt: str,
    output_path: Path,
    width: int = 1024,
    height: int = 576,
    progress_callback: Any = None,
    status_callback: Any = None,
    seed: int | None = None,
    reference_images: list[Path] | None = None,
) -> None:
    """Generate an image using local FLUX.2-klein-4B with optional progress callback."""
    result = None
    image = None
    generator = None
    condition_images = None
    slot_acquired = False
    try:
        _emit_generation_status(status_callback, "queue_wait", "正在等待本地模型推理队列")
        for queue_status in _INFERENCE_QUEUE.wait("FLUX"):
            print(f"[ModelQueue] {queue_status.message}", flush=True)
            _emit_generation_status(status_callback, "queue_wait", queue_status.message)
        slot_acquired = True

        _emit_generation_status(status_callback, "releasing_qwen", "正在释放语言模型占用的内存")
        release_qwen()
        _emit_generation_status(status_callback, "memory_wait", "正在检查可用内存")
        for memory_status in _wait_for_available_memory("FLUX", FLUX_MIN_AVAILABLE_GB):
            print(f"[ModelQueue] {memory_status.message}", flush=True)
            _emit_generation_status(status_callback, "memory_wait", memory_status.message)

        import torch

        resolved_seed = int(seed) % (2**31) if seed is not None else abs(hash(prompt)) % (2**31)
        selected_references = [
            Path(path)
            for path in (reference_images or [])
            if Path(path).is_file()
        ][:FLUX_MAX_REFERENCE_IMAGES]
        if selected_references:
            from PIL import Image

            condition_images = []
            for reference_path in selected_references:
                with Image.open(reference_path) as source:
                    condition_images.append(source.convert("RGB").copy())
            _emit_generation_status(
                status_callback,
                "reference_conditioning",
                f"已加载 {len(condition_images)} 张角色身份参考图",
                reference_count=len(condition_images),
            )
        callback_kwargs = {}
        if progress_callback is not None:
            callback_kwargs["callback_on_step_end"] = progress_callback
            callback_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        attempt_modes = [FLUX_MPS_OFFLOAD_MODE]
        if FLUX_MPS_OFFLOAD_MODE != "sequential" and FLUX_MPS_OOM_RETRIES > 0:
            attempt_modes.extend(["sequential"] * FLUX_MPS_OOM_RETRIES)

        last_error: RuntimeError | None = None
        for attempt_index, offload_mode in enumerate(attempt_modes):
            try:
                _emit_generation_status(
                    status_callback,
                    "model_loading",
                    f"正在加载 FLUX 模型（{offload_mode} CPU offload）",
                    offload_mode=offload_mode,
                    attempt=attempt_index + 1,
                    total_attempts=len(attempt_modes),
                )
                pipe, device = _load_flux(offload_mode)
                _emit_generation_status(
                    status_callback,
                    "model_ready",
                    f"FLUX 模型已就绪，设备 {device}，offload={offload_mode}",
                    device=device,
                    offload_mode=offload_mode,
                )
                gen_device = "cpu" if device == "mps" else device
                generator = torch.Generator(device=gen_device).manual_seed(resolved_seed)
                _emit_generation_status(
                    status_callback,
                    "denoising",
                    f"开始图像去噪，共 {FLUX_INFERENCE_STEPS} 步",
                    step=0,
                    total_steps=FLUX_INFERENCE_STEPS,
                )
                with torch.inference_mode():
                    reference_kwargs = (
                        {"image": condition_images} if condition_images else {}
                    )
                    result = pipe(
                        prompt=prompt,
                        width=width,
                        height=height,
                        guidance_scale=FLUX_GUIDANCE_SCALE,
                        num_inference_steps=FLUX_INFERENCE_STEPS,
                        generator=generator,
                        **reference_kwargs,
                        **callback_kwargs,
                    )
                _emit_generation_status(status_callback, "vae_decode", "去噪完成，正在进行 VAE 解码")
                image = result.images[0]
                _emit_generation_status(status_callback, "saving", f"正在保存图像：{output_path.name}")
                image.save(str(output_path), format="PNG")
                _emit_generation_status(status_callback, "completed", "图像生成并保存完成")
                return
            except RuntimeError as exc:
                last_error = exc
                is_last_attempt = attempt_index == len(attempt_modes) - 1
                if not _is_mps_oom_error(exc) or is_last_attempt:
                    raise
                print(
                    "[FLUX] MPS 内存不足，正在释放模型并切换到 sequential CPU offload 后重试一次。",
                    flush=True,
                )
                _emit_generation_status(
                    status_callback,
                    "oom_retry",
                    "MPS 内存不足，已释放模型，正在切换 sequential CPU offload 重试",
                    attempt=attempt_index + 2,
                    total_attempts=len(attempt_modes),
                )
                image = None
                result = None
                generator = None
                pipe = None
                release_flux()
        if last_error is not None:
            raise last_error
    finally:
        del image, result, generator, condition_images
        _cleanup_gpu_memory()
        if slot_acquired:
            _INFERENCE_QUEUE.release()


def _emit_generation_status(
    callback: Any,
    phase: str,
    message: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback({"phase": phase, "message": message, **details})


def _is_mps_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "mps" in message and (
        "out of memory" in message or "out of-memory" in message
    )


def _build_vlm_judge_prompt(rubric: dict[str, Any], prompt: str, task_type: str) -> str:
    dims = rubric.get("dimensions", [])
    tags = rubric.get("bad_case_tags", [])

    dim_lines = []
    for d in dims:
        anchors = d.get("anchors", {})
        anchor_text = "; ".join(f"{k}分={v}" for k, v in sorted(anchors.items(), key=lambda x: int(x[0])))
        req = "必评" if d.get("required", True) else "选评"
        dim_lines.append(f'- "{d["key"]}" ({d["label"]}, 权重{d.get("weight",1.0)}, {req}): {anchor_text}')

    tag_list = "、".join(tags) if tags else "无"

    return f"""你是专业的多模态内容评审专家。请根据以下评分标准（Rubric）对AI生成的图像进行客观评分。

【原始Prompt/创作意图】：{prompt}
【素材类型】：{task_type}

【评分维度】（每个维度1-5分，严格按锚点描述评分）：
{chr(10).join(dim_lines)}

【Bad Case标签】（如发现以下问题请勾选，没有则留空）：{tag_list}

请仔细观察图像内容，对照每个维度的评分锚点给出1-5的整数分数，并：
1. 对每个评分给出简短的理由（1句话）
2. 如果发现Bad Case问题，列出对应的标签
3. 给出整体评价和置信度（0-1）

只返回JSON，不要Markdown，不要解释，格式必须严格为：
{{
  "dimension_scores": {{"dim_key": score_int, ...}},
  "bad_case_tags": ["标签1", "标签2"],
  "explanation": "整体评价（50-100字中文）",
  "confidence": 0.85
}}

注意：
- 分数必须是1-5的整数，不得给小数
- dimension_scores必须包含所有必评维度，选评维度无明显问题时可给3-4分
- 严格对照锚点描述，不要给人情分
- bad_case_tags只能从给定列表中选择，不要自创标签"""


def qwen_vlm_judge_stream(
    image_paths: list[Path],
    prompt: str,
    task_type: str,
    rubric: dict[str, Any],
    device_override: str | None = None,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Use Qwen3-VL to judge images against a rubric.

    The evaluation shares the same FIFO slot and memory guard as all other
    heavyweight inference tasks. Generator cancellation always releases the
    slot and temporary tensors.
    """
    inputs = None
    image_inputs = None
    video_inputs = None
    generation_kwargs = None
    thread = None
    slot_acquired = False
    stop_event = threading.Event()

    try:
        for queue_status in _INFERENCE_QUEUE.wait("Qwen评测"):
            yield {"type": "status", "message": queue_status.message}
        slot_acquired = True

        release_flux()
        for memory_status in _wait_for_available_memory(
            "Qwen评测", QWEN_JUDGE_MIN_AVAILABLE_GB
        ):
            yield {"type": "status", "message": memory_status.message}

        torch, processor, model, device = _load_qwen(device_override)
        yield {
            "type": "status",
            "message": f"[VLM] 评测设备：{device}；视觉输入上限：{QWEN_JUDGE_MAX_PIXELS} 像素/张",
        }
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        class _StopOnInterrupt(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return stop_event.is_set()

        judge_prompt = _build_vlm_judge_prompt(rubric, prompt, task_type)
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "image": f"file://{Path(path).resolve()}",
                "min_pixels": QWEN_JUDGE_MIN_PIXELS,
                "max_pixels": QWEN_JUDGE_MAX_PIXELS,
            }
            for path in image_paths
            if path and Path(path).exists()
        ]
        content.append({"type": "text", "text": judge_prompt})
        messages = [{"role": "user", "content": content}]

        try:
            rendered_prompt = _apply_qwen_chat_template(processor, messages)
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[rendered_prompt],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(device)
        except Exception as exc:
            raise LocalModelError(f"VLM 视觉处理失败：{exc}") from exc

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=MODEL_STREAM_TIMEOUT_SECONDS,
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=QWEN_JUDGE_MAX_NEW_TOKENS,
            do_sample=False,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_StopOnInterrupt()]),
        )
        generation_error: list[Exception] = []

        def run_generate() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**generation_kwargs)  # noqa: F821
            except Exception as exc:
                generation_error.append(exc)
            finally:
                streamer.end()

        yield {"type": "status", "message": "[VLM] 正在分析图像..."}
        thread = threading.Thread(
            target=run_generate, name="qwen-vlm-judge", daemon=True
        )
        thread.start()

        full_response = ""
        in_think_block = False
        think_buffer = ""
        for new_text in streamer:
            if generation_error:
                raise LocalModelError(f"VLM推理失败：{generation_error[0]}")
            if "<think>" in new_text:
                in_think_block = True
                before_think, think_buffer = new_text.split("<think>", 1)
                if before_think:
                    full_response += before_think
                    yield {"type": "judge_delta", "text": before_think}
                continue
            if in_think_block:
                think_buffer += new_text
                if "</think>" in think_buffer:
                    in_think_block = False
                    after_think = think_buffer.split("</think>", 1)[1]
                    think_buffer = ""
                    if after_think:
                        full_response += after_think
                        yield {"type": "judge_delta", "text": after_think}
                continue
            full_response += new_text
            yield {"type": "judge_delta", "text": new_text}

        thread.join(timeout=MODEL_THREAD_JOIN_TIMEOUT_SECONDS)
        if generation_error:
            raise LocalModelError(f"VLM推理失败：{generation_error[0]}")
        result = _extract_json(full_response)
        return {
            "dimension_scores": result.get("dimension_scores", {}),
            "bad_case_tags": result.get("bad_case_tags", []),
            "explanation": result.get("explanation", ""),
            "confidence": result.get("confidence", 0.5),
            "raw_response": full_response[:2000],
        }
    finally:
        stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=MODEL_THREAD_JOIN_TIMEOUT_SECONDS)
        del inputs, image_inputs, video_inputs, generation_kwargs
        _cleanup_gpu_memory()
        if slot_acquired:
            _INFERENCE_QUEUE.release()


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: int = 8,
    fps: float | None = None,
) -> list[Path]:
    """Extract representative frames from a video using FFmpeg.

    Strategy: evenly spaced frames across video duration, up to max_frames.
    Returns list of extracted frame paths.
    """
    import subprocess

    if not video_path.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
    except Exception:
        duration = 0.0

    if duration <= 0:
        fps_val = fps or 1.0
        n_frames = max_frames
    else:
        n_frames = min(max_frames, max(3, int(duration)))
        fps_val = fps or (n_frames / duration if duration > 0 else 1.0)

    frame_pattern = str(output_dir / "frame_%04d.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"fps={fps_val},scale=512:-1",
             "-q:v", "3", frame_pattern],
            capture_output=True, timeout=60,
        )
    except Exception:
        return []

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    return frames[:max_frames]
