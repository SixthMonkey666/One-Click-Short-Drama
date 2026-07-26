"""Adapters layer: bridges between workflow orchestration and model providers."""

from __future__ import annotations

import hashlib
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator

from app.config import (
    CHARACTER_VIEWS,
    DEFAULT_IMAGE_RESOLUTION,
    ENVIRONMENT_REF_SIZE,
    FLUX_MAX_REFERENCE_IMAGES,
    GENERATED_IMAGE_MIN_BYTES,
    IMAGE_RESOLUTION_PRESETS,
    PROP_REF_SIZE,
)
from app.providers import (
    get_image_provider,
    get_llm_provider,
    get_video_provider,
)
from app.repository import (
    asset_directory,
    character_asset_directory,
    environment_asset_directory,
    get_project,
    prop_asset_directory,
)

CHARACTER_CONSISTENCY_SYSTEM_PROMPT = """
SYSTEM DIRECTIVE — CHARACTER IDENTITY LOCK (highest priority):
Every named character must preserve exactly the same identity across all images.
Treat the canonical identity description and supplied reference images as immutable facts.
Keep the same species, apparent age, face geometry, eye color, hair or fur color and pattern,
body proportions, skin tone, signature clothing colors and materials, accessories, and unique marks.
You may change only pose, expression, camera angle, framing, lighting, and action requested by the shot.
Do not redesign, restyle, recolor, replace clothing, add or remove accessories, or merge identities.
When a reference image is supplied, reproduce that character's visual identity rather than inventing a new one.
""".strip()


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def build_character_identity_prompt(character: dict[str, Any]) -> str:
    """Build one canonical, reusable identity block from editable character fields."""
    character_id = str(character.get("id") or "character")
    identity_token = re.sub(r"[^A-Za-z0-9]+", "_", character_id).strip("_").upper()
    name = str(character.get("name") or character_id)
    kind = str(character.get("type") or "character")
    appearance = str(
        character.get("appearance")
        or character.get("prompt")
        or character.get("description")
        or "distinct, stable character design"
    ).strip()
    return (
        f"CANONICAL IDENTITY [{identity_token}] — {name} ({kind}): {appearance}. "
        "This identity block is immutable in every view and every shot."
    )


def _canonical_character_reference(character: dict[str, Any]) -> Path | None:
    images = character.get("reference_images", [])
    candidate = next(
        (image for image in images if image.get("view") == "front"),
        images[0] if images else None,
    )
    if not candidate or not candidate.get("path"):
        return None
    path = Path(candidate["path"])
    return path if path.is_file() else None


def _keyframe_character_consistency(
    project_id: str,
    shot: dict[str, Any],
    analysis: dict[str, Any] | None,
) -> tuple[str, list[Path], int, list[dict[str, Any]]]:
    if not analysis:
        return "", [], _stable_seed(project_id, shot.get("id"), "keyframe"), []
    by_id = {
        str(character.get("id")): character
        for character in analysis.get("characters", [])
    }
    characters = [
        by_id[character_id]
        for character_id in map(str, shot.get("character_ids", []))
        if character_id in by_id
    ]
    if not characters:
        return "", [], _stable_seed(project_id, shot.get("id"), "keyframe"), []

    identity_blocks: list[str] = []
    references: list[Path] = []
    reference_names: list[str] = []
    metadata: list[dict[str, Any]] = []
    for character in characters:
        identity_prompt = build_character_identity_prompt(character)
        identity_seed = _stable_seed(project_id, character.get("id"), "identity")
        character["identity_prompt"] = identity_prompt
        character["identity_seed"] = identity_seed
        identity_blocks.append(identity_prompt)
        reference = _canonical_character_reference(character)
        if reference is not None and len(references) < FLUX_MAX_REFERENCE_IMAGES:
            references.append(reference)
            reference_names.append(str(character.get("name") or character.get("id")))
        metadata.append({
            "character_id": character.get("id"),
            "name": character.get("name"),
            "identity_seed": identity_seed,
            "reference_image": str(reference) if reference else None,
        })

    reference_map = "\n".join(
        f"REFERENCE IMAGE {index}: canonical visual identity for {name}."
        for index, name in enumerate(reference_names, start=1)
    )
    prompt = "\n".join(filter(None, [
        CHARACTER_CONSISTENCY_SYSTEM_PROMPT,
        "IDENTITIES PRESENT IN THIS SHOT:",
        *identity_blocks,
        reference_map,
        "Render one coherent cinematic frame while preserving every identity above.",
    ]))
    shot_seed = _stable_seed(
        project_id,
        shot.get("id"),
        *(item["identity_seed"] for item in metadata),
    )
    return prompt, references, shot_seed, metadata


def generate_random_idea() -> str:
    llm = get_llm_provider()
    try:
        gen = llm.generate_idea_stream()
        ideas = None
        while True:
            try:
                next(gen)
            except StopIteration as e:
                ideas = e.value
                break
        if ideas and isinstance(ideas, list):
            return ideas[0].get("text", "")
        raise RuntimeError("本地模型未返回有效创意")
    finally:
        llm.release()


def generate_random_idea_stream() -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
    llm = get_llm_provider()
    try:
        gen = llm.generate_idea_stream()
        while True:
            try:
                event = next(gen)
                if isinstance(event, dict):
                    yield event
                else:
                    yield {"type": "text_delta", "index": 0, "delta": str(event)}
            except StopIteration as e:
                ideas = e.value
                if ideas and isinstance(ideas, list):
                    return ideas
                raise RuntimeError("本地模型未返回有效创意") from None
    finally:
        llm.release()


def generate_analysis_stream(source_text: str) -> Generator[dict[str, Any], None, dict[str, Any]]:
    llm = get_llm_provider()
    try:
        gen = llm.generate_analysis_stream(source_text)
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                return e.value
    finally:
        llm.release()


def generate_characters_stream(source_text: str) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
    llm = get_llm_provider()
    try:
        gen = llm.generate_characters_stream(source_text)
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                return e.value
    finally:
        llm.release()


def generate_storyboard_stream(
    source_text: str, analysis: dict[str, Any]
) -> Generator[dict[str, Any], None, tuple[str, list[dict[str, Any]]]]:
    llm = get_llm_provider()
    try:
        gen = llm.generate_storyboard_stream(source_text, analysis)
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                return e.value
    finally:
        llm.release()


def generate_video_prompts_stream(
    project_id: str,
    source_text: str,
    script: str,
    shots: list[dict[str, Any]],
    characters: list[dict[str, Any]] | None = None,
) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
    llm = get_llm_provider()
    try:
        gen = llm.generate_video_prompts_stream(source_text, script, shots, characters)
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                return e.value
    finally:
        llm.release()


def _get_image_dimensions(project_id: str) -> tuple[int, int]:
    preset_name = DEFAULT_IMAGE_RESOLUTION
    try:
        project = get_project(project_id)
        if project and project.get("resolution_preset"):
            preset_name = project["resolution_preset"]
    except Exception:
        pass
    preset = IMAGE_RESOLUTION_PRESETS.get(preset_name, IMAGE_RESOLUTION_PRESETS[DEFAULT_IMAGE_RESOLUTION])
    return int(preset["width"]), int(preset["height"])


def _generate_image(
    img_provider: Any,
    prompt: str,
    target: Path,
    width: int,
    height: int,
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    seed: int | None = None,
    reference_images: list[Path] | None = None,
) -> str:
    """Generate one real or explicitly configured Mock image and validate output."""
    try:
        img_provider.generate_image(
            prompt,
            target,
            width=width,
            height=height,
            progress_callback=progress_callback,
            status_callback=status_callback,
            seed=seed,
            reference_images=reference_images,
        )
        if not target.exists() or target.stat().st_size < GENERATED_IMAGE_MIN_BYTES:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"图像 Provider 未生成有效文件：{target.name}")
        return img_provider.name
    finally:
        img_provider.release()


GENERATION_HEARTBEAT_SECONDS = 5.0


def _generate_image_stream(
    img_provider: Any,
    prompt: str,
    target: Path,
    width: int,
    height: int,
    progress_event_type: str,
    context: dict[str, Any],
    seed: int | None = None,
    reference_images: list[Path] | None = None,
) -> Generator[dict[str, Any], None, str]:
    """Run blocking local inference in a worker and expose progress plus liveness heartbeats."""
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    started = time.monotonic()
    latest_phase = "starting"
    latest_message = "正在启动图像生成任务"

    def report_progress(step: int, total_steps: int) -> None:
        events.put(("event", {
            "type": progress_event_type,
            "step": step,
            "total_steps": total_steps,
            **context,
        }))

    def report_status(payload: dict[str, Any]) -> None:
        events.put(("event", {
            "type": "generation_phase",
            **context,
            **payload,
        }))

    def worker() -> None:
        try:
            provider_name = _generate_image(
                img_provider,
                prompt,
                target,
                width,
                height,
                progress_callback=report_progress,
                status_callback=report_status,
                seed=seed,
                reference_images=reference_images,
            )
            events.put(("done", provider_name))
        except BaseException as exc:
            events.put(("error", exc))

    thread = threading.Thread(
        target=worker,
        name=f"image-generation-{target.stem}",
        daemon=True,
    )
    thread.start()
    while True:
        try:
            kind, payload = events.get(timeout=GENERATION_HEARTBEAT_SECONDS)
        except queue.Empty:
            yield {
                "type": "generation_heartbeat",
                **context,
                "phase": latest_phase,
                "message": latest_message,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            continue
        if kind == "event":
            if payload.get("type") == "generation_phase":
                latest_phase = str(payload.get("phase") or latest_phase)
                latest_message = str(payload.get("message") or latest_message)
            payload["elapsed_seconds"] = round(time.monotonic() - started, 1)
            yield payload
        elif kind == "done":
            thread.join(timeout=0.1)
            return str(payload)
        else:
            thread.join(timeout=0.1)
            raise payload


def generate_character_images_stream(
    project_id: str, character: dict[str, Any]
) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
    char_dir = character_asset_directory(project_id, character["id"])
    images: list[dict[str, Any]] = []
    identity_prompt = build_character_identity_prompt(character)
    identity_seed = _stable_seed(project_id, character["id"], "identity")
    character["identity_prompt"] = identity_prompt
    character["identity_seed"] = identity_seed
    existing_canonical_reference = _canonical_character_reference(character)
    total_views = len(CHARACTER_VIEWS)
    img_provider = get_image_provider()

    for idx, view in enumerate(CHARACTER_VIEWS, start=1):
        w, h = view["size"]
        version = len([img for img in character.get("reference_images", []) if img["view"] == view["key"]]) + 1
        target = char_dir / f"{view['key']}-v{version}.png"
        full_prompt = (
            f"{CHARACTER_CONSISTENCY_SYSTEM_PROMPT}\n"
            f"{identity_prompt}\n"
            f"Create the requested canonical character sheet view: {view['prompt_suffix']}. "
            "The view must depict the exact same individual as every other view."
        )
        generated_front = next(
            (
                Path(image["path"])
                for image in images
                if image.get("view") == "front" and Path(image["path"]).is_file()
            ),
            None,
        )
        canonical_reference = existing_canonical_reference or generated_front
        reference_paths = [canonical_reference] if canonical_reference else []

        yield {
            "type": "asset_start", "asset_type": "character", "asset_name": character["name"],
            "view": view["key"], "view_label": view["label"],
            "index": idx, "total": total_views, "width": w, "height": h,
        }
        context = {
            "asset_type": "character",
            "asset_name": character["name"],
            "view": view["key"],
            "view_label": view["label"],
            "index": idx,
            "total": total_views,
        }
        provider_name = yield from _generate_image_stream(
            img_provider,
            full_prompt,
            target,
            w,
            h,
            "asset_progress",
            context,
            seed=identity_seed,
            reference_images=reference_paths,
        )
        img = {
            "view": view["key"], "view_label": view["label"], "version": version,
            "path": str(target), "prompt": full_prompt, "provider": provider_name,
            "width": w, "height": h, "status": "ready",
            "identity_seed": identity_seed,
            "identity_prompt": identity_prompt,
            "reference_conditioning": [str(path) for path in reference_paths],
        }
        images.append(img)
        yield {"type": "asset_done", "image": img, "asset_type": "character"}

    return images


def generate_prop_image_stream(
    project_id: str, prop: dict[str, Any]
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    prop_dir = prop_asset_directory(project_id, prop["id"])
    w, h = PROP_REF_SIZE
    version = len(prop.get("reference_images", [])) + 1
    target = prop_dir / f"ref-v{version}.png"
    prompt = prop.get("visual_prompt") or prop.get("prompt") or prop.get("name", "detailed prop")
    full_prompt = f"{prompt}, product photography style, centered composition, white background, high detail"
    img_provider = get_image_provider()
    yield {
        "type": "asset_start", "asset_type": "prop", "asset_name": prop["name"],
        "index": 1, "total": 1, "width": w, "height": h,
    }
    context = {"asset_type": "prop", "asset_name": prop["name"], "index": 1, "total": 1}
    provider_name = yield from _generate_image_stream(
        img_provider, full_prompt, target, w, h, "asset_progress", context,
    )
    img = {
        "version": version, "path": str(target), "prompt": full_prompt,
        "provider": provider_name, "width": w, "height": h, "status": "ready",
    }
    prop["reference_images"].append(img)
    yield {"type": "asset_done", "image": img, "asset_type": "prop", "asset_name": prop["name"]}
    return img


def generate_environment_image_stream(
    project_id: str, environment: dict[str, Any]
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    env_dir = environment_asset_directory(project_id, environment["id"])
    w, h = ENVIRONMENT_REF_SIZE
    version = len(environment.get("reference_images", [])) + 1
    target = env_dir / f"ref-v{version}.png"
    prompt = environment.get("visual_prompt") or environment.get("prompt") or environment.get("name", "cinematic environment")
    full_prompt = f"{prompt}, establishing shot, wide angle, cinematic lighting, detailed environment, epic composition"
    img_provider = get_image_provider()
    yield {
        "type": "asset_start", "asset_type": "environment", "asset_name": environment["name"],
        "index": 1, "total": 1, "width": w, "height": h,
    }
    context = {
        "asset_type": "environment",
        "asset_name": environment["name"],
        "index": 1,
        "total": 1,
    }
    provider_name = yield from _generate_image_stream(
        img_provider, full_prompt, target, w, h, "asset_progress", context,
    )
    img = {
        "version": version, "path": str(target), "prompt": full_prompt,
        "provider": provider_name, "width": w, "height": h, "status": "ready",
    }
    environment["reference_images"].append(img)
    yield {"type": "asset_done", "image": img, "asset_type": "environment", "asset_name": environment["name"]}
    return img


def generate_keyframe_image_stream(
    project_id: str, shot: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    version = len(shot["image_versions"]) + 1
    target = asset_directory(project_id, shot["id"]) / f"keyframe-v{version}.png"
    width, height = _get_image_dimensions(project_id)
    img_provider = get_image_provider()
    prompt = shot["prompt"]
    consistency_prompt, reference_paths, shot_seed, identity_metadata = (
        _keyframe_character_consistency(project_id, shot, analysis)
    )
    if analysis:
        visual_style = analysis.get("visual_style", "cinematic film style")
        if visual_style and visual_style not in prompt:
            prompt = f"{prompt}, {visual_style}"
    if consistency_prompt:
        prompt = f"{consistency_prompt}\nSHOT INSTRUCTION: {prompt}"

    yield {
        "type": "keyframe_start",
        "entity": "shot",
        "entity_name": shot.get("title", f"镜头{shot.get('number', '?')}"),
        "shot_id": shot["id"],
        "shot_number": shot.get("number"),
        "width": width,
        "height": height,
    }
    context = {
        "asset_type": "keyframe",
        "asset_name": shot.get("title", f"镜头{shot.get('number', '?')}"),
        "shot_id": shot["id"],
        "shot_number": shot.get("number"),
    }
    provider_name = yield from _generate_image_stream(
        img_provider,
        prompt,
        target,
        width,
        height,
        "keyframe_progress",
        context,
        seed=shot_seed,
        reference_images=reference_paths,
    )
    asset = {
        "version": version, "path": str(target), "prompt": prompt,
        "provider": provider_name, "status": "ready", "width": width, "height": height,
        "seed": shot_seed,
        "character_consistency": identity_metadata,
        "reference_conditioning": [str(path) for path in reference_paths],
    }
    shot["image_versions"].append(asset)
    shot["selected_image"] = asset
    shot["review_status"] = "pending"
    yield {"type": "keyframe_done", "image": asset, "entity": "shot", "shot_id": shot["id"]}

    return asset


def generate_keyframe_image(
    project_id: str, shot: dict[str, Any], analysis: dict[str, Any] | None = None
) -> dict[str, Any]:
    version = len(shot["image_versions"]) + 1
    target = asset_directory(project_id, shot["id"]) / f"keyframe-v{version}.png"
    width, height = _get_image_dimensions(project_id)
    img_provider = get_image_provider()
    prompt = shot["prompt"]
    consistency_prompt, reference_paths, shot_seed, identity_metadata = (
        _keyframe_character_consistency(project_id, shot, analysis)
    )
    if analysis:
        visual_style = analysis.get("visual_style", "cinematic")
        if visual_style:
            prompt = f"{prompt}, {visual_style}"
    if consistency_prompt:
        prompt = f"{consistency_prompt}\nSHOT INSTRUCTION: {prompt}"
    provider_name = _generate_image(
        img_provider,
        prompt,
        target,
        width,
        height,
        seed=shot_seed,
        reference_images=reference_paths,
    )
    asset = {
        "version": version, "path": str(target), "prompt": prompt,
        "provider": provider_name, "status": "ready", "width": width, "height": height,
        "seed": shot_seed,
        "character_consistency": identity_metadata,
        "reference_conditioning": [str(path) for path in reference_paths],
    }
    shot["image_versions"].append(asset)
    shot["selected_image"] = asset
    shot["review_status"] = "pending"
    return asset


def generate_final_video(project_id: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
    vid = get_video_provider()
    try:
        result = vid.generate_video(project_id, shots)
        return result
    finally:
        vid.release()
