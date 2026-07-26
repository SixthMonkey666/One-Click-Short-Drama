"""Mock providers for offline testing and development without GPU models."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Generator

from PIL import Image, ImageDraw

from app.providers.base import ImageProvider, LLMProvider, StatusCallback, VideoProvider
from app.repository import asset_directory

PALETTE = ["#7757d7", "#147a8a", "#ca6a31", "#ad4664", "#4d7f3c"]

_MOCK_IDEAS = [
    "一只流浪橘猫在雨夜的便利店门口遇到了撑伞下班的女孩，女孩把自己的围巾分给它一半，从此橘猫每天都在便利店门口等她，直到有一天女孩带来了一个猫包。",
    "深海中最后一只发光的水母，在黑暗里孤独漂浮了百年，某天它遇到了一艘沉没的古老潜水艇，里面有一盏还在闪烁的小灯，水母把光献给了它。",
    "老书店的老板发现每到午夜，书架上的书就会自动翻开，故事里的角色跑出来在店里开派对，直到某天一本从未售出的童话书里走出了一个迷路的小王子。",
    "宇航员在火星表面执行任务时捡到了一颗会发出音乐的小石头，他把石头带回地球送给了失明的女儿，女儿说她看到了星星的颜色。",
    "城市里最古老的邮筒在即将被拆除的前一晚，吐出了一封五十年前寄出的信，邮递员顺着地址找到收信人，发现是一位等待了一辈子的老奶奶。",
]


def _draw_placeholder(target: Path, label: str, width: int, height: int, color_idx: int = 0) -> None:
    base = PALETTE[color_idx % len(PALETTE)]
    image = Image.new("RGB", (width, height), "#101522")
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / height
        shade = int(28 + ratio * 24)
        draw.line((0, y, width, y), fill=(shade, shade + 5, shade + 20))
    draw.ellipse((width * 0.15, height * 0.1, width * 0.85, height * 0.9), fill=base)
    font_size = max(12, int(min(width, height) * 0.05))
    draw.text((width * 0.1, height * 0.45), label, fill="white", font_size=font_size)
    image.save(target, format="PNG")


def _draw_keyframe(target: Path, shot: dict[str, Any], width: int = 1280, height: int = 720) -> None:
    base = PALETTE[(shot.get("number", 1) - 1) % len(PALETTE)]
    seed = f"{shot.get('title', '')}:{shot.get('description', '')}:{len(shot.get('image_versions', []))}"
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), "#101522")
    draw = ImageDraw.Draw(image)
    sx = width / 1280
    sy = height / 720
    for y in range(height):
        ratio = y / height
        shade = int(28 + ratio * 24)
        draw.line((0, y, width, y), fill=(shade, shade + 5, shade + 20))
    draw.ellipse((int(850 * sx), int(80 * sy), int(1190 * sx), int(420 * sy)), fill=base)
    draw.polygon([(0, int(540 * sy)), (int(310 * sx), int(320 * sy)), (int(620 * sx), int(540 * sy))], fill="#202a3d")
    draw.polygon([(int(440 * sx), int(550 * sy)), (int(790 * sx), int(270 * sy)), (int(1120 * sx), int(550 * sy))], fill="#171e30")
    subject_x = int((330 + rng.randint(-80, 80)) * sx)
    subject_y = int((430 + rng.randint(-20, 60)) * sy)
    draw.ellipse((subject_x, subject_y - int(130 * sy), subject_x + int(110 * sx), subject_y - int(20 * sy)), fill="#e6c5a3")
    draw.rounded_rectangle((subject_x - int(35 * sx), subject_y - int(30 * sy), subject_x + int(145 * sx), subject_y + int(220 * sy)), radius=int(40 * min(sx, sy)), fill="#111827")
    draw.rectangle((0, int(610 * sy), width, height), fill="#0b1020")
    draw.rounded_rectangle((int(52 * sx), int(48 * sy), int(360 * sx), int(112 * sy)), radius=int(20 * min(sx, sy)), fill="#0b1020")
    font_size = max(12, int(30 * min(sx, sy)))
    draw.text((int(78 * sx), int(68 * sy)), f"KEYFRAME  {shot.get('number', 1):02d}", fill="white", font_size=font_size)
    image.save(target, format="PNG")


def _draw_asset_image(target: Path, asset_type: str, name: str, color_idx: int, width: int = 1024, height: int = 1024) -> None:
    base = PALETTE[color_idx % len(PALETTE)]
    image = Image.new("RGB", (width, height), "#101522")
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / height
        shade = int(28 + ratio * 24)
        draw.line((0, y, width, y), fill=(shade, shade + 5, shade + 20))
    cx, cy = width // 2, height // 2
    if asset_type == "character":
        draw.ellipse((cx - 140, cy - 200, cx + 140, cy + 200), fill="#e6c5a3")
        draw.rounded_rectangle((cx - 200, cy + 100, cx + 200, cy + 440), radius=60, fill=base)
    elif asset_type == "prop":
        draw.rounded_rectangle((cx - 250, cy - 180, cx + 250, cy + 180), radius=40, fill=base)
    else:
        draw.rectangle((0, int(height * 0.6), width, height), fill="#1a2235")
        draw.polygon([(0, int(height * 0.6)), (width * 0.3, int(height * 0.3)), (width * 0.6, int(height * 0.6))], fill="#1e2d4a")
        draw.polygon([(width * 0.4, int(height * 0.6)), (width * 0.7, int(height * 0.2)), (width, int(height * 0.6))], fill="#253658")
        draw.ellipse((int(width * 0.72), int(height * 0.12), int(width * 0.92), int(height * 0.32)), fill="#e8d080")
    font_size = max(14, int(min(width, height) * 0.04))
    label = f"{asset_type.upper()}: {name[:25]}"
    draw.text((width * 0.08, height * 0.05), label, fill="white", font_size=font_size)
    image.save(target, format="PNG")


class MockLLMProvider(LLMProvider):
    name = "mock"
    display_name = "模拟模板（离线测试）"

    def generate_idea_stream(
        self,
    ) -> Generator[dict[str, Any] | str, None, list[dict[str, Any]]]:
        idea = random.choice(_MOCK_IDEAS)
        for ch in idea:
            time.sleep(0.02)
            yield ch
        return [{"text": idea, "topics": []}]

    def generate_analysis_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        yield {"type": "status", "message": "使用模拟模式分析创意..."}
        yield {"type": "text_start"}
        chunks = [
            '{"title": "雨夜橘猫的故事",',
            '"genre": "温情治愈",',
            '"theme": "人与动物的羁绊、温暖与陪伴",',
            '"tone": "温暖、治愈、细腻",',
            '"visual_style": "cinematic film style, rain-soaked streets at night, warm convenience store lights",',
            '"total_duration_seconds": 30,',
            f'"narrative_summary": "{source_text[:80]}",',
            '"characters": [{"id":"char-01","name":"流浪橘猫","type":"动物","description":"一只在雨夜流浪的橘猫","appearance":"fluffy orange tabby cat, wet fur, big golden eyes"},{"id":"char-02","name":"下班女孩","type":"人物","description":"撑伞下班的年轻女孩","appearance":"young woman in office clothes, holding transparent umbrella, gentle expression"}],',
            '"props": [{"id":"prop-01","name":"红色围巾","description":"女孩戴的温暖围巾","visual_prompt":"red knit scarf, warm texture, detailed fabric"},{"id":"prop-02","name":"透明雨伞","description":"女孩撑的透明雨伞","visual_prompt":"clear transparent umbrella with rain drops"},{"id":"prop-03","name":"猫包","description":"用来接橘猫回家的猫包","visual_prompt":"soft cat carrier bag, pet carrier"}],',
            '"environments": [{"id":"env-01","name":"雨夜便利店门口","description":"夜晚下雨的便利店外，灯光温暖","visual_prompt":"convenience store entrance at night, rain, warm neon lights, wet pavement reflection"}]}'
        ]
        for chunk in chunks:
            time.sleep(0.1)
            yield {"type": "text_delta", "delta": chunk}
        yield {"type": "text_end"}
        analysis = {
            "title": "雨夜橘猫的故事",
            "genre": "温情治愈",
            "theme": "人与动物的羁绊、温暖与陪伴",
            "tone": "温暖、治愈、细腻",
            "visual_style": "cinematic film style, rain-soaked streets at night, warm convenience store lights",
            "total_duration_seconds": 30,
            "narrative_summary": source_text[:200],
            "characters": [
                {"id": "char-01", "name": "流浪橘猫", "type": "动物", "description": "一只在雨夜流浪的橘猫", "appearance": "fluffy orange tabby cat, wet fur, big golden eyes", "reference_images": []},
                {"id": "char-02", "name": "下班女孩", "type": "人物", "description": "撑伞下班的年轻女孩", "appearance": "young woman in office clothes, holding transparent umbrella, gentle expression", "reference_images": []},
            ],
            "props": [
                {"id": "prop-01", "name": "红色围巾", "description": "女孩戴的温暖围巾", "visual_prompt": "red knit scarf, warm texture, detailed fabric", "prompt": "red knit scarf", "appearance": "red knit scarf", "reference_images": []},
                {"id": "prop-02", "name": "透明雨伞", "description": "女孩撑的透明雨伞", "visual_prompt": "clear transparent umbrella with rain drops", "prompt": "clear umbrella", "appearance": "clear umbrella", "reference_images": []},
                {"id": "prop-03", "name": "猫包", "description": "用来接橘猫回家的猫包", "visual_prompt": "soft cat carrier bag, pet carrier", "prompt": "cat carrier", "appearance": "cat carrier", "reference_images": []},
            ],
            "environments": [
                {"id": "env-01", "name": "雨夜便利店门口", "description": "夜晚下雨的便利店外，灯光温暖", "visual_prompt": "convenience store entrance at night, rain, warm neon lights, wet pavement", "prompt": "convenience store night rain", "appearance": "convenience store night", "reference_images": []},
            ],
        }
        yield {"type": "status", "message": "创意分析完成"}
        return analysis

    def generate_characters_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        yield {"type": "status", "message": "使用模拟模式识别角色..."}
        time.sleep(0.3)
        chars = [
            {"id": "char-01", "name": "主角", "type": "人物",
             "description": f"模拟角色：基于「{source_text[:20]}...」推断的主角。",
             "appearance": "cinematic character portrait, detailed face, expressive eyes",
             "prompt": "cinematic character portrait", "reference_images": []}
        ]
        for c in chars:
            yield {"type": "character_found", "character": c}
        return chars

    def generate_storyboard_stream(
        self, source_text: str, analysis: dict[str, Any]
    ) -> Generator[dict[str, Any], None, tuple[str, list[dict[str, Any]]]]:
        yield {"type": "status", "message": "使用模拟模式生成分镜表..."}
        time.sleep(0.3)
        char_ids = [c["id"] for c in analysis.get("characters", [])]
        prop_ids = [p["id"] for p in analysis.get("props", [])]
        env_ids = [e["id"] for e in analysis.get("environments", [])]
        default_env = env_ids[0] if env_ids else "env-01"
        default_chars = char_ids[:1] if char_ids else []

        _mock = [
            ("开场建立", "广角交代环境，雨夜便利店的温暖灯光", "cinematic wide shot, rain-soaked street, warm convenience store neon", "远景", "固定镜头", default_chars, prop_ids[:1], default_env, 4),
            ("橘猫出现", "橘猫瑟缩在门口躲雨，被路人忽略", "cinematic medium shot, orange tabby cat shivering in rain outside store", "中景", "缓慢推进", char_ids[:1], prop_ids[:2], default_env, 5),
            ("女孩遇见", "女孩撑伞走近，注意到橘猫", "cinematic medium shot, young woman with transparent umbrella noticing cat", "中景", "轻微横移", default_chars, prop_ids[1:2], default_env, 5),
            ("分享围巾", "女孩把围巾分给橘猫一半，橘猫温暖起来", "cinematic close-up, red scarf shared with cat, warm lighting on faces", "特写", "缓慢推近", char_ids, prop_ids[:2], default_env, 5),
            ("日常等待", "橘猫每天在便利店门口等待女孩到来", "cinematic time-lapse, cat waiting at store entrance across different weather", "远景→中景", "时间流逝蒙太奇", char_ids[:1], [], default_env, 6),
            ("猫包到来", "女孩带来猫包，橘猫终于有了家", "cinematic medium shot, girl opening cat carrier, cat stepping in, warm ending", "中景", "缓慢拉远", char_ids, prop_ids[2:3], default_env, 5),
        ]
        shots = []
        for i, (t, d, p, ca, cm, cids, pids, eid, dur) in enumerate(_mock, start=1):
            shots.append({
                "id": f"shot-{i:02d}",
                "number": i,
                "title": t,
                "description": d,
                "duration_seconds": dur,
                "camera_angle": ca,
                "camera_movement": cm,
                "character_ids": cids or default_chars,
                "prop_ids": pids,
                "environment_id": eid or default_env,
                "prompt": p,
                "image_versions": [],
                "selected_image": None,
                "review_status": "pending",
                "video_prompt": None,
            })
        total_dur = sum(s["duration_seconds"] for s in shots)
        script = f"【模拟分镜脚本】\n片名：{analysis.get('title', '未命名')}\n创意：{source_text}\n\n" + "\n".join(
            f"镜头{s['number']}：{s['title']}（{s['duration_seconds']}秒）- {s['description']}" for s in shots
        ) + f"\n（全片预计 {total_dur} 秒）"
        yield {"type": "status", "message": f"分镜表已生成，共 {len(shots)} 个镜头，全片约 {total_dur} 秒"}
        return script, shots

    def generate_video_prompts_stream(
        self, source_text: str, script: str, shots: list[dict[str, Any]],
        characters: list[dict[str, Any]] | None = None,
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        yield {"type": "status", "message": "使用模拟模式生成视频提示词..."}
        time.sleep(0.3)
        defaults = [
            ("slow zoom in", "缓慢推进镜头，建立氛围"),
            ("gentle pan right", "缓慢横移展现环境"),
            ("subtle dolly forward", "轻微前移聚焦动作"),
            ("slow push in", "缓慢推近情绪细节"),
            ("time-lapse dissolve", "时间流逝蒙太奇效果"),
            ("slow zoom out", "缓慢拉远收束情绪"),
        ]
        result = []
        for i, shot in enumerate(shots):
            cam, cn = defaults[i % len(defaults)]
            result.append({
                "shot_number": shot["number"],
                "motion_prompt_en": f"{cam}, cinematic motion, smooth camera, detailed scene with subtle movement",
                "motion_prompt_cn": cn,
                "camera_movement": cam,
            })
        return result

    @classmethod
    def is_available(cls) -> bool:
        return True


class MockImageProvider(ImageProvider):
    name = "mock"
    display_name = "模拟绘图（离线测试）"
    default_steps = 1

    def generate_image(
        self, prompt: str, output_path: Path, width: int = 1024, height: int = 576,
        progress_callback: Callable[[int, int], None] | None = None,
        status_callback: StatusCallback | None = None,
        seed: int | None = None,
        reference_images: list[Path] | None = None,
    ) -> None:
        del seed, reference_images
        if status_callback:
            status_callback({"phase": "model_ready", "message": "模拟绘图模型已就绪"})
        if progress_callback:
            progress_callback(0, 1)
        if status_callback:
            status_callback({"phase": "denoising", "message": "正在生成图像"})
        time.sleep(0.3)
        color_idx = abs(hash(prompt)) % len(PALETTE)
        _draw_placeholder(output_path, prompt[:30], width, height, color_idx=color_idx)
        if progress_callback:
            progress_callback(1, 1)
        if status_callback:
            status_callback({"phase": "saving", "message": "图像已保存"})

    @classmethod
    def is_available(cls) -> bool:
        return True


class MockVideoProvider(VideoProvider):
    name = "mock"
    display_name = "模拟视频（离线测试）"

    def generate_video(self, project_id: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
        project_dir = asset_directory(project_id, "final")
        project_dir.mkdir(parents=True, exist_ok=True)
        manifest = project_dir / "video-manifest.json"
        payload = {
            "provider": self.name,
            "generation_method": "mock",
            "note": "模拟视频生成，等待接入线上 I2V API",
            "shots": [
                {"number": s["number"], "duration_seconds": s["duration_seconds"],
                 "image": s.get("selected_image"), "video_prompt": s.get("video_prompt")}
                for s in shots
            ],
        }
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"provider": self.name, "status": "completed", "path": str(manifest), "manifest": str(manifest)}

    def release(self) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        return True


class FFmpegVideoProvider(VideoProvider):
    name = "ffmpeg"
    display_name = "FFmpeg 关键帧拼接"

    def generate_video(self, project_id: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
        project_dir = asset_directory(project_id, "final")
        project_dir.mkdir(parents=True, exist_ok=True)
        manifest = project_dir / "video-manifest.json"
        payload = {
            "provider": self.name,
            "generation_method": "keyframe_sequence",
            "shots": [
                {"number": s["number"], "duration_seconds": s["duration_seconds"],
                 "image": s["selected_image"], "video_prompt": s.get("video_prompt")}
                for s in shots
            ],
        }
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output = project_dir / "preview.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("找不到 FFmpeg，无法生成 MP4 视频")
        if not shots or any(not shot.get("selected_image") for shot in shots):
            raise RuntimeError("部分镜头缺少关键帧，无法生成完整视频")

        keyframes = [Path(s["selected_image"]["path"]) for s in shots]
        durations = [s["duration_seconds"] for s in shots]
        if not all(f.suffix.lower() == ".png" and f.is_file() for f in keyframes):
            raise RuntimeError("关键帧文件不存在或格式不是 PNG，无法生成视频")

        try:
            inputs: list[str] = []
            for frame, dur in zip(keyframes, durations, strict=True):
                inputs.extend(["-loop", "1", "-t", str(dur), "-i", str(frame)])
            filters = "".join(f"[{i}:v]" for i in range(len(keyframes)))
            timeout_seconds = max(60, int(sum(durations) * 4))
            subprocess.run(
                [ffmpeg, "-y", *inputs,
                 "-filter_complex", f"{filters}concat=n={len(keyframes)}:v=1:a=0,format=yuv420p[v]",
                 "-map", "[v]", "-r", "24", str(output)],
                check=True, capture_output=True, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output.unlink(missing_ok=True)
            raise RuntimeError("FFmpeg 视频合成超时") from exc
        except subprocess.CalledProcessError as exc:
            output.unlink(missing_ok=True)
            detail = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
            message = f"FFmpeg 视频合成失败：{detail[-500:]}" if detail else "FFmpeg 视频合成失败"
            raise RuntimeError(message) from exc
        except OSError as exc:
            output.unlink(missing_ok=True)
            raise RuntimeError(f"无法启动 FFmpeg：{exc}") from exc
        return {"provider": self.name, "status": "completed", "path": str(output), "manifest": str(manifest)}

    def release(self) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        return True
