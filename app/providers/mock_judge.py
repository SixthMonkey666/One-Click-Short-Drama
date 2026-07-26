from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any, Generator

from app.providers.judge_base import JudgeProvider, JudgeResult


class MockJudgeProvider(JudgeProvider):
    name = "mock"
    display_name = "Mock Judge (模拟评测)"
    prompt_version = "v1.0-mock"
    model_version = "mock-v1"

    def evaluate_image(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
    ) -> JudgeResult:
        seed_src = f"{str(image_path)}::{prompt}::{task_type}"
        seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)

        dimensions = rubric.get("dimensions", [])
        dim_scores: dict[str, int] = {}
        base = rng.randint(2, 4)
        for dim in dimensions:
            key = dim.get("key", "")
            variation = rng.choices([-1, 0, 0, 1], weights=[1, 3, 4, 2])[0]
            score = max(1, min(5, base + variation))
            dim_scores[key] = score

        all_tags = rubric.get("bad_case_tags", [
            "指令未遵循", "主体缺失", "主体/角色漂移", "属性错误", "场景跳变",
            "构图异常", "肢体或物理错误", "文字错误", "风格不一致", "画面瑕疵",
            "镜头重复", "动作不连贯", "跨帧闪烁", "Prompt幻觉", "其他",
        ])

        tags: list[str] = []
        avg = sum(dim_scores.values()) / len(dim_scores) if dim_scores else 3.0
        if avg <= 2.5:
            num_tags = rng.randint(1, 3)
            tags = rng.sample(all_tags, min(num_tags, len(all_tags)))
        elif avg <= 3.5:
            if rng.random() < 0.5:
                tags = [rng.choice(all_tags)]

        total = round(sum(dim_scores.values()) / len(dim_scores), 2) if dim_scores else 3.0
        confidence = round(rng.uniform(0.6, 0.9), 2)

        explanation = self._gen_explanation(dim_scores, tags, total, rubric)

        time.sleep(0.3)
        return JudgeResult(
            dimension_scores=dim_scores,
            total_score=total,
            bad_case_tags=tags,
            explanation=explanation,
            confidence=confidence,
            raw_response=f"mock_judge_seed={seed}",
        )

    def evaluate_image_stream(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
    ) -> Generator[dict[str, Any], None, JudgeResult]:
        dims = rubric.get("dimensions", [])
        total_steps = max(len(dims), 1)
        yield {"type": "status", "message": "[Mock] 正在分析图像..."}
        for i in range(total_steps):
            time.sleep(0.15)
            yield {
                "type": "judge_progress",
                "step": i + 1,
                "total_steps": total_steps,
                "dimension": dims[i].get("label", f"维度{i+1}") if i < len(dims) else "",
            }
        result = self.evaluate_image(image_path, prompt, task_type, rubric, reference_images)
        yield {"type": "judge_complete", "result": result.to_dict()}
        return result

    def _gen_explanation(
        self, scores: dict[str, int], tags: list[str], total: float, rubric: dict[str, Any]
    ) -> str:
        if total >= 4.0:
            base = "整体质量优秀，符合Prompt要求。"
        elif total >= 3.0:
            base = "整体基本可用，但存在一些可改进的地方。"
        elif total >= 2.0:
            base = "存在较明显问题，建议修改后重新生成。"
        else:
            base = "质量较差，不满足使用要求。"
        dim_map = {d["key"]: d.get("label", d["key"]) for d in rubric.get("dimensions", [])}
        weak = [dim_map.get(k, k) for k, v in scores.items() if v <= 2]
        strong = [dim_map.get(k, k) for k, v in scores.items() if v >= 4]
        parts = [base]
        if strong:
            parts.append(f"表现较好的维度：{'、'.join(strong[:3])}。")
        if weak:
            parts.append(f"需要改进的维度：{'、'.join(weak[:3])}。")
        if tags:
            parts.append(f"发现Bad Case：{'、'.join(tags)}。")
        parts.append("（注意：此为Mock自动评分结果，仅供流程测试使用。）")
        return "".join(parts)

    @classmethod
    def is_available(cls) -> bool:
        return True
