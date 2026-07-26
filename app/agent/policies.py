from __future__ import annotations

import re
from typing import Any

from app.agent.schemas import EvaluationResult, ReflectionFeedback, RepairAction
from app.config import AGENT_STORYBOARD_PASS_SCORE

_RISK_TERMS = ("未成年人色情", "性侵", "仇恨煽动", "自杀教程")


def evaluate_storyboard(
    shots: list[dict[str, Any]],
    analysis: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> EvaluationResult:
    issues: list[str] = []
    dimensions: list[str] = []
    actions: list[RepairAction] = []
    if not shots:
        return EvaluationResult(
            score=1, passed=False, issues=["分镜为空"], feedback="必须生成至少两个镜头",
            failed_dimensions=["structure"], recommended_action="repair",
        )

    target = int((constraints or {}).get("duration_seconds") or analysis.get("total_duration_seconds", 20))
    total = sum(int(shot.get("duration_seconds", 0) or 0) for shot in shots)
    if abs(total - target) > 5:
        issues.append(f"总时长 {total} 秒与目标 {target} 秒偏差超过 5 秒")
        dimensions.append("duration")
        actions.append(RepairAction(target="storyboard", action="shorten_shot" if total > target else "extend_shot", reason=issues[-1]))

    required = ("id", "number", "description", "duration_seconds", "prompt")
    for index, shot in enumerate(shots, start=1):
        missing = [key for key in required if not shot.get(key)]
        if missing:
            dimensions.append("structure")
            issues.append(f"镜头 {index} 缺少字段：{', '.join(missing)}")
            actions.append(RepairAction(target=f"shot-{index:02d}", action="complete_field", reason=issues[-1]))
        duration = int(shot.get("duration_seconds", 0) or 0)
        if duration < 2 or duration > 8:
            dimensions.append("shot_duration")
            issues.append(f"镜头 {index} 时长 {duration} 秒不在 2-8 秒范围")
        description = str(shot.get("description", ""))
        if len(re.findall(r"[，,；;然后并且]", description)) >= 5:
            dimensions.append("action_complexity")
            issues.append(f"镜头 {index} 包含过多连续动作")
            actions.append(RepairAction(target=f"shot-{index:02d}", action="split_shot", reason=issues[-1]))

    combined = " ".join(str(shot.get("description", "")) for shot in shots)
    high_risk = any(term in combined for term in _RISK_TERMS)
    if high_risk:
        issues.append("检测到需要人工确认的高风险内容")
        dimensions.append("safety")

    unique_dimensions = list(dict.fromkeys(dimensions))
    score = max(1.0, round(5.0 - 0.55 * len(unique_dimensions) - 0.15 * max(0, len(issues) - 1), 2))
    passed = score >= AGENT_STORYBOARD_PASS_SCORE and not high_risk
    return EvaluationResult(
        score=score,
        passed=passed,
        issues=issues,
        feedback="；".join(issues) if issues else "结构、时长与镜头可执行性检查通过",
        failed_dimensions=unique_dimensions,
        recommended_action="human_review" if high_risk else ("continue" if passed else "repair"),
        high_risk=high_risk,
    )


def build_reflection(evaluation: dict[str, Any]) -> ReflectionFeedback:
    failed = list(evaluation.get("failed_dimensions", []))
    actions: list[RepairAction] = []
    for issue in evaluation.get("issues", []):
        target_match = re.search(r"镜头\s*(\d+)", issue)
        target = f"shot-{int(target_match.group(1)):02d}" if target_match else "storyboard"
        if "总时长" in issue:
            action = "shorten_shot" if "偏差" in issue else "extend_shot"
        elif "字段" in issue:
            action = "complete_field"
        elif "连续动作" in issue:
            action = "split_shot"
        elif "风险" in issue:
            action = "remove_risk"
        else:
            action = "regenerate"
        actions.append(RepairAction(target=target, action=action, reason=str(issue)))
    return ReflectionFeedback(
        failed_dimensions=failed,
        repair_actions=actions,
        summary="请仅修复列出的失败维度并保持其余镜头设定稳定。",
    )
