from __future__ import annotations

from typing import Any

from app.evaluation.rubric import Rubric

SCORE_LABELS = {
    1: "严重失败，不可使用",
    2: "明显问题，需要大幅修改",
    3: "基本可用，存在明显瑕疵",
    4: "质量较好，仅需轻微调整",
    5: "表现优秀，可直接使用",
}


def calculate_weighted_total(
    scores: dict[str, int], rubric: Rubric
) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for dim in rubric.dimensions:
        raw = scores.get(dim.key)
        if raw is None:
            if dim.required:
                return 0.0
            continue
        try:
            score_int = int(raw)
        except (TypeError, ValueError):
            if dim.required:
                return 0.0
            continue
        if score_int < 1 or score_int > 5:
            if dim.required:
                return 0.0
            continue
        weighted_sum += score_int * dim.weight
        total_weight += dim.weight
    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 2)


def validate_scores(scores: dict[str, int], rubric: Rubric) -> tuple[bool, list[str]]:
    errors = []
    for dim in rubric.required_dimensions:
        val = scores.get(dim.key)
        if val is None:
            errors.append(f"必选维度「{dim.label}」未评分")
        else:
            try:
                iv = int(val)
                if iv < 1 or iv > 5:
                    errors.append(f"维度「{dim.label}」分数需在1-5之间")
            except (TypeError, ValueError):
                errors.append(f"维度「{dim.label}」分数无效")
    for key, val in scores.items():
        dim = rubric.get_dimension(key)
        if not dim:
            continue
        try:
            iv = int(val)
            if iv < 1 or iv > 5:
                errors.append(f"维度「{dim.label}」分数需在1-5之间")
        except (TypeError, ValueError):
            errors.append(f"维度「{dim.label}」分数无效")
    return len(errors) == 0, errors


def score_color(score: float) -> str:
    if score >= 4.0:
        return "#22c55e"
    elif score >= 3.0:
        return "#eab308"
    elif score >= 2.0:
        return "#f97316"
    else:
        return "#ef4444"


def score_progress_value(score: float) -> float:
    return max(0.0, min(1.0, (score - 1) / 4.0)) if score else 0.0


def filter_items(
    items: list[dict[str, Any]],
    *,
    statuses: set[str] | None = None,
    asset_types: set[str] | None = None,
    model_providers: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply dashboard filters while treating empty selections as no filter."""
    return [
        item
        for item in items
        if (not statuses or item.get("status") in statuses)
        and (not asset_types or item.get("asset_type") in asset_types)
        and (not model_providers or item.get("model_provider") in model_providers)
    ]


def summarize_items(items: list[dict[str, Any]], rubric: Rubric) -> dict[str, Any]:
    submitted = [it for it in items if it.get("status") == "submitted"]
    if not submitted:
        return {
            "total": len(items),
            "submitted": 0,
            "avg_total": 0.0,
            "dim_avgs": {},
            "tag_counts": {},
            "distribution": {"1分": 0, "2分": 0, "3分": 0, "4分": 0, "5分": 0},
        }

    totals = [it.get("total_score", 0) for it in submitted if it.get("total_score") is not None]
    dim_sums: dict[str, float] = {d.key: 0.0 for d in rubric.dimensions}
    dim_counts: dict[str, int] = {d.key: 0 for d in rubric.dimensions}
    tag_counts: dict[str, int] = {}
    dist = {"1分": 0, "2分": 0, "3分": 0, "4分": 0, "5分": 0}

    for it in submitted:
        ts = it.get("total_score")
        if ts is not None:
            bucket = max(1, min(5, round(ts)))
            dist[f"{bucket}分"] += 1
        for dim_key, sdata in (it.get("scores") or {}).items():
            sc = sdata.get("score") if isinstance(sdata, dict) else sdata
            try:
                sci = int(sc)
                dim_sums[dim_key] = dim_sums.get(dim_key, 0) + sci
                dim_counts[dim_key] = dim_counts.get(dim_key, 0) + 1
            except (TypeError, ValueError):
                pass
        for t in it.get("bad_case_tags") or []:
            tag = t.get("tag", t) if isinstance(t, dict) else t
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    dim_avgs = {}
    for d in rubric.dimensions:
        cnt = dim_counts.get(d.key, 0)
        dim_avgs[d.key] = round(dim_sums.get(d.key, 0) / cnt, 2) if cnt > 0 else 0.0

    return {
        "total": len(items),
        "submitted": len(submitted),
        "avg_total": round(sum(totals) / len(totals), 2) if totals else 0.0,
        "dim_avgs": dim_avgs,
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        "distribution": dist,
    }
