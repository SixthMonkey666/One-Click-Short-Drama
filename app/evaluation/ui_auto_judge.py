from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from app.evaluation import repository as eval_repo
from app.evaluation.rubric import load_default_rubric
from app.providers import list_available_providers
from app.task_manager import TaskBusyError, get_active_task, submit_task


def _format_elapsed(task: dict[str, Any]) -> str:
    started_at = task.get("started_at") or task.get("created_at")
    if not started_at:
        return "00:00:00"
    try:
        started = datetime.fromisoformat(started_at)
        elapsed = max((datetime.now(started.tzinfo) - started).total_seconds(), 0)
    except (TypeError, ValueError):
        return "00:00:00"
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@st.fragment(run_every=2)
def _render_auto_judge_task_status(run_id: str) -> None:
    task = get_active_task()
    if not task or task.get("kind") != "auto_judge":
        return

    context = task.get("context") or {}
    task_run_id = context.get("eval_run_id")
    if task_run_id and task_run_id != run_id:
        return

    with st.container(border=True):
        st.markdown("#### :material/progress_activity: 自动机评后台进度")
        status_col, phase_col, elapsed_col = st.columns(3)
        status_col.metric("状态", task.get("status", "queued"))
        phase_col.metric(
            "当前阶段",
            "自动机评" if task.get("phase") == "auto_judge" else task.get("phase", "准备中"),
        )
        elapsed_col.metric("已耗时", _format_elapsed(task))

        current_asset = task.get("current_asset")
        if current_asset:
            st.caption(f"当前素材：{current_asset}")

        overall_done = int(task.get("overall_done") or 0)
        overall_total = int(task.get("overall_total") or 0)
        if overall_total:
            st.progress(
                min(overall_done / overall_total, 1.0),
                text=f"自动机评总体进度 {overall_done}/{overall_total}",
            )
        else:
            st.progress(0.0, text="正在准备评测素材")

        heartbeat_at = task.get("heartbeat_at")
        heartbeat_label = "等待首次心跳"
        if heartbeat_at:
            try:
                heartbeat_label = datetime.fromisoformat(heartbeat_at).astimezone().strftime("%H:%M:%S")
            except (TypeError, ValueError):
                pass
        alive_label = "线程存活" if task.get("worker_alive", True) else "线程已停止"
        st.caption(f"最近后台心跳：{heartbeat_label} · {alive_label}")

        message = task.get("message")
        if message:
            st.info(str(message))

        events = [
            str(event.get("message"))
            for event in task.get("events", [])[-12:]
            if event.get("message")
        ]
        if events:
            with st.expander("最近评测事件", expanded=False):
                st.markdown("  \n".join(f"- {message}" for message in events[-8:]))


def render_auto_judge() -> None:
    st.subheader(":material/smart_toy: VLM 自动机评")
    st.caption("使用本地 Qwen3-VL-4B 模型对素材进行自动化多维度评分。视频会自动抽帧后再由VLM评分。结果与人评独立存储，可进行一致性对比。")

    providers = list_available_providers().get("judge", [])
    available = [p for p in providers if p["available"]]

    if not available:
        st.error(":material/error: 无可用的 Judge Provider")
        return

    judge_names = [p["display_name"] for p in available]
    judge_map = {p["display_name"]: p["name"] for p in available}

    runs = eval_repo.list_runs()
    if not runs:
        st.info(":material/info: 暂无评测任务，请先创建评测任务。")
        return

    pc1, pc2 = st.columns([1, 1])
    with pc1:
        selected_judge_label = st.selectbox("评分引擎", judge_names, index=0)
    with pc2:
        run_options = {r["name"] + f" ({r['id'][:8]})": r["id"] for r in runs}
        default_idx = 0
        current_run_id = st.session_state.get("_eval_run_id")
        if current_run_id:
            for i, rid in enumerate(run_options.values()):
                if rid == current_run_id:
                    default_idx = i
                    break
        selected_run_label = st.selectbox(
            "选择评测任务",
            list(run_options.keys()),
            index=default_idx,
        )
    selected_run_id = run_options[selected_run_label]
    st.session_state["_eval_run_id"] = selected_run_id

    run_data = eval_repo.get_run(selected_run_id)
    if not run_data:
        st.error("任务不存在")
        return

    rubric = run_data.get("rubric") or load_default_rubric()
    items = eval_repo.list_run_items(selected_run_id)
    pending = [it for it in items if it.get("auto_judged_at") is None]
    judged = [it for it in items if it.get("auto_judged_at") is not None]

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("总素材", len(items))
    ic2.metric("已机评", len(judged))
    ic3.metric("待机评", len(pending))
    ic4.metric("人评完成", sum(1 for it in items if it.get("status") == "submitted"))

    st.progress(len(judged) / len(items) if items else 0)

    bc1, bc2, _ = st.columns([1, 1, 3])
    with bc1:
        if st.button(":material/play_arrow: 开始自动评分", type="primary", use_container_width=True,
                     disabled=len(pending) == 0):
            _submit_batch_judge(
                pending,
                rubric,
                judge_map[selected_judge_label],
                selected_run_id,
                run_data.get("project_id"),
                force=False,
            )
    with bc2:
        if st.button(":material/refresh: 重新评分全部", use_container_width=True):
            _submit_batch_judge(
                items,
                rubric,
                judge_map[selected_judge_label],
                selected_run_id,
                run_data.get("project_id"),
                force=True,
            )

    _render_auto_judge_task_status(selected_run_id)

    if judged:
        st.markdown("---")
        st.markdown("#### :material/psychology_alt: 人评机评一致性")
        from app.evaluation.repository import get_agreement_analysis
        agree = get_agreement_analysis(selected_run_id)
        if agree.get("n_pairs", 0) >= 2:
            ac1, ac2, ac3, ac4 = st.columns(4)
            pearson = agree["pearson_r"]
            spearman = agree["spearman_rho"]
            ac1.metric("Pearson r", f"{pearson:.3f}" if pearson is not None else "N/A")
            ac2.metric("Spearman ρ", f"{spearman:.3f}" if spearman is not None else "N/A")
            ac3.metric("MAE", f"{agree['mae']:.2f}" if agree['mae'] is not None else "N/A")
            ac4.metric("±1分一致率", f"{agree['within_one_rate']*100:.0f}%" if agree['within_one_rate'] is not None else "N/A")

            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("人评均分", f"{agree['human_avg']:.2f}")
            with sc2:
                st.metric("机评均分", f"{agree['auto_avg']:.2f}")

            if pearson is not None:
                if pearson >= 0.7:
                    st.success(f":material/check_circle: 人评机评一致性良好（r={pearson:.2f}），VLM评分具有参考价值。")
                elif pearson >= 0.4:
                    st.warning(f":material/warning: 人评机评一致性中等（r={pearson:.2f}），VLM评分可作为初筛参考。")
                else:
                    st.error(f":material/error: 人评机评一致性较低（r={pearson:.2f}），建议检查Rubric或模型效果。")

            if agree.get("dimension_agreement"):
                st.markdown("##### 各维度一致性")
                dim_label_map = {d.key: d.label for d in rubric.dimensions}
                dim_data = []
                for dk, stats in agree["dimension_agreement"].items():
                    label = dim_label_map.get(dk, dk)
                    if stats.get("pearson") is not None:
                        dim_data.append({"维度": label, "样本n": stats["n"], "Pearson r": stats["pearson"], "MAE": stats["mae"], "±1分率": f"{stats['within_one_rate']*100:.0f}%"})
                if dim_data:
                    st.dataframe(dim_data, use_container_width=True, hide_index=True)
        else:
            st.info(f":material/info: {agree.get('note', '需要至少2个同时有人评和机评的样本才能计算一致性。')}")

        st.markdown("---")
        st.markdown("#### :material/list: 机评结果列表")
        for it in judged[:20]:
            auto_total = it.get("auto_total_score", 0) or 0
            human_total = it.get("total_score")
            delta = f" (Δ{auto_total - human_total:+.1f})" if human_total is not None else ""
            with st.expander(f"{Path(it['asset_path']).name} · 机评 {auto_total:.1f}{delta} · {it.get('auto_judge_provider','')}"):
                vc1, vc2 = st.columns([1, 2])
                with vc1:
                    fp = Path(it["asset_path"])
                    if fp.exists():
                        st.image(str(fp), use_container_width=True)
                with vc2:
                    st.caption(f"**Prompt**: {it.get('prompt','')[:200]}")
                    if it.get("auto_explanation"):
                        st.markdown(f"**机评评价**: {it['auto_explanation']}")
                    st.caption(f"置信度: {it.get('auto_confidence', 0)*100:.0f}%")
                    auto_scores = it.get("auto_scores", {})
                    if auto_scores:
                        dim_label_map = {d.key: d.label for d in rubric.dimensions}
                        score_cols = st.columns(3)
                        for i, (dk, sdata) in enumerate(auto_scores.items()):
                            with score_cols[i % 3]:
                                label = dim_label_map.get(dk, dk)
                                sv = sdata.get("score", 0) if isinstance(sdata, dict) else sdata
                                st.metric(label, f"{sv}分")
                    auto_tags = it.get("auto_bad_case_tags", [])
                    if auto_tags:
                        tag_names = [t.get("tag","") if isinstance(t,dict) else str(t) for t in auto_tags]
                        st.caption("Bad Case: " + "、".join(f":orange[{t}]" for t in tag_names if t))


def _submit_batch_judge(
    items: list[dict[str, Any]],
    rubric: Any,
    provider_name: str,
    run_id: str,
    project_id: str | None,
    *,
    force: bool,
) -> None:
    if not items:
        st.info("没有需要自动评测的素材")
        return
    rubric_dict = rubric.to_dict()

    def runner(emit):
        return _run_batch_judge(
            items,
            rubric_dict,
            provider_name,
            emit,
        )

    try:
        task = submit_task(
            "auto_judge",
            f"{'重新' if force else ''}自动机评 · {len(items)} 个素材",
            runner,
            project_id,
            context={
                "app_view": "evaluation",
                "eval_view": "auto",
                "eval_run_id": run_id,
            },
        )
    except TaskBusyError as exc:
        st.warning(
            f":material/lock: 当前已有任务运行："
            f"**{exc.active_task.get('title', '生成任务')}**。"
            "自动机评需要独占模型内存，请等待当前任务完成。"
        )
        return
    st.session_state["_auto_judge_task_id"] = task["id"]
    st.session_state["_eval_run_id"] = run_id
    st.rerun()


def _run_batch_judge(
    items: list[dict[str, Any]],
    rubric_dict: dict[str, Any],
    provider_name: str,
    emit,
) -> dict[str, Any]:
    from app.config import JUDGE_MAX_RETRIES, QWEN_JUDGE_RELEASE_EACH_ITEM
    from app.evaluation.repository import save_auto_scores
    from app.providers import get_judge_provider

    judge = get_judge_provider(provider_name)
    if hasattr(judge, "set_status_callback"):
        judge.set_status_callback(emit)

    success = 0
    failed = 0
    failures: list[dict[str, str]] = []
    emit({
        "type": "judge_batch_start",
        "total": len(items),
        "message": f"开始串行自动机评，共 {len(items)} 个素材",
    })

    for i, item in enumerate(items):
        asset_path = Path(item["asset_path"])
        prompt = item.get("prompt", "") or ""
        task_type = item.get("asset_type", "keyframe")
        emit({
            "type": "judge_item_start",
            "index": i + 1,
            "total": len(items),
            "asset_name": asset_path.name,
            "message": f"正在评测 {i + 1}/{len(items)}：{asset_path.name}",
        })

        try:
            result = judge.evaluate_validated(
                image_path=asset_path,
                prompt=prompt,
                task_type=task_type,
                rubric=rubric_dict,
                reference_images=[
                    Path(path)
                    for path in item.get("reference_images", [])
                    if Path(path).exists()
                ],
                max_retries=JUDGE_MAX_RETRIES,
            )

            save_auto_scores(
                item_id=item["id"],
                dim_scores=result.dimension_scores,
                bad_case_tags=result.bad_case_tags,
                total_score=result.total_score,
                explanation=result.explanation,
                confidence=result.confidence,
                provider_name=judge.name,
                raw_response=result.raw_response,
            )
            success += 1
            device_mode = getattr(judge, "_device_override", None) or "mps"
            emit({
                "type": "judge_item_done",
                "index": i + 1,
                "total": len(items),
                "asset_name": asset_path.name,
                "score": result.total_score,
                "device": device_mode,
                "message": (
                    f"[{i + 1}/{len(items)}] {asset_path.name} → "
                    f"{result.total_score:.1f}分（{device_mode.upper()}）"
                ),
            })

        except Exception as e:
            failed += 1
            failures.append({"asset": asset_path.name, "error": str(e)})
            emit({
                "type": "judge_item_failed",
                "index": i + 1,
                "total": len(items),
                "asset_name": asset_path.name,
                "message": f"[{i + 1}/{len(items)}] {asset_path.name} → 失败：{e}",
            })
        finally:
            if QWEN_JUDGE_RELEASE_EACH_ITEM:
                judge.release()

    summary = {
        "success": success,
        "failed": failed,
        "total": len(items),
        "failures": failures,
    }
    emit({
        "type": "judge_batch_complete",
        **summary,
        "message": f"批量机评完成：成功 {success}，失败 {failed}",
    })
    return summary
