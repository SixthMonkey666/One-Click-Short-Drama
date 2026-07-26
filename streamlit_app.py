from __future__ import annotations

import atexit
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generator

import streamlit as st


def _cleanup_on_exit() -> None:
    try:
        import gc

        from app.providers import get_image_provider, get_llm_provider, release_all
        try:
            get_llm_provider().release()
        except Exception:
            pass
        try:
            get_image_provider().release()
        except Exception:
            pass
        release_all()
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
    except Exception:
        pass


atexit.register(_cleanup_on_exit)


def _signal_handler(signum: int, frame: Any) -> None:
    _cleanup_on_exit()
    raise SystemExit(0)


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _signal_handler)
    except (ValueError, OSError):
        pass


from app.adapters import generate_random_idea_stream
from app.agent import clear_agent_thread, get_agent_state, stream_agent
from app.config import (
    DEFAULT_IMAGE_RESOLUTION,
    IMAGE_RESOLUTION_PRESETS,
)
from app.evaluation.ui_manual import render_evaluation_center
from app.providers import get_configured_provider_status
from app.repository import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    rename_project,
    save_project,
)
from app.task_manager import (
    TaskBusyError,
    get_active_task,
    get_latest_task,
    get_task,
    submit_generator,
    submit_task,
)
from app.workflow import (
    approve_analysis,
    approve_storyboard,
    generate_assets_stream,
    generate_storyboard_from_analysis_stream,
    regenerate_character_images_stream,
    render_video_stream,
)

st.set_page_config(
    page_title="一键成片",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
    }
    hr {
        margin-top: 0.6rem !important;
        margin-bottom: 0.6rem !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }
    pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }
    pre > code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    div[data-testid="stCodeBlock"] pre {
        white-space: pre-wrap !important;
    }
</style>
""", unsafe_allow_html=True)

_provider_status = get_configured_provider_status()


def _provider_label(kind: str) -> str:
    provider = _provider_status[kind]
    suffix = "" if provider["available"] else "（不可用）"
    return f"{provider['display_name']}{suffix}"


script_engine = _provider_label("llm")
image_engine = _provider_label("image")
video_engine = _provider_label("video")


def _format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return iso_str


_STATUS_META = {
    "new": (":material/fiber_new:", "未开始"),
    "analyzing_idea": (":material/psychology:", "分析中"),
    "awaiting_analysis_review": (":material/psychology:", "待确认分析"),
    "generating_storyboard": (":material/assignment:", "生成分镜"),
    "awaiting_storyboard_review": (":material/assignment:", "待确认分镜"),
    "generating_assets": (":material/palette:", "生成资产"),
    "generating_video_prompts": (":material/movie:", "生成视频提示"),
    "awaiting_video_prompt_review": (":material/movie:", "待确认视频提示"),
    "rendering_video": (":material/theaters:", "渲染视频"),
    "agent_planning": (":material/account_tree:", "Agent 规划"),
    "agent_storyboard_evaluation": (":material/fact_check:", "Agent 分镜评测"),
    "agent_asset_evaluation": (":material/fact_check:", "Agent 资产评测"),
    "agent_error": (":material/error:", "Agent 执行失败"),
    "completed": (":material/check_circle:", "已完成"),
}


def _status_icon(status: str) -> str:
    return _STATUS_META.get(status, ("•", status))[0]


def _status_label(status: str) -> str:
    return _STATUS_META.get(status, ("•", status))[1]


def _resolution_options() -> list[str]:
    return list(IMAGE_RESOLUTION_PRESETS.keys())


def _auto_title(text: str) -> str:
    import re
    t = text.strip().strip("。！？!?，,；;：:\n\r\t ")
    if not t:
        return "·未命名项目"
    t = re.sub(r"\s+", "", t)
    for sep in ["。", "！", "？", "!", "?", "；", ";", "，", ","]:
        if sep in t:
            first = t.split(sep)[0]
            if len(first) >= 4:
                t = first
                break
    if len(t) > 12:
        t = t[:12]
    return "·" + t


def _find_asset_by_id(analysis: dict[str, Any] | None, aid: str) -> tuple[str, dict[str, Any]] | None:
    if not analysis:
        return None
    for c in analysis.get("characters", []):
        if c["id"] == aid:
            return ("character", c)
    for p in analysis.get("props", []):
        if p["id"] == aid:
            return ("prop", p)
    for e in analysis.get("environments", []):
        if e["id"] == aid:
            return ("environment", e)
    return None


def _asset_names_from_ids(analysis: dict[str, Any] | None, ids: list[str]) -> str:
    if not analysis or not ids:
        return ""
    names = []
    for aid in ids:
        found = _find_asset_by_id(analysis, aid)
        if found:
            names.append(found[1].get("name", aid))
        else:
            names.append(aid)
    return ", ".join(names)


def _env_name(analysis: dict[str, Any] | None, eid: str) -> str:
    if not analysis:
        return eid
    for e in analysis.get("environments", []):
        if e["id"] == eid:
            return e.get("name", eid)
    return eid


def _char_type_label(t: str) -> str:
    return ":material/person:" if t == "人物" else ":material/pets:"


def _get_image_path(img_dict: dict[str, Any] | None) -> str | None:
    if not img_dict:
        return None
    p = img_dict.get("path")
    if p and Path(p).exists():
        return p
    return None


def _legacy_run_pipeline_stream(gen_factory: Callable[[], Generator[dict[str, Any], None, None]], title: str = "正在生成..."):
    status = st.status(title, expanded=True)
    text_so_far = ""
    progress_bar = None
    progress_text = None
    current_asset = ""
    overall_bar = None
    overall_text = None
    log_area = status.empty()
    logs: list[str] = []
    pipeline_error: str | None = None
    st.session_state["_pipeline_error"] = None

    try:
        gen = gen_factory()
        while True:
            try:
                event = next(gen)
            except StopIteration:
                break

            etype = event.get("type", "")
            if etype == "status":
                msg = event.get("message", "")
                logs.append(f":material/info: {msg}")
                log_area.markdown("\n\n".join(logs[-6:]))
            elif etype == "text_start":
                text_so_far = ""
            elif etype == "text_delta":
                text_so_far += event.get("delta", "")
                log_area.code(text_so_far[-800:], language="json")
            elif etype == "text_end":
                pass
            elif etype in ("asset_start", "keyframe_start"):
                aname = event.get("asset_name") or event.get("entity_name", "")
                atype = event.get("asset_type", event.get("entity", ""))
                current_asset = f"{atype} · {aname}"
                if progress_bar is None:
                    progress_bar = status.progress(0)
                    progress_text = status.empty()
                progress_text.markdown(f":material/palette: 正在生成：**{current_asset}**")
                progress_bar.progress(0)
            elif etype in ("asset_progress", "keyframe_progress"):
                step = event.get("step", 0)
                total = event.get("total_steps", 4)
                if total > 0 and progress_bar is not None:
                    progress_bar.progress(min(1.0, step / total))
            elif etype in ("asset_done", "keyframe_done"):
                if progress_bar is not None:
                    progress_bar.progress(1.0)
                logs.append(f":material/check_circle: {current_asset} 完成")
                log_area.markdown("\n\n".join(logs[-6:]))
            elif etype == "assets_start":
                logs.append(f":material/rocket_launch: 开始资产生成：角色 {event.get('characters', 0)} · 道具 {event.get('props', 0)} · 场景 {event.get('environments', 0)} · 关键帧 {event.get('keyframes', 0)}")
                overall_bar = status.progress(0)
                overall_text = status.empty()
                log_area.markdown("\n\n".join(logs[-6:]))
            elif etype == "asset_progress_overall":
                done = event.get("done", 0)
                total = event.get("total", 1)
                msg = event.get("message", "")
                if overall_bar is not None:
                    overall_bar.progress(min(1.0, done / max(total, 1)))
                if overall_text is not None:
                    overall_text.markdown(f":material/bar_chart: 进度：{done}/{total} — {msg}")
                logs.append(f":material/check_circle: {msg}")
                log_area.markdown("\n\n".join(logs[-6:]))
            elif etype == "phase_start":
                phase = event.get("phase", "")
                name = event.get("name", "")
                icons = {"characters": ":material/person:", "props": ":material/inventory_2:",
                         "environments": ":material/landscape:", "keyframes": ":material/movie:"}
                icon = icons.get(phase, "•")
                logs.append(f"{icon} 开始生成：**{name}**")
                log_area.markdown("\n\n".join(logs[-6:]))
            elif etype in ("analysis_complete", "storyboard_complete", "assets_complete", "video_prompts_complete", "video_complete"):
                logs.append(f":material/celebration: {etype.replace('_', ' ').title()} 完成！")
                log_area.markdown("\n\n".join(logs[-6:]))
            elif etype == "error":
                pipeline_error = event.get("message", "生成出错")
                st.session_state["_pipeline_error"] = pipeline_error
                logs.append(f":material/close: 错误：{pipeline_error}")
                log_area.markdown("\n\n".join(logs[-6:]))
                st.error(pipeline_error)
    except Exception as e:
        pipeline_error = str(e)
        st.session_state["_pipeline_error"] = pipeline_error
        st.error(f"生成失败：{pipeline_error}")
        traceback.print_exc()
    finally:
        try:
            if pipeline_error:
                status.update(label="生成失败", state="error", expanded=True)
            else:
                status.update(label="完成", state="complete", expanded=True)
        except Exception:
            pass
    if pipeline_error is None:
        st.rerun()


def _legacy_run_video_agent(
    project_id: str,
    decision: dict[str, Any] | None = None,
    title: str = "Agent 正在执行...",
) -> None:
    node_labels = {
        "requirement_analysis": "理解创意需求",
        "planner": "制定执行计划",
        "visual_bible": "整理视觉设定",
        "storyboard_generation": "生成分镜脚本",
        "storyboard_evaluation": "评测分镜",
        "storyboard_reflection": "反思并修复分镜",
        "storyboard_human_review": "等待分镜审核",
        "asset_planning": "规划生成资产",
        "asset_generation": "生成角色、道具、场景和关键帧",
        "asset_evaluation": "评测生成资产",
        "asset_repair": "修复失败资产",
        "video_prompt_generation": "生成视频提示词",
        "video_prompt_evaluation": "评测视频提示词",
        "render_approval": "等待渲染确认",
        "video_render": "合成最终视频",
        "final_evaluation": "最终质量检查",
        "result_packaging": "整理交付结果",
    }
    phase_labels = {
        "queue_wait": "等待推理队列",
        "releasing_qwen": "释放语言模型内存",
        "memory_wait": "等待可用内存",
        "model_loading": "加载 FLUX 模型",
        "model_ready": "模型已就绪",
        "denoising": "图像去噪",
        "vae_decode": "VAE 解码",
        "saving": "保存图像",
        "oom_retry": "内存降级重试",
        "completed": "单张图像完成",
    }
    started = time.monotonic()
    logs: list[str] = []

    def elapsed_text(seconds: float | None = None) -> str:
        value = time.monotonic() - started if seconds is None else seconds
        minutes, secs = divmod(int(value), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    with st.status(title, expanded=True) as status_widget:
        phase_slot = status_widget.empty()
        detail_slot = status_widget.empty()
        progress_slot = status_widget.empty()
        heartbeat_slot = status_widget.empty()
        log_slot = status_widget.empty()
        progress_bar = progress_slot.progress(0, text="等待第一个进度事件")

        def append_log(message: str) -> None:
            stamp = datetime.now().strftime("%H:%M:%S")
            logs.append(f"`{stamp}` {message}")
            log_slot.markdown("  \n".join(logs[-10:]))

        def render_pipeline_event(event: dict[str, Any]) -> None:
            event_type = event.get("type")
            asset_name = str(event.get("asset_name") or event.get("entity_name") or "")
            view_label = str(event.get("view_label") or "")
            subject = " · ".join(part for part in (asset_name, view_label) if part)
            if event_type in {"asset_start", "keyframe_start"}:
                label = subject or str(event.get("shot_id") or "新资产")
                phase_slot.markdown(f"#### :material/image: 正在处理：{label}")
                size = f"{event.get('width', '?')}×{event.get('height', '?')}"
                detail_slot.caption(f"准备生成 · 输出尺寸 {size}")
                progress_bar.progress(0, text="准备模型")
                append_log(f"开始生成 **{label}**")
            elif event_type == "assets_start":
                total = int(event.get("total") or 0)
                phase_slot.markdown("#### :material/palette: 开始生成全部资产")
                detail_slot.info(
                    f"共 {total} 张：角色 {event.get('characters', 0)} · "
                    f"道具 {event.get('props', 0)} · 场景 {event.get('environments', 0)} · "
                    f"关键帧 {event.get('keyframes', 0)}"
                )
                append_log(f"资产批次开始，共 **{total}** 张")
            elif event_type == "asset_progress_overall":
                done = int(event.get("done") or 0)
                total = max(int(event.get("total") or 1), 1)
                progress_bar.progress(
                    min(done / total, 1.0),
                    text=f"全部资产 {done}/{total}",
                )
                append_log(str(event.get("message") or f"完成 {done}/{total}"))
            elif event_type == "assets_complete":
                progress_bar.progress(1.0, text="全部资产生成完成")
                append_log(":material/check_circle: **全部资产生成完成**")
            elif event_type == "generation_phase":
                phase = str(event.get("phase") or "")
                phase_name = phase_labels.get(phase, phase or "处理中")
                message = str(event.get("message") or phase_name)
                phase_slot.markdown(f"#### :material/progress_activity: {phase_name}")
                detail_slot.info(f"{subject + ' · ' if subject else ''}{message}")
                heartbeat_slot.caption(
                    f"后台事件刚刚更新 · 本次任务已运行 {elapsed_text()}"
                )
                append_log(f"**{phase_name}**：{message}")
            elif event_type in {"asset_progress", "keyframe_progress"}:
                step = int(event.get("step") or 0)
                total = max(int(event.get("total_steps") or 1), 1)
                progress_bar.progress(
                    min(step / total, 1.0),
                    text=f"去噪进度 {step}/{total}",
                )
                heartbeat_slot.caption(
                    f"后台事件刚刚更新 · 本次任务已运行 {elapsed_text()}"
                )
            elif event_type == "generation_heartbeat":
                phase = str(event.get("phase") or "处理中")
                message = str(event.get("message") or "")
                phase_slot.markdown(
                    f"#### :material/favorite: 后台仍在运行 · {phase_labels.get(phase, phase)}"
                )
                detail_slot.info(f"{subject + ' · ' if subject else ''}{message}")
                heartbeat_slot.caption(
                    f"最近心跳：{datetime.now().strftime('%H:%M:%S')} · "
                    f"当前单张耗时 {elapsed_text(float(event.get('elapsed_seconds') or 0))} · "
                    f"本次任务总耗时 {elapsed_text()}"
                )
            elif event_type in {"asset_done", "keyframe_done"}:
                progress_bar.progress(1.0, text="当前图像已完成")
                append_log(f":material/check_circle: **{subject or '当前图像'}** 生成完成")
            elif event_type == "status" and event.get("message"):
                detail_slot.info(str(event["message"]))
                append_log(str(event["message"]))

        try:
            for stream_item in stream_agent(project_id, decision=decision):
                mode = stream_item.get("stream_mode")
                update = stream_item.get("data")
                if mode == "custom" and isinstance(update, dict):
                    if update.get("type") == "pipeline_event":
                        event = update.get("event")
                        if isinstance(event, dict):
                            render_pipeline_event(event)
                    elif update.get("type") == "node_started":
                        node = str(update.get("node") or "")
                        label = node_labels.get(node, node)
                        phase_slot.markdown(f"#### :material/play_circle: {label}")
                        detail_slot.caption(f"Agent 节点 `{node}` 已开始")
                        append_log(f"进入 **{label}**")
                    elif update.get("type") == "node_completed":
                        node = str(update.get("node") or "")
                        label = node_labels.get(node, node)
                        append_log(
                            f":material/check_circle: **{label}** 完成，"
                            f"耗时 {elapsed_text(float(update.get('duration_seconds') or 0))}"
                        )
                    elif update.get("type") == "node_failed":
                        append_log(
                            f":material/error: **{update.get('node')}** 失败：{update.get('error')}"
                        )
                    continue
                if mode != "updates" or not isinstance(update, dict):
                    continue
                if "__interrupt__" in update:
                    status_widget.write(":material/pause_circle: 已到达人工审核节点")
                    continue
                for node_name, payload in update.items():
                    if isinstance(payload, dict):
                        evaluation = payload.get("evaluation_results", {})
                        latest = evaluation.get("storyboard") or evaluation.get("assets")
                        if latest and latest.get("score") is not None:
                            append_log(f"评测分数：**{latest['score']} / 5**")
            status_widget.update(label="Agent 已暂停等待审核或执行完成", state="complete")
            st.session_state["_pipeline_error"] = None
        except Exception as exc:
            st.session_state["_pipeline_error"] = str(exc)
            status_widget.update(label="Agent 执行失败", state="error")
            st.error(f"Agent 执行失败：{exc}")
            traceback.print_exc()
            return
    st.rerun()


def _safe_agent_state(project_id: str) -> dict[str, Any]:
    try:
        return get_agent_state(project_id)
    except Exception:
        return {"values": {}, "next": [], "interrupts": [], "waiting_for_human": False}


def _submit_error(exc: TaskBusyError) -> None:
    task = exc.active_task
    project_hint = f"（项目 {str(task.get('project_id') or '')[:8]}）" if task.get("project_id") else ""
    st.warning(
        f":material/lock: 当前已有唯一生成任务在运行："
        f"**{task.get('title', '生成任务')}**{project_hint}。请等待它完成后再提交。"
    )


def run_pipeline_stream(
    gen_factory: Callable[[], Generator[dict[str, Any], None, None]],
    title: str = "正在生成...",
) -> None:
    project_id = st.session_state.get("selected_pid")
    try:
        task = submit_generator(
            kind="pipeline",
            title=title,
            generator_factory=gen_factory,
            project_id=project_id,
        )
    except TaskBusyError as exc:
        _submit_error(exc)
        return
    st.session_state["_last_task_id"] = task["id"]
    st.session_state["_pipeline_error"] = None
    st.rerun()


def run_video_agent(
    project_id: str,
    decision: dict[str, Any] | None = None,
    title: str = "Agent 正在执行...",
) -> None:
    def runner(emit):
        for item in stream_agent(project_id, decision=decision):
            mode = item.get("stream_mode")
            payload = item.get("data")
            if mode == "custom" and isinstance(payload, dict):
                if payload.get("type") == "pipeline_event" and isinstance(payload.get("event"), dict):
                    emit(payload["event"])
                else:
                    emit(payload)
            elif mode == "updates" and isinstance(payload, dict) and "__interrupt__" in payload:
                emit({"type": "agent_interrupt", "message": "Agent 已暂停，等待人工审核"})
        return _safe_agent_state(project_id)

    try:
        task = submit_task("agent", title, runner, project_id)
    except TaskBusyError as exc:
        _submit_error(exc)
        return
    st.session_state["_last_task_id"] = task["id"]
    st.session_state["_pipeline_error"] = None
    st.rerun()


def _start_idea_generation() -> None:
    try:
        task = submit_generator(
            kind="idea",
            title="正在生成智能创意",
            generator_factory=generate_random_idea_stream,
        )
    except TaskBusyError as exc:
        _submit_error(exc)
        return
    st.session_state["_idea_task_id"] = task["id"]
    st.session_state["_idea_candidates"] = None
    st.session_state["_idea_generation_error"] = None
    st.rerun()


def _format_task_elapsed(task: dict[str, Any]) -> str:
    started_at = task.get("started_at") or task.get("created_at")
    if not started_at:
        return "00:00:00"
    try:
        started = datetime.fromisoformat(started_at)
        end = (
            datetime.fromisoformat(task["finished_at"])
            if task.get("finished_at")
            else datetime.now(started.tzinfo)
        )
        seconds = max(0, int((end - started).total_seconds()))
    except (TypeError, ValueError):
        seconds = 0
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _task_event_text(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    if event_type == "node_started":
        return f"Agent 节点开始：{event.get('node', '')}"
    if event_type == "node_completed":
        return f"Agent 节点完成：{event.get('node', '')}"
    if event_type in {"asset_start", "keyframe_start"}:
        name = event.get("asset_name") or event.get("entity_name") or event.get("shot_id")
        view = event.get("view_label") or event.get("view") or ""
        return f"开始生成：{name}{' · ' + str(view) if view else ''}"
    if event_type in {"asset_done", "keyframe_done"}:
        return "当前图像生成完成"
    if event_type == "asset_progress_overall":
        return str(event.get("message") or "")
    if event_type in {"generation_phase", "generation_heartbeat", "status", "task_failed"}:
        return str(event.get("message") or "")
    if event_type in {
        "judge_batch_start",
        "judge_item_start",
        "judge_item_done",
        "judge_item_failed",
        "judge_batch_complete",
    }:
        return str(event.get("message") or "")
    if event_type == "task_heartbeat":
        return "后台工作线程仍在运行" if event.get("worker_alive") else "后台工作线程已停止"
    if event_type in {"analysis_complete", "storyboard_complete", "assets_complete", "video_complete"}:
        return f"{event_type.replace('_', ' ')}"
    return None


def _render_task_snapshot(task: dict[str, Any], compact: bool = False) -> None:
    status = task.get("status")
    phase = str(task.get("phase") or "—")
    phase_label = {
        "reference_conditioning": "加载角色身份参考图",
        "asset_generation": "生成图像资产",
        "model_loading": "加载图像模型",
        "memory_wait": "等待可用内存",
        "denoising": "图像去噪",
        "vae_decode": "图像解码",
        "saving": "保存图像",
        "auto_judge": "自动机评",
        "auto_judge_complete": "自动机评完成",
    }.get(phase, phase)
    icon = {
        "queued": ":material/schedule:",
        "running": ":material/progress_activity:",
        "completed": ":material/check_circle:",
        "failed": ":material/error:",
        "interrupted": ":material/power_off:",
    }.get(status, ":material/info:")
    st.markdown(f"**{icon} {task.get('title', '生成任务')}**")
    current_asset = task.get("current_asset")
    current_view = task.get("current_view")
    if current_asset:
        st.caption(
            f"当前对象：{current_asset}"
            f"{' · ' + str(current_view) if current_view else ''}"
        )
    st.caption(
        f"状态：{status} · 阶段：{phase_label} · "
        f"耗时：{_format_task_elapsed(task)}"
    )
    if status in ACTIVE_TASK_STATUSES:
        heartbeat_at = task.get("heartbeat_at")
        heartbeat_label = "等待首次心跳"
        if heartbeat_at:
            try:
                heartbeat_label = datetime.fromisoformat(heartbeat_at).astimezone().strftime("%H:%M:%S")
            except (TypeError, ValueError):
                pass
        alive_label = "线程存活" if task.get("worker_alive", True) else "线程已停止"
        st.caption(f"最近后台心跳：{heartbeat_label} · {alive_label}")
    overall_total = int(task.get("overall_total") or 0)
    if overall_total:
        overall_done = int(task.get("overall_done") or 0)
        st.progress(
            min(overall_done / overall_total, 1.0),
            text=f"全部进度 {overall_done}/{overall_total}",
        )
    total_steps = int(task.get("total_steps") or 0)
    if total_steps:
        step = int(task.get("step") or 0)
        st.progress(min(step / total_steps, 1.0), text=f"当前图像去噪 {step}/{total_steps}")
    message = task.get("message")
    if message:
        if status == "failed":
            st.error(message)
        elif status == "interrupted":
            st.warning(message)
        else:
            st.info(message)
    if not compact:
        messages = [
            text
            for text in (
                _task_event_text(event) for event in task.get("events", [])[-30:]
            )
            if text
        ]
        if messages:
            with st.expander("最近关键事件", expanded=status in ACTIVE_TASK_STATUSES):
                st.markdown("  \n".join(f"- {item}" for item in messages[-12:]))


ACTIVE_TASK_STATUSES = {"queued", "running"}


def _task_target(task: dict[str, Any]) -> dict[str, Any]:
    context = task.get("context") or {}
    if task.get("kind") == "auto_judge":
        return {
            "app_view": "evaluation",
            "eval_view": "auto",
            "eval_run_id": (
                context.get("eval_run_id")
                or st.session_state.get("_eval_run_id")
            ),
            "label": "自动机评任务",
        }
    return {
        "app_view": "workflow",
        "project_id": task.get("project_id"),
        "label": "一键成片任务",
    }


def _is_on_task_target(target: dict[str, Any]) -> bool:
    if st.session_state.get("_app_view") != target["app_view"]:
        return False
    if target["app_view"] == "evaluation":
        if st.session_state.get("_eval_view") != target.get("eval_view"):
            return False
        target_run_id = target.get("eval_run_id")
        return (
            target_run_id is None
            or st.session_state.get("_eval_run_id") == target_run_id
        )
    return st.session_state.get("selected_pid") == target.get("project_id")


def _navigate_to_task_target(target: dict[str, Any]) -> None:
    st.session_state["_app_view"] = target["app_view"]
    if target["app_view"] == "evaluation":
        st.session_state["_eval_view"] = target.get("eval_view", "auto")
        st.session_state["_eval_nav_radio"] = ":material/smart_toy: 自动机评"
        if target.get("eval_run_id"):
            st.session_state["_eval_run_id"] = target["eval_run_id"]
    else:
        st.session_state["selected_pid"] = target.get("project_id")


@st.fragment(run_every=2)
def render_global_task_status(compact: bool = False, project_id: str | None = None) -> None:
    active = get_active_task()
    task = active
    if active:
        st.session_state["_global_active_task_id"] = active["id"]
    if task is None and project_id:
        task = get_latest_task(project_id=project_id)
    if task is None:
        previous_task_id = st.session_state.get("_global_active_task_id")
        if previous_task_id:
            task = get_task(previous_task_id)
    if task is None:
        return

    previous_status = st.session_state.get(f"_task_status_{task['id']}")
    st.session_state[f"_task_status_{task['id']}"] = task.get("status")
    if previous_status in ACTIVE_TASK_STATUSES and task.get("status") not in ACTIVE_TASK_STATUSES:
        if task.get("kind") == "idea":
            if task.get("status") == "completed":
                st.session_state["_idea_candidates"] = task.get("result")
                st.session_state["_idea_generation_error"] = None
            else:
                st.session_state["_idea_generation_error"] = task.get("message")
        st.session_state["_global_active_task_id"] = None
        st.rerun(scope="app")
    if compact and task.get("status") not in ACTIVE_TASK_STATUSES:
        return
    target = _task_target(task)
    already_on_task = _is_on_task_target(target)
    if compact:
        header_col, return_col = st.columns([4, 1], vertical_alignment="center")
        with header_col:
            st.caption("全局唯一生成任务")
        with return_col:
            if not already_on_task and st.button(
                ":material/arrow_back:",
                key=f"_return_to_active_task_{task['id']}",
                help=f"返回{target['label']}",
                type="primary",
                use_container_width=True,
            ):
                _navigate_to_task_target(target)
                st.rerun(scope="app")
    _render_task_snapshot(task, compact=compact)


if "selected_pid" not in st.session_state:
    st.session_state.selected_pid = None
if "_apply_idea_to_input" not in st.session_state:
    st.session_state._apply_idea_to_input = None
if "_clear_input" not in st.session_state:
    st.session_state._clear_input = False
if "_generating_random" not in st.session_state:
    st.session_state._generating_random = False
# Generation now belongs to the process-wide task manager, never to a page rerun.
st.session_state._generating_random = False
if "_idea_candidates" not in st.session_state:
    st.session_state._idea_candidates = None
if "_idea_generation_error" not in st.session_state:
    st.session_state._idea_generation_error = None
if "_random_idea_text" not in st.session_state:
    st.session_state._random_idea_text = ""
if "_proj_search" not in st.session_state:
    st.session_state._proj_search = ""
if "_selected_proj_ids" not in st.session_state:
    st.session_state._selected_proj_ids = set()
if "_chk_version" not in st.session_state:
    st.session_state._chk_version = 0
if "_app_view" not in st.session_state:
    st.session_state._app_view = "workflow"
if "_mgmt_rename_pid" not in st.session_state:
    st.session_state._mgmt_rename_pid = None
if "_pipeline_error" not in st.session_state:
    st.session_state._pipeline_error = None

if st.session_state._pipeline_error:
    st.error(f"上次生成失败：{st.session_state._pipeline_error}")

_unavailable_providers = [
    provider
    for provider in _provider_status.values()
    if not provider["available"]
]
if _unavailable_providers:
    details = "；".join(
        f"{provider['display_name']}：{provider['error']}"
        for provider in _unavailable_providers
    )
    st.warning(f"部分 Provider 当前不可用，页面仍可浏览和配置。开始生成前请检查：{details}")


def _bump_chk_version():
    st.session_state._chk_version = st.session_state.get("_chk_version", 0) + 1
    for k in list(st.session_state.keys()):
        if k.startswith("_chk_v"):
            del st.session_state[k]


def render_project_management():
    all_projects = list_projects()

    title_col, back_col = st.columns([5, 1.2])
    with title_col:
        st.header(":material/folder_open: 项目管理")
    with back_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(":material/arrow_back: 返回工作台", use_container_width=True, key="_mgmt_back"):
            st.session_state._app_view = "workflow"
            _bump_chk_version()
            st.rerun()

    search = st.text_input(":material/search: 搜索项目", value=st.session_state.get("_mgmt_search", ""),
                           placeholder="按标题搜索...", key="_mgmt_search_input")
    st.session_state._mgmt_search = search.strip().lower()

    if st.session_state._mgmt_search:
        mgmt_filtered = [p for p in all_projects if st.session_state._mgmt_search in p["title"].lower()]
    else:
        mgmt_filtered = all_projects

    mgmt_visible = {p["id"] for p in mgmt_filtered}
    st.session_state._selected_proj_ids &= mgmt_visible

    cv = st.session_state._chk_version

    if not all_projects:
        st.info("暂无项目")
        return

    if not mgmt_filtered:
        st.caption("未找到匹配的项目")
        return

    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        visible_chk_keys = {f"_chk_v{cv}_{p['id']}": p["id"] for p in mgmt_filtered}
        checked_visible = {pid for k, pid in visible_chk_keys.items() if st.session_state.get(k, False)}
        all_sel = bool(mgmt_visible) and checked_visible >= mgmt_visible
        if st.button("全选", use_container_width=True, disabled=all_sel, key="_mgmt_sel_all"):
            for p in mgmt_filtered:
                k = f"_chk_v{cv}_{p['id']}"
                st.session_state[k] = True
            st.rerun()
    with cols[1]:
        if st.button("反选", use_container_width=True, key="_mgmt_inv"):
            for p in mgmt_filtered:
                k = f"_chk_v{cv}_{p['id']}"
                st.session_state[k] = not st.session_state.get(k, False)
            st.rerun()
    with cols[2]:
        any_checked = any(st.session_state.get(f"_chk_v{cv}_{p['id']}", False) for p in mgmt_filtered)
        if st.button("清空", use_container_width=True, disabled=not any_checked, key="_mgmt_clr"):
            for p in mgmt_filtered:
                k = f"_chk_v{cv}_{p['id']}"
                st.session_state[k] = False
            st.rerun()
    with cols[3]:
        sel_count = sum(1 for p in mgmt_filtered if st.session_state.get(f"_chk_v{cv}_{p['id']}", False))
        st.session_state._selected_proj_ids = {p["id"] for p in mgmt_filtered if st.session_state.get(f"_chk_v{cv}_{p['id']}", False)}
        if sel_count > 0:
            if st.button(f":material/delete: 批量删除 ({sel_count})", use_container_width=True, type="primary", key="_mgmt_batch_del"):
                for did in list(st.session_state._selected_proj_ids):
                    delete_project(did)
                    if did == st.session_state.selected_pid:
                        st.session_state.selected_pid = None
                st.session_state._selected_proj_ids = set()
                _bump_chk_version()
                st.rerun()

    st.caption(f"共 {len(all_projects)} 个项目" +
               (f"，匹配 {len(mgmt_filtered)} 个" if st.session_state._mgmt_search else "") +
               (f"，已选 {sel_count} 个" if sel_count else ""))

    header_cols = st.columns([0.5, 2.6, 1, 0.8, 1, 2.5])
    header_cols[0].markdown("**选**")
    header_cols[1].markdown("**项目标题**")
    header_cols[2].markdown("**状态**")
    header_cols[3].markdown("**镜头**")
    header_cols[4].markdown("**更新**")
    header_cols[5].markdown("**操作**")
    st.divider()

    for p in mgmt_filtered:
        pid = p["id"]
        is_cur = pid == st.session_state.get("selected_pid")
        chk_key = f"_chk_v{cv}_{pid}"
        if chk_key not in st.session_state:
            st.session_state[chk_key] = pid in st.session_state._selected_proj_ids

        icon = _status_icon(p["status"])
        slabel = _status_label(p["status"])
        shots_n = len(p.get("shots", []))
        tstr = _format_time(p["updated_at"])
        title_disp = p["title"][:35] + ("…" if len(p["title"]) > 35 else "")
        if is_cur:
            title_disp = ":material/play_arrow: " + title_disp

        is_renaming = (st.session_state._mgmt_rename_pid == pid)
        if is_renaming:
            rcols = st.columns([0.5, 2.6, 1, 0.8, 1, 2.5], vertical_alignment="center")
            rcols[0].checkbox("选", key=chk_key, label_visibility="collapsed")
            ren_input = rcols[1].text_input("重命名", value=p["title"], key=f"_mgmt_inline_ren_{pid}",
                                            label_visibility="collapsed")
            rcols[2].markdown(slabel)
            rcols[3].markdown(f"{shots_n} 镜")
            rcols[4].markdown(tstr)
            bcols = rcols[5].columns([1, 1, 1])
            if bcols[0].button(":material/check:", key=f"_mgmt_renok_{pid}", use_container_width=True, type="primary"):
                if ren_input.strip():
                    rename_project(pid, ren_input.strip())
                st.session_state._mgmt_rename_pid = None
                st.rerun()
            if bcols[1].button(":material/close:", key=f"_mgmt_renc_{pid}", use_container_width=True):
                st.session_state._mgmt_rename_pid = None
                st.rerun()
            bcols[2].button("删除", key=f"_mgmt_rdel_{pid}", use_container_width=True, disabled=True)
        else:
            rcols = st.columns([0.5, 2.6, 1, 0.8, 1, 2.5], vertical_alignment="center")
            rcols[0].checkbox("选", key=chk_key, label_visibility="collapsed")
            rcols[1].markdown(f"{icon} **{title_disp}**")
            rcols[2].markdown(slabel)
            rcols[3].markdown(f"{shots_n} 镜")
            rcols[4].markdown(tstr)
            bcols = rcols[5].columns([1, 1, 1])
            if bcols[0].button("打开", key=f"_mgmt_open_{pid}", use_container_width=True,
                               type="primary" if is_cur else "secondary", disabled=is_cur):
                st.session_state.selected_pid = pid
                st.session_state._app_view = "workflow"
                _bump_chk_version()
                st.rerun()
            if bcols[1].button("重命名", key=f"_mgmt_ren_{pid}", use_container_width=True):
                st.session_state._mgmt_rename_pid = pid
                st.rerun()
            if bcols[2].button("删除", key=f"_mgmt_del_{pid}", use_container_width=True):
                delete_project(pid)
                if pid == st.session_state.selected_pid:
                    st.session_state.selected_pid = None
                st.session_state._selected_proj_ids.discard(pid)
                if chk_key in st.session_state:
                    del st.session_state[chk_key]
                if st.session_state._mgmt_rename_pid == pid:
                    st.session_state._mgmt_rename_pid = None
                st.rerun()

with st.sidebar:
    st.header(":material/assignment: 项目管理")

    projects = list_projects()
    current_pid = st.session_state.get("selected_pid")
    current_project = get_project(current_pid) if current_pid else None

    new_project_selected = (
        st.session_state.get("_app_view") == "workflow"
        and current_pid is None
    )
    if st.button(
        ":material/add: 新建项目",
        use_container_width=True,
        type="primary" if new_project_selected else "secondary",
    ):
        st.session_state.selected_pid = None
        st.session_state._app_view = "workflow"
        st.session_state._clear_input = True
        _bump_chk_version()
        st.rerun()

    if st.button(
        ":material/folder_open: 管理全部项目",
        use_container_width=True,
        type="primary" if st.session_state.get("_app_view") == "proj_mgmt" else "secondary",
    ):
        st.session_state._app_view = "proj_mgmt"
        st.rerun()

    if st.button(
        ":material/rate_review: 评测中心",
        use_container_width=True,
        type="primary" if st.session_state.get("_app_view") == "evaluation" else "secondary",
    ):
        st.session_state._app_view = "evaluation"
        st.session_state._eval_view = "runs"
        from app.evaluation.ui_manual import _NAV_KEYS, _NAV_LABELS, _NAV_RADIO_KEY
        st.session_state[_NAV_RADIO_KEY] = _NAV_LABELS[_NAV_KEYS.index("runs")]
        st.rerun()

    if get_active_task():
        st.divider()
        with st.container(border=True):
            render_global_task_status(compact=True)

    st.divider()

    if not projects:
        st.info("暂无项目，点击「新建项目」开始")
    else:
        st.caption(f"共 {len(projects)} 个项目 · 最近更新")
        recent = projects[:8]
        list_box = st.container(border=True, height=min(360, max(80, len(recent) * 50)))
        with list_box:
            for p in recent:
                pid = p["id"]
                is_cur = pid == current_pid and st.session_state._app_view == "workflow"
                icon = _status_icon(p["status"])
                shots_n = len(p.get("shots", []))
                tstr = _format_time(p["updated_at"])
                tdisp = p["title"][:22] + ("…" if len(p["title"]) > 22 else "")
                label = f"{icon} {tdisp}"
                if is_cur:
                    label = f":material/play_arrow: {icon} {tdisp}"
                if st.button(label, key=f"_sb_open_{pid}", use_container_width=True,
                             type="primary" if is_cur else "secondary"):
                    st.session_state.selected_pid = pid
                    st.session_state._app_view = "workflow"
                    st.rerun()
                st.caption(f"{tstr}  ·  {shots_n}镜  ·  {_status_label(p['status'])}")

        if len(projects) > 8:
            st.caption(f"还有 {len(projects) - 8} 个项目，点击「管理全部项目」查看")

    if current_project:
        with st.expander(":material/settings: 当前项目", expanded=False):
            new_title = st.text_input("项目标题", value=current_project["title"], key="rename_title")
            c1, c2 = st.columns(2)
            if c1.button(":material/edit: 重命名", use_container_width=True):
                rename_project(current_project["id"], new_title)
                st.rerun()
            if c2.button(":material/delete: 删除", use_container_width=True):
                delete_project(current_project["id"])
                st.session_state.selected_pid = None
                st.session_state._app_view = "workflow"
                st.session_state._selected_proj_ids.discard(current_project["id"])
                _bump_chk_version()
                st.rerun()

if st.session_state._app_view == "workflow":
    st.title("一键成片")
    st.caption(
        "输入一句创意 → AI 分析理解 → 分镜脚本 → 资产生成 → 一键出片。"
        f"脚本引擎：{script_engine}；图像引擎：{image_engine}；"
        f"视频引擎：{video_engine}。"
    )
    st.divider()

if st.session_state._app_view == "evaluation":
    if st.session_state.get("_eval_view") == "select_assets":
        from app.evaluation.ui_manual import _render_asset_selector
        _render_asset_selector()
    else:
        render_evaluation_center()
    st.stop()

if st.session_state._app_view == "proj_mgmt":
    render_project_management()
    st.stop()

if not current_project:
    st.subheader(":material/auto_awesome: 创建新项目")

    if st.session_state.get("_auto_create_idea"):
        idea_text = st.session_state._auto_create_idea
        st.session_state._auto_create_idea = None
        st.session_state["source_input"] = idea_text
        res_preset_auto = st.session_state.get("new_res", DEFAULT_IMAGE_RESOLUTION)
        project = create_project(_auto_title(idea_text), idea_text, resolution_preset=res_preset_auto)
        st.session_state.selected_pid = project["id"]
        st.rerun()

    input_val = ""
    if st.session_state._apply_idea_to_input:
        input_val = st.session_state._apply_idea_to_input
        st.session_state._apply_idea_to_input = None
        st.session_state["source_input"] = input_val
    elif st.session_state._clear_input:
        input_val = ""
        st.session_state._clear_input = False
        st.session_state["source_input"] = ""

    idea_col, rand_col = st.columns([5, 1])
    with idea_col:
        source_text = st.text_area(
            "输入你的视频创意", value=input_val, height=120,
            placeholder="例如：一只流浪橘猫在雨夜遇到了撑伞下班的女孩...", key="source_input")
    with rand_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(":material/casino: 智能创意", use_container_width=True):
            _start_idea_generation()

    idea_task_id = st.session_state.get("_idea_task_id")
    if idea_task_id:
        idea_task = get_task(idea_task_id)
        if idea_task and idea_task.get("status") == "completed" and not st.session_state.get("_idea_candidates"):
            st.session_state._idea_candidates = idea_task.get("result")
        elif idea_task and idea_task.get("status") in {"failed", "interrupted"}:
            st.session_state._idea_generation_error = idea_task.get("message")

    if st.session_state.get("_idea_generation_error"):
        st.error(st.session_state._idea_generation_error)

    if st.session_state.get("_idea_candidates") and not st.session_state.get("_generating_random"):
        candidates = st.session_state._idea_candidates
        st.markdown("---")
        st.markdown("<div style='display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;'>"
                    "<span style='font-size:1.1rem;font-weight:600;'>🎯 为你推荐3个创意方案</span>"
                    "<span style='color:#888;font-size:0.85rem;'>选择一个开始创作</span>"
                    "</div>", unsafe_allow_html=True)

        cols = st.columns(3)
        chosen = None
        for i, (col, idea) in enumerate(zip(cols, candidates, strict=False)):
            with col:
                topics = idea.get("topics", [])
                text = idea.get("text", "")
                tags_html = ""
                if topics:
                    tags_html = "<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:0.5rem;'>"
                    for t in topics:
                        tags_html += (
                            f"<span style='background:linear-gradient(135deg,#7757d7,#9b7ffc);"
                            f"color:white;padding:2px 8px;border-radius:10px;font-size:0.72rem;'>🔥 {t}</span>"
                        )
                    tags_html += "</div>"
                card_html = (
                    f"<div style='border:2px solid #e8e8ef;border-radius:12px;padding:0.9rem;"
                    f"background:#fafbfe;height:100%;transition:all 0.2s;'>"
                    f"<div style='font-size:0.85rem;font-weight:600;color:#7757d7;margin-bottom:0.4rem;'>方案 {i+1}</div>"
                    f"{tags_html}"
                    f"<div style='font-size:0.9rem;line-height:1.6;color:#1e1e2e;'>{text}</div>"
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(f":material/check_circle: 使用方案 {i+1}", key=f"use_idea_{i}", use_container_width=True, type="primary"):
                    chosen = text

        if chosen is not None:
            st.session_state._idea_candidates = None
            st.session_state._auto_create_idea = chosen
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        col_regen, col_empty = st.columns([1, 3])
        with col_regen:
            if st.button(":material/refresh: 换一批", use_container_width=True):
                _start_idea_generation()

    if st.session_state._generating_random:
        with st.status("正在生成智能创意...", expanded=True) as status:
            st.write("🔄 初始化创意引擎...")
            idea_texts = {0: "", 1: "", 2: ""}
            idea_topics_map = {}
            hot_topics_all = []
            hot_topics_shown = False
            previews = {0: st.empty(), 1: st.empty(), 2: st.empty()}
            candidates_result = None
            generation_error = None
            current_idx = 0

            try:
                gen = generate_random_idea_stream()
                while True:
                    try:
                        event = next(gen)
                    except StopIteration as e:
                        candidates_result = e.value
                        break
                    except Exception as ex:
                        generation_error = f"真实模型生成失败：{ex}"
                        break

                    if not isinstance(event, dict):
                        continue

                    etype = event.get("type", "")
                    if etype == "status":
                        msg = event.get("message", "")
                        if "检查网络" in msg:
                            st.write("🔍 " + msg)
                        elif "搜索" in msg:
                            st.write("🌐 " + msg)
                        elif "方案" in msg and "生成" in msg:
                            st.write("✨ " + msg)
                        elif "失败" in msg or "离线" in msg or "不可用" in msg:
                            st.write("⚠️ " + msg)
                        elif "排队" in msg or "内存不足" in msg:
                            st.write("⏳ " + msg)
                        elif "已生成" in msg or "推荐" in msg:
                            st.write("✅ " + msg)
                        elif "热点" in msg or "构思" in msg:
                            st.write("✨ " + msg)
                        else:
                            st.write("ℹ️ " + msg)
                        status.update(label=msg)
                    elif etype == "hot_topics":
                        hot_topics_all = event.get("topics", [])
                        if hot_topics_all and not hot_topics_shown:
                            tags_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin:0.3rem 0 0.5rem 0;'>"
                            for t in hot_topics_all:
                                tags_html += (
                                    f"<span style='background:#f0f0f5;color:#555;padding:3px 10px;"
                                    f"border-radius:12px;font-size:0.78rem;'>{t}</span>"
                                )
                            tags_html += "</div>"
                            st.markdown(
                                "<div style='font-size:0.8rem;color:#888;margin-top:0.3rem;'>📡 当前网络热门话题：</div>"
                                + tags_html,
                                unsafe_allow_html=True
                            )
                            hot_topics_shown = True
                    elif etype == "idea_start":
                        current_idx = event.get("index", 0)
                    elif etype == "idea_topics":
                        idx = event.get("index", 0)
                        topics = event.get("topics", [])
                        idea_topics_map[idx] = topics
                    elif etype == "text_delta":
                        idx = event.get("index", 0)
                        delta = event.get("delta", "")
                        idea_texts[idx] = idea_texts.get(idx, "") + delta
                        topics = idea_topics_map.get(idx, [])
                        tags_html = ""
                        if topics:
                            tags_html = "<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:0.3rem;'>"
                            for t in topics:
                                tags_html += (
                                    f"<span style='background:linear-gradient(135deg,#7757d7,#9b7ffc);"
                                    f"color:white;padding:2px 8px;border-radius:10px;font-size:0.72rem;'>🔥 {t}</span>"
                                )
                            tags_html += "</div>"
                        previews[idx].markdown(
                            f"<div style='padding:0.7rem;background:#f8f9fa;border-radius:8px;"
                            f"border-left:3px solid #7757d7;font-size:0.88rem;line-height:1.55;color:#1e1e2e;'>"
                            f"<div style='font-size:0.78rem;font-weight:600;color:#7757d7;margin-bottom:0.3rem;'>方案 {idx+1}</div>"
                            f"{tags_html}{idea_texts[idx]}</div>",
                            unsafe_allow_html=True
                        )
                    elif etype == "idea_end":
                        pass
            except Exception as ex:
                generation_error = f"真实模型生成失败：{ex}"

            if not generation_error and not candidates_result:
                candidates_result = [{"text": idea_texts.get(i, ""), "topics": idea_topics_map.get(i, [])} for i in range(3)]
                candidates_result = [c for c in candidates_result if c["text"].strip()]

            if generation_error or not candidates_result or len(candidates_result) != 3:
                generation_error = generation_error or "真实模型未完整返回3个创意方案，请重新生成。"
                status.update(label="创意生成失败", state="error", expanded=True)
            else:
                status.update(label="创意生成完成！", state="complete", expanded=True)

        st.session_state._generating_random = False
        st.session_state._idea_candidates = candidates_result if not generation_error else None
        st.session_state._idea_generation_error = generation_error
        st.rerun()

    res_preset = st.selectbox("画面分辨率", _resolution_options(),
                              index=_resolution_options().index(DEFAULT_IMAGE_RESOLUTION),
                              key="new_res")

    if st.button(":material/rocket_launch: 开始创作", type="primary", use_container_width=True, disabled=not source_text.strip()):
        project = create_project(_auto_title(source_text), source_text, resolution_preset=res_preset)
        st.session_state.selected_pid = project["id"]
        st.rerun()
else:
    project = current_project
    status = project["status"]
    analysis = project.get("analysis")
    shots = project.get("shots", [])
    total_dur = sum(s.get("duration_seconds", 0) for s in shots) if shots else (analysis.get("total_duration_seconds", 0) if analysis else 0)

    steps = ["创意分析", "分镜脚本", "资产生成", "视频合成"]
    status_map = {
        "new": 0, "analyzing_idea": 0, "awaiting_analysis_review": 0,
        "agent_planning": 0,
        "generating_storyboard": 1, "awaiting_storyboard_review": 1,
        "agent_storyboard_evaluation": 1,
        "generating_assets": 2, "agent_asset_evaluation": 2,
        "generating_video_prompts": 3, "awaiting_video_prompt_review": 3,
        "rendering_video": 3, "agent_error": 3, "completed": 3,
    }
    cur_step = status_map.get(status, 0)
    step_cols = st.columns(4)
    for i, (sc, name) in enumerate(zip(step_cols, steps, strict=False)):
        if i < cur_step:
            sc.markdown(f":material/check_circle: {name}")
        elif i == cur_step:
            sc.markdown(f":material/autorenew: **{name}**")
        else:
            sc.markdown(f":material/radio_button_unchecked: {name}")

    title_col, reset_col = st.columns([5, 1])
    with title_col:
        st.markdown(f"### :material/movie: {project['title']}")
        if shots:
            st.caption(f":material/movie: 共 {len(shots)} 个镜头 · 全片预计 **{total_dur} 秒**")
    with reset_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(":material/delete_sweep: 重置项目", use_container_width=True, help="清除所有中间结果，回到初始状态"):
            active_task = get_active_task()
            if active_task:
                _submit_error(TaskBusyError(active_task))
                st.stop()
            clear_agent_thread(project["id"])
            project["status"] = "new"
            project["analysis"] = None
            project["shots"] = []
            project["script"] = ""
            project["video"] = None
            save_project(project)
            st.rerun()

    with st.container(border=True):
        st.caption("后台生成状态（切换页面后仍会保留）")
        render_global_task_status(project_id=project["id"])

    agent_snapshot = _safe_agent_state(project["id"])
    agent_values = agent_snapshot.get("values", {})
    if agent_values:
        with st.expander(":material/account_tree: Agent 执行状态", expanded=agent_snapshot.get("waiting_for_human", False)):
            storyboard_eval = agent_values.get("evaluation_results", {}).get("storyboard", {})
            retries = agent_values.get("retry_counts", {})
            a1, a2, a3 = st.columns(3)
            a1.metric("当前阶段", agent_values.get("current_stage", "—"))
            a2.metric("分镜评分", storyboard_eval.get("score", "—"))
            a3.metric("自动重试", retries.get("storyboard", 0))
            st.caption("下一节点：" + ("、".join(agent_snapshot.get("next", [])) or "流程已结束"))
            if agent_snapshot.get("waiting_for_human"):
                review = (agent_snapshot.get("interrupts") or [{}])[0]
                st.info(f"等待人工审核：{review.get('review_type', 'unknown')}")
                if review.get("review_type") == "assets":
                    hr1, hr2, hr3 = st.columns(3)
                    with hr1:
                        if st.button("接受现有资产并继续", type="primary", key="_agent_assets_approve"):
                            run_video_agent(project["id"], {"action": "approve"}, "Agent 正在继续执行...")
                    with hr2:
                        if st.button("重新修复失败资产", key="_agent_assets_retry"):
                            run_video_agent(project["id"], {"action": "regenerate"}, "Agent 正在修复失败资产...")
                    with hr3:
                        if st.button("终止", key="_agent_assets_stop"):
                            run_video_agent(project["id"], {"action": "terminate"}, "正在终止 Agent...")
            if agent_values.get("feedback"):
                st.json(agent_values["feedback"])
            if agent_values.get("errors"):
                st.error(agent_values["errors"][-1].get("message", "Agent 执行失败"))

    src_text = project["source_text"]
    with st.expander(":material/description: 原始创意", expanded=True):
        st.write(src_text)

    analysis_confirmed = analysis and status not in ("new", "analyzing_idea", "awaiting_analysis_review")
    storyboard_confirmed = shots and status not in ("new", "analyzing_idea", "awaiting_analysis_review", "generating_storyboard", "awaiting_storyboard_review")
    assets_done = status in ("generating_video_prompts", "awaiting_video_prompt_review", "rendering_video", "completed")

    if analysis_confirmed and status != "awaiting_analysis_review":
        with st.expander(":material/check_circle: AI 智能理解（已确认）", expanded=False):
            if analysis:
                a1, a2 = st.columns(2)
                with a1:
                    st.markdown(f"**片名**：{analysis.get('title','')}")
                    st.markdown(f"**类型/风格**：{analysis.get('genre','')}")
                    st.markdown(f"**情绪基调**：{analysis.get('tone','')}")
                with a2:
                    st.markdown(f"**主题**：{analysis.get('theme','')}")
                    st.markdown(f"**目标时长**：{analysis.get('total_duration_seconds','')}秒")
                st.markdown(f"**故事梗概**：{analysis.get('narrative_summary','')}")
                if analysis.get("characters"):
                    st.markdown("**角色**：" + "、".join(c.get("name","") for c in analysis["characters"]))
                if analysis.get("props"):
                    st.markdown("**道具**：" + "、".join(p.get("name","") for p in analysis["props"]))
                if analysis.get("environments"):
                    st.markdown("**场景**：" + "、".join(e.get("name","") for e in analysis["environments"]))
                if st.button(":material/edit: 重新编辑分析", key="reedit_analysis"):
                    project["status"] = "awaiting_analysis_review"
                    save_project(project)
                    st.rerun()

    if storyboard_confirmed and status != "awaiting_storyboard_review":
        with st.expander(":material/check_circle: 分镜脚本（已确认）", expanded=False):
            if project.get("script"):
                st.caption("脚本全文：")
                st.write(project["script"])
            if shots:
                for shot in shots:
                    st.markdown(f"**镜头{shot['number']}：{shot.get('title','')}**（{shot.get('duration_seconds',3)}秒）")
                    st.caption(f"{shot.get('camera_angle','')} | {shot.get('camera_movement','')} | {shot.get('description','')[:60]}...")
                st.caption(f"共 {len(shots)} 个镜头，预计 {total_dur} 秒")
                if st.button(":material/edit: 重新编辑分镜", key="reedit_storyboard"):
                    project["status"] = "awaiting_storyboard_review"
                    save_project(project)
                    st.rerun()

    if assets_done:
        with st.expander(":material/check_circle: 资产与关键帧（已生成）", expanded=False):
            if analysis:
                for c in analysis.get("characters", []):
                    imgs = c.get("reference_images", [])
                    fp = _get_image_path(next((img for img in imgs if img.get("view")=="front"), imgs[0] if imgs else None))
                    if fp:
                        ci, ct = st.columns([1, 3])
                        with ci:
                            st.image(fp, width=100, caption=c["name"])
                        with ct:
                            st.markdown(f"**{c['name']}**：{c.get('description','')}")
                    else:
                        st.markdown(f"**{c['name']}**：{c.get('description','')}")
            kf_imgs = [s for s in shots if s.get("selected_image")]
            if kf_imgs:
                st.markdown("**关键帧**：")
                kfc = st.columns(min(4, len(kf_imgs)))
                for i, s in enumerate(kf_imgs):
                    with kfc[i % len(kfc)]:
                        fp = _get_image_path(s.get("selected_image"))
                        if fp:
                            st.image(fp, caption=f"镜头{s['number']}", use_container_width=True)

    if status in ("new", "analyzing_idea"):
        if st.button(":material/psychology: 开始分析创意", type="primary", use_container_width=True):
            run_video_agent(project["id"], title="Agent 正在分析、规划并生成分镜...")

    elif status == "awaiting_analysis_review" and analysis:
        st.subheader(":material/psychology: AI 智能理解")
        st.info("以下是 AI 对你的创意的理解，你可以修改后确认继续。")

        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("片名", value=analysis.get("title", ""))
            genre = st.text_input("类型/风格", value=analysis.get("genre", ""))
            tone = st.text_input("情绪基调", value=analysis.get("tone", ""))
        with col2:
            theme = st.text_input("主题", value=analysis.get("theme", ""))
            visual_style = st.text_input("视觉风格（英文 Prompt）", value=analysis.get("visual_style", ""))
            max_dur = st.number_input("目标时长（秒，10-60）", min_value=10, max_value=60,
                                       value=min(60, max(10, analysis.get("total_duration_seconds", 20))))

        narrative_summary = st.text_area("故事梗概", value=analysis.get("narrative_summary", ""), height=80)

        st.markdown("---")
        st.markdown("#### :material/groups: 角色")
        chars = analysis.get("characters", [])
        char_names = {}
        for i, c in enumerate(chars):
            with st.container(border=True):
                cc1, cc2 = st.columns([1, 2])
                with cc1:
                    cname = st.text_input(f"角色{i+1}名称", value=c.get("name", ""), key=f"char_name_{i}")
                    ctype = st.selectbox("类型", ["人物", "动物"], index=0 if c.get("type") == "人物" else 1, key=f"char_type_{i}")
                with cc2:
                    cdesc = st.text_input("描述", value=c.get("description", ""), key=f"char_desc_{i}")
                    capp = st.text_area("外观描述（英文）", value=c.get("appearance", c.get("prompt", "")), height=60, key=f"char_app_{i}")
                c["name"] = cname
                c["type"] = ctype
                c["description"] = cdesc
                c["appearance"] = capp
                char_names[c["id"]] = cname

        st.markdown("#### :material/inventory_2: 道具")
        props = analysis.get("props", [])
        for i, p in enumerate(props):
            with st.container(border=True):
                pc1, pc2 = st.columns([1, 2])
                with pc1:
                    pname = st.text_input(f"道具{i+1}名称", value=p.get("name", ""), key=f"prop_name_{i}")
                with pc2:
                    pdesc = st.text_input("描述", value=p.get("description", ""), key=f"prop_desc_{i}")
                    pvis = st.text_input("视觉描述（英文）", value=p.get("visual_prompt", p.get("prompt", "")), key=f"prop_vis_{i}")
                p["name"] = pname
                p["description"] = pdesc
                p["visual_prompt"] = pvis
                p.setdefault("prompt", pvis)
                p.setdefault("appearance", pvis)

        add_prop = st.button(":material/add: 添加道具")
        if add_prop:
            new_id = f"prop-{len(props)+1:02d}"
            props.append({"id": new_id, "name": "新道具", "description": "", "visual_prompt": "detailed object",
                          "prompt": "detailed object", "appearance": "detailed object", "reference_images": []})
            analysis.update({
                "title": title, "genre": genre, "theme": theme, "tone": tone,
                "visual_style": visual_style, "total_duration_seconds": int(max_dur),
                "narrative_summary": narrative_summary,
                "characters": chars, "props": props,
            })
            project["analysis"] = analysis
            project["title"] = title
            save_project(project)
            st.rerun()

        st.markdown("#### :material/landscape: 场景")
        envs = analysis.get("environments", [])
        for i, e in enumerate(envs):
            with st.container(border=True):
                ec1, ec2 = st.columns([1, 2])
                with ec1:
                    ename = st.text_input(f"场景{i+1}名称", value=e.get("name", ""), key=f"env_name_{i}")
                with ec2:
                    edesc = st.text_input("描述", value=e.get("description", ""), key=f"env_desc_{i}")
                    evis = st.text_input("视觉描述（英文）", value=e.get("visual_prompt", e.get("prompt", "")), key=f"env_vis_{i}")
                e["name"] = ename
                e["description"] = edesc
                e["visual_prompt"] = evis
                e.setdefault("prompt", evis)
                e.setdefault("appearance", evis)

        add_env = st.button(":material/add: 添加场景")
        if add_env:
            new_id = f"env-{len(envs)+1:02d}"
            envs.append({"id": new_id, "name": "新场景", "description": "", "visual_prompt": "cinematic environment",
                         "prompt": "cinematic environment", "appearance": "cinematic environment", "reference_images": []})
            analysis.update({
                "title": title, "genre": genre, "theme": theme, "tone": tone,
                "visual_style": visual_style, "total_duration_seconds": int(max_dur),
                "narrative_summary": narrative_summary,
                "characters": chars, "props": props, "environments": envs,
            })
            project["analysis"] = analysis
            project["title"] = title
            save_project(project)
            st.rerun()

        st.divider()
        if st.button(":material/check_circle: 确认分析，生成分镜表", type="primary", use_container_width=True):
            active_task = get_active_task()
            if active_task:
                _submit_error(TaskBusyError(active_task))
            else:
                updated = {
                    "title": title, "genre": genre, "theme": theme, "tone": tone,
                    "visual_style": visual_style, "total_duration_seconds": int(max_dur),
                    "narrative_summary": narrative_summary,
                    "characters": chars, "props": props, "environments": envs,
                }
                approve_analysis(project["id"], updated)
                run_pipeline_stream(
                    lambda: generate_storyboard_from_analysis_stream(project["id"]),
                    "正在生成分镜表...",
                )

    elif status in ("generating_storyboard",):
        st.subheader(":material/assignment: 正在生成分镜表")
        st.caption("AI 正在根据分析结果生成分镜脚本，请稍候...")
        if shots:
            st.info(f"已有 {len(shots)} 个镜头，可以重新生成或继续。")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(":material/refresh: 重新生成分镜表", type="primary", use_container_width=True):
                run_pipeline_stream(lambda: generate_storyboard_from_analysis_stream(project["id"]), "正在重新生成分镜表...")
        with col2:
            if shots and st.button(":material/arrow_forward: 跳过，直接查看现有分镜", use_container_width=True):
                project["status"] = "awaiting_storyboard_review"
                save_project(project)
                st.rerun()

    elif status == "awaiting_storyboard_review":
        st.subheader(":material/assignment: 分镜脚本表")
        if project.get("script"):
            with st.expander("分镜脚本全文", expanded=False):
                st.write(project["script"])

        for shot in shots:
            with st.container(border=True):
                sc1, sc2 = st.columns([3, 2])
                with sc1:
                    st.markdown(f"**镜头 {shot['number']}：{shot.get('title', '')}** · {shot.get('duration_seconds', 3)}秒")
                    st.caption(f":material/photo_camera: {shot.get('camera_angle', '')} | :material/videocam: {shot.get('camera_movement', '')}")
                    st.write(shot.get("description", ""))
                    chars_str = _asset_names_from_ids(analysis, shot.get("character_ids", []))
                    props_str = _asset_names_from_ids(analysis, shot.get("prop_ids", []))
                    env_str = _env_name(analysis, shot.get("environment_id", ""))
                    st.caption(f":material/person: {chars_str or '—'} | :material/inventory_2: {props_str or '—'} | :material/landscape: {env_str}")
                with sc2:
                    shot["title"] = st.text_input("镜头标题", value=shot.get("title", ""), key=f"shot_title_{shot['id']}")
                    shot["duration_seconds"] = st.number_input("时长(秒)", min_value=2, max_value=8,
                                                               value=shot.get("duration_seconds", 3), key=f"shot_dur_{shot['id']}")
                    shot["description"] = st.text_area("描述", value=shot.get("description", ""), height=60, key=f"shot_desc_{shot['id']}")
                    shot["prompt"] = st.text_area("画面提示词(英文)", value=shot.get("prompt", ""), height=60, key=f"shot_prompt_{shot['id']}")

        st.divider()
        review_interrupt = (agent_snapshot.get("interrupts") or [{}])[0]
        is_agent_storyboard_review = review_interrupt.get("review_type") == "storyboard"
        review_cols = st.columns(3 if is_agent_storyboard_review else 1)
        with review_cols[0]:
            if st.button(":material/check_circle: 确认分镜，开始生成资产", type="primary", use_container_width=True):
                active_task = get_active_task()
                if active_task:
                    _submit_error(TaskBusyError(active_task))
                else:
                    if is_agent_storyboard_review:
                        run_video_agent(
                            project["id"],
                            {"action": "edit_and_continue", "edited_storyboard": shots},
                            "Agent 正在生成并评测资产...",
                        )
                    else:
                        approve_storyboard(project["id"], shots)
                        run_pipeline_stream(
                            lambda: generate_assets_stream(project["id"]),
                            "正在生成资产...",
                        )
        if is_agent_storyboard_review:
            with review_cols[1]:
                if st.button(":material/refresh: 按反馈重新生成", use_container_width=True):
                    run_video_agent(
                        project["id"], {"action": "regenerate"},
                        "Agent 正在反思并定向修复分镜...",
                    )
            with review_cols[2]:
                if st.button(":material/stop_circle: 终止 Agent", use_container_width=True):
                    run_video_agent(project["id"], {"action": "terminate"}, "正在终止 Agent...")

    elif status in ("generating_assets",):
        st.subheader(":material/palette: 正在生成资产")
        st.caption("正在按顺序生成：角色参考图 → 道具图 → 场景图 → 关键帧")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(":material/refresh: 重新生成全部资产", type="primary", use_container_width=True):
                run_pipeline_stream(lambda: generate_assets_stream(project["id"]), "正在重新生成资产...")
        with col2:
            if shots and any(s.get("selected_image") for s in shots):
                if st.button(":material/arrow_forward: 跳过，继续到视频提示词", use_container_width=True):
                    project["status"] = "generating_video_prompts"
                    save_project(project)
                    st.rerun()

    elif status in ("generating_video_prompts", "awaiting_video_prompt_review"):
        st.subheader(":material/movie: 资产与关键帧")
        if analysis:
            with st.expander(":material/person: 角色参考图", expanded=True):
                for c in analysis.get("characters", []):
                    cc1, cc2 = st.columns([1, 2])
                    with cc1:
                        imgs = c.get("reference_images", [])
                        front_img = next((img for img in imgs if img.get("view") == "front"), imgs[0] if imgs else None)
                        fp = _get_image_path(front_img)
                        if fp:
                            st.image(fp, width=150, caption=c["name"])
                        else:
                            st.markdown(f":material/person: {c['name']}")
                    with cc2:
                        st.markdown(f"**{_char_type_label(c.get('type','人物'))} {c['name']}**")
                        st.caption(c.get("description", ""))
                        if st.button(":material/refresh: 重新生成", key=f"regen_char_{c['id']}"):
                            run_pipeline_stream(lambda: regenerate_character_images_stream(project["id"], c["id"]), f"重新生成 {c['name']}")

            with st.expander(":material/inventory_2: 道具参考图", expanded=bool(analysis.get("props"))):
                for p in analysis.get("props", []):
                    pc1, pc2 = st.columns([1, 2])
                    with pc1:
                        imgs = p.get("reference_images", [])
                        if imgs:
                            fp = _get_image_path(imgs[-1])
                            if fp:
                                st.image(fp, width=120, caption=p["name"])
                    with pc2:
                        st.markdown(f"**:material/inventory_2: {p['name']}**")
                        st.caption(p.get("description", ""))

            with st.expander(":material/landscape: 场景参考图", expanded=bool(analysis.get("environments"))):
                for e in analysis.get("environments", []):
                    ec1, ec2 = st.columns([1, 2])
                    with ec1:
                        imgs = e.get("reference_images", [])
                        if imgs:
                            fp = _get_image_path(imgs[-1])
                            if fp:
                                st.image(fp, width=200, caption=e["name"])
                    with ec2:
                        st.markdown(f"**:material/landscape: {e['name']}**")
                        st.caption(e.get("description", ""))

        st.markdown("---")
        st.subheader(":material/theaters: 关键帧预览")
        kf_cols = st.columns(min(3, len(shots)) if shots else 1)
        for i, shot in enumerate(shots):
            with kf_cols[i % len(kf_cols)]:
                sel = shot.get("selected_image")
                fp = _get_image_path(sel)
                if fp:
                    st.image(fp, caption=f"镜头{shot['number']}·{shot['title']}({shot.get('duration_seconds',3)}s)", use_container_width=True)
                else:
                    st.markdown(f":material/movie: 镜头{shot['number']}: {shot.get('title','')}")
                if shot.get("video_prompt"):
                    vp = shot["video_prompt"]
                    st.caption(f":material/videocam: {vp.get('camera_movement', '')}")

        vp_count = sum(1 for s in shots if s.get("video_prompt"))
        btn_label = ":material/movie: 一键生成视频" if vp_count == 0 else ":material/movie: 一键生成视频（已生成提示词）"
        if st.button(btn_label, type="primary", use_container_width=True):
            review_interrupt = (agent_snapshot.get("interrupts") or [{}])[0]
            if review_interrupt.get("review_type") == "render":
                run_video_agent(
                    project["id"], {"action": "approve"},
                    "Agent 正在渲染并执行最终评测...",
                )
            else:
                run_pipeline_stream(lambda: render_video_stream(project["id"]), "正在生成视频...")

    elif status in ("rendering_video", "completed"):
        st.subheader(":material/celebration: 视频已生成")
        video = project.get("video")
        if video:
            st.json(video)
            vpath = video.get("path")
            if vpath and Path(vpath).exists():
                if vpath.endswith(".mp4"):
                    st.video(vpath)
                else:
                    st.info(f"视频清单已写入：{vpath}\n（当前为模拟模式，接入线上 I2V API 后可生成真实视频）")
                    with open(vpath) as f:
                        st.code(f.read(), language="json")

        st.markdown("---")
        if analysis:
            st.markdown("### :material/bar_chart: 完整分镜表")
            table_data = []
            for shot in shots:
                table_data.append({
                    "#": shot["number"],
                    "镜头": shot.get("title", ""),
                    "时长(s)": shot.get("duration_seconds", 0),
                    "景别": shot.get("camera_angle", ""),
                    "运动": shot.get("camera_movement", ""),
                    "角色": _asset_names_from_ids(analysis, shot.get("character_ids", [])),
                    "道具": _asset_names_from_ids(analysis, shot.get("prop_ids", [])),
                    "场景": _env_name(analysis, shot.get("environment_id", "")),
                })
            import pandas as pd
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button(":material/refresh: 基于此创意重新生成", use_container_width=True):
            project["status"] = "awaiting_analysis_review"
            save_project(project)
            st.rerun()

    if status in ("generating_assets", "generating_video_prompts", "rendering_video"):
        active_task = get_active_task()
        if active_task is None:
            st.warning(
                ":material/warning: 项目记录显示仍在生成，但当前没有后台任务运行。"
                "这通常表示上一次任务因页面旧版本或应用重启而中断；"
                "请使用上方对应的“重新生成”按钮恢复。"
            )
    elif status == "agent_error":
        if st.button(":material/restart_alt: 清除 Agent 状态并重新开始", type="primary"):
            clear_agent_thread(project["id"])
            project["status"] = "new"
            project.pop("agent_error", None)
            save_project(project)
            st.rerun()
