from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st

from app.evaluation import export as eval_export
from app.evaluation import repository as eval_repo
from app.evaluation.rubric import Rubric, load_default_rubric
from app.evaluation.scoring import (
    calculate_weighted_total,
    filter_items,
    summarize_items,
    validate_scores,
)
from app.repository import get_project, list_projects

ASSET_TYPE_LABELS = {
    "keyframe": "关键帧",
    "character": "角色图",
    "prop": "道具图",
    "environment": "场景图",
    "video": "视频",
}

_NAV_OPTIONS = [
    ("runs", ":material/folder_open: 评测任务"),
    ("manual", ":material/edit_note: 人工评测"),
    ("auto", ":material/smart_toy: 自动机评"),
    ("dashboard", ":material/dashboard: 评测看板"),
    ("arena", ":material/sports_martial_arts: A/B竞技场"),
]
_NAV_KEYS = [o[0] for o in _NAV_OPTIONS]
_NAV_LABELS = [o[1] for o in _NAV_OPTIONS]
_NAV_RADIO_KEY = "_eval_nav_radio"


def _set_eval_view(view: str) -> None:
    st.session_state["_eval_view"] = view
    if view in _NAV_KEYS:
        st.session_state["_eval_nav_pending"] = _NAV_LABELS[_NAV_KEYS.index(view)]


def render_evaluation_center() -> None:
    if "_eval_view" not in st.session_state:
        st.session_state["_eval_view"] = "runs"
    pending_nav = st.session_state.pop("_eval_nav_pending", None)
    if pending_nav is not None:
        st.session_state[_NAV_RADIO_KEY] = pending_nav

    if _NAV_RADIO_KEY not in st.session_state:
        st.session_state[_NAV_RADIO_KEY] = _NAV_LABELS[0]
        st.session_state["_eval_view"] = "runs"

    current_view = st.session_state["_eval_view"]

    title_col, back_col, create_col = st.columns(
        [4.6, 1.15, 1.15],
        vertical_alignment="top",
    )
    with title_col:
        st.title(":material/rate_review: 多模态评测中心")
    with back_col:
        st.markdown("<div style='padding-top: 0.3rem;'>", unsafe_allow_html=True)
        if st.button(":material/arrow_back: 返回工作台", use_container_width=True, type="secondary"):
            st.session_state["_app_view"] = "workflow"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with create_col:
        st.markdown("<div style='padding-top: 0.3rem;'>", unsafe_allow_html=True)
        create_btn_type = "primary" if current_view == "create" else "secondary"
        if st.button(
            ":material/add: 创建任务",
            key="eval_nav_create",
            use_container_width=True,
            type=create_btn_type,
        ):
            _set_eval_view("create")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    nav_cols = st.columns(len(_NAV_OPTIONS))
    for i, (view_key, label) in enumerate(_NAV_OPTIONS):
        with nav_cols[i]:
            btn_type = "primary" if current_view == view_key else "secondary"
            if st.button(label, key=f"eval_nav_{view_key}", use_container_width=True, type=btn_type):
                st.session_state[_NAV_RADIO_KEY] = label
                st.session_state["_eval_view"] = view_key
                st.rerun()

    st.markdown("---")

    view = st.session_state["_eval_view"]
    if view == "runs":
        _render_runs_list()
    elif view == "create":
        _render_create_run()
    elif view == "manual":
        _render_manual_eval()
    elif view == "auto":
        from app.evaluation.ui_auto_judge import render_auto_judge
        render_auto_judge()
    elif view == "dashboard":
        _render_dashboard()
    elif view == "arena":
        from app.evaluation.ui_arena import render_arena
        render_arena()


def _render_runs_list() -> None:
    st.subheader(":material/folder_open: 评测任务列表")
    st.markdown(
        """
        <style>
        div[class*="st-key-eval_run_actions_"] button {
            white-space: nowrap !important;
        }
        div[class*="st-key-eval_run_actions_"] button p {
            white-space: nowrap !important;
            word-break: keep-all !important;
            overflow-wrap: normal !important;
            hyphens: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    runs = eval_repo.list_runs()

    if not runs:
        st.info(":material/info: 暂无评测任务，点击顶部「创建任务」开始。")
        return

    for run in runs:
        progress = eval_repo.get_run_progress(run["id"])
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(
                [2.6, 0.95, 0.95, 5.8],
                vertical_alignment="center",
            )
            with c1:
                st.markdown(f"**{run['name']}**")
                st.caption(
                    f"类型: {run.get('task_type', 'universal')} · "
                    f"评判: {run.get('judge_type', 'human')} · "
                    f"状态: {run.get('status', 'pending')}"
                )
            with c2:
                st.metric("已提交", f"{progress['submitted']}/{progress['total']}")
            with c3:
                auto_count = eval_repo._auto_judged_count(run["id"]) if hasattr(eval_repo, '_auto_judged_count') else 0
                st.metric("已机评", str(auto_count))

            with c4:
                with st.container(
                    horizontal=True,
                    horizontal_alignment="right",
                    vertical_alignment="center",
                    gap="small",
                    key=f"eval_run_actions_{run['id']}",
                ):
                    if st.button(
                        "人工评测",
                        key=f"start_{run['id']}",
                        width=104,
                    ):
                        st.session_state["_eval_run_id"] = run["id"]
                        st.session_state["_eval_item_idx"] = 0
                        _set_eval_view("manual")
                        st.rerun()
                    if st.button(
                        "自动机评",
                        key=f"auto_{run['id']}",
                        width=104,
                    ):
                        st.session_state["_eval_run_id"] = run["id"]
                        _set_eval_view("auto")
                        st.rerun()
                    if st.button(
                        "评测看板",
                        key=f"dash_{run['id']}",
                        width=104,
                    ):
                        st.session_state["_eval_run_id"] = run["id"]
                        _set_eval_view("dashboard")
                        st.rerun()
                    if st.button(
                        ":material/delete:",
                        key=f"del_{run['id']}",
                        help="删除评测任务",
                        width=44,
                    ):
                        eval_repo.delete_run(run["id"])
                        st.rerun()
            st.progress(min(1.0, progress["progress_pct"] / 100.0))


def _render_create_run() -> None:
    st.subheader(":material/add: 创建评测任务")

    projects = list_projects()
    if not projects:
        st.warning(":material/warning: 暂无项目，请先创建项目并生成资产。")
        if st.button(":material/arrow_back: 返回任务列表"):
            _set_eval_view("runs")
            st.rerun()
        return

    with st.form("create_eval_run"):
        run_name = st.text_input("任务名称", value=f"评测任务 - {len(eval_repo.list_runs()) + 1}")
        project_options = {}
        seen_titles: dict[str, int] = {}
        for p in projects:
            seen_titles[p["title"]] = seen_titles.get(p["title"], 0) + 1
        for p in projects:
            if seen_titles.get(p["title"], 0) > 1:
                label = f'{p["title"]} ({p["id"][:6]})'
            else:
                label = p["title"]
            project_options[label] = p["id"]
        selected_label = st.selectbox("选择项目", list(project_options.keys()))
        selected_pid = project_options[selected_label]

        asset_types = st.multiselect(
            "评测资产类型",
            ["keyframe", "character", "prop", "environment", "video"],
            default=["keyframe"],
            format_func=lambda x: ASSET_TYPE_LABELS.get(x, x),
        )
        judge_type = st.selectbox(
            "评判方式",
            ["human"],
            format_func=lambda x: {"human": "人工评分"}.get(x, x),
        )
        rubrics = eval_repo.list_rubrics()
        rubric_options = {
            f"{entry['name']} · {entry['version']}": entry["id"]
            for entry in rubrics
        }
        selected_rubric_label = st.selectbox(
            "评分标准版本",
            list(rubric_options),
            help="任务创建后固定使用该版本，确保历史评分可复现。",
        )

        submitted = st.form_submit_button(":material/add: 创建任务并选择素材", type="primary")

    if submitted:
        project = get_project(selected_pid)
        if not project:
            st.error("项目不存在")
            return
        all_assets = eval_repo.discover_project_assets(project)
        filtered = [a for a in all_assets if a["asset_type"] in asset_types]
        if not filtered:
            st.warning(f":material/warning: 项目「{project['title']}」中未找到{'、'.join(ASSET_TYPE_LABELS[t] for t in asset_types)}类型的资产。")
            return

        st.session_state["_eval_pending_assets"] = filtered
        st.session_state["_eval_pending_run_name"] = run_name
        st.session_state["_eval_pending_pid"] = selected_pid
        st.session_state["_eval_pending_judge"] = judge_type
        st.session_state["_eval_pending_rubric_id"] = rubric_options[selected_rubric_label]
        _set_eval_view("select_assets")
        st.rerun()

    if st.button(":material/arrow_back: 返回"):
        _set_eval_view("runs")
        st.rerun()

def _render_asset_selector() -> None:
    st.subheader(":material/checklist: 选择评测素材")
    assets = st.session_state.get("_eval_pending_assets", [])
    if not assets:
        st.warning("未找到待选素材")
        if st.button(":material/arrow_back: 返回"):
            _set_eval_view("create")
            st.rerun()
        return

    st.caption(f"共找到 {len(assets)} 个可评测素材，勾选需要加入评测任务的素材：")

    search = st.text_input(":material/search: 搜索（标题/Prompt/模型）", key="_eval_asset_search")
    filtered = assets
    if search:
        s = search.lower()
        filtered = [a for a in assets if s in a.get("title", "").lower() or s in a.get("prompt", "").lower() or s in str(a.get("model_provider", "")).lower()]

    if "select_all_assets" not in st.session_state:
        st.session_state["select_all_assets"] = True

    c1, c2, c3, _ = st.columns([1, 1, 1, 4])
    with c1:
        if st.button(":material/select_all: 全选"):
            st.session_state["select_all_assets"] = True
            for a in filtered:
                st.session_state[f"sel_{a['asset_path']}"] = True
            st.rerun()
    with c2:
        if st.button(":material/deselect: 全不选"):
            st.session_state["select_all_assets"] = False
            for a in filtered:
                st.session_state[f"sel_{a['asset_path']}"] = False
            st.rerun()

    selected = []
    type_counts: dict[str, int] = {}
    for i, a in enumerate(filtered):
        key = f"sel_{a['asset_path']}"
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get("select_all_assets", True)
        is_selected = st.checkbox(
            f"**[{ASSET_TYPE_LABELS.get(a['asset_type'], a['asset_type'])}]** {a.get('title', '')} · "
            f"{a.get('model_provider', '?')} · {Path(a['asset_path']).name}",
            key=key,
        )
        if is_selected:
            selected.append(a)
            type_counts[a["asset_type"]] = type_counts.get(a["asset_type"], 0) + 1

    st.markdown("---")
    st.caption(f"已选 **{len(selected)}** 个素材：" + " · ".join(
        f"{ASSET_TYPE_LABELS.get(k, k)} {v}个" for k, v in type_counts.items()
    ))

    c1, c2 = st.columns(2)
    with c1:
        if st.button(":material/arrow_back: 返回", use_container_width=True):
            _set_eval_view("create")
            st.rerun()
    with c2:
        if st.button(":material/check: 确认创建并开始评测", type="primary", use_container_width=True, disabled=len(selected) == 0):
            run = eval_repo.create_evaluation_run(
                name=st.session_state["_eval_pending_run_name"],
                task_type="mixed",
                judge_type=st.session_state["_eval_pending_judge"],
                project_id=st.session_state["_eval_pending_pid"],
                rubric_version_id=st.session_state["_eval_pending_rubric_id"],
            )
            for idx, a in enumerate(selected):
                eval_repo.add_evaluation_item(
                    run_id=run["id"],
                    asset_path=a["asset_path"],
                    asset_type=a["asset_type"],
                    project_id=a.get("project_id"),
                    shot_id=a.get("shot_id"),
                    asset_id=a.get("asset_id"),
                    prompt=a.get("prompt"),
                    model_provider=a.get("model_provider"),
                    model_version=a.get("model_version"),
                    order_index=idx,
                )
            st.session_state["_eval_run_id"] = run["id"]
            st.session_state["_eval_item_idx"] = 0
            _set_eval_view("manual")
            for key in list(st.session_state.keys()):
                if key.startswith("sel_") or key.startswith("_eval_pending_") or key == "select_all_assets":
                    del st.session_state[key]
            st.rerun()


def _render_manual_eval() -> None:
    run_id = st.session_state.get("_eval_run_id")
    if not run_id:
        st.info(":material/info: 请先从任务列表选择一个评测任务。")
        if st.button(":material/arrow_back: 返回任务列表"):
            _set_eval_view("runs")
            st.rerun()
        return

    run = eval_repo.get_run(run_id)
    if not run:
        st.error("评测任务不存在")
        _set_eval_view("runs")
        st.rerun()
        return

    items = eval_repo.list_run_items(run_id)
    if not items:
        st.warning("该任务没有待评素材")
        if st.button(":material/arrow_back: 返回"):
            _set_eval_view("runs")
            st.rerun()
        return

    rubric: Rubric = run.get("rubric") or load_default_rubric()
    bad_case_tags = run.get("bad_case_tags", rubric.bad_case_tags)

    idx = st.session_state.get("_eval_item_idx", 0)
    idx = max(0, min(idx, len(items) - 1))
    st.session_state["_eval_item_idx"] = idx
    item = items[idx]

    progress = eval_repo.get_run_progress(run_id)

    st.subheader(f":material/edit_note: {run['name']}")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("总素材", progress["total"])
    pc2.metric("已提交", progress["submitted"])
    pc3.metric("进行中", progress["in_progress"])
    pc4.metric("完成率", f"{progress['progress_pct']:.0f}%")
    st.progress(min(1.0, progress["progress_pct"] / 100.0))

    st.markdown(f"**素材 {idx + 1} / {len(items)}**  ·  "
                f"状态: **{item.get('status', 'pending')}**"
                + (f"  ·  当前总分: **{item.get('total_score', '-'):.2f}**" if item.get("total_score") is not None else ""))

    nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([1, 1, 1, 1, 3])
    with nav_c1:
        if st.button(":material/arrow_back: 上一条", use_container_width=True, disabled=(idx == 0)):
            _save_current_draft(item, rubric, submit=False)
            st.session_state["_eval_item_idx"] = idx - 1
            st.rerun()
    with nav_c2:
        if st.button(":material/arrow_forward: 下一条", use_container_width=True, disabled=(idx == len(items) - 1)):
            _save_current_draft(item, rubric, submit=False)
            st.session_state["_eval_item_idx"] = idx + 1
            st.rerun()
    with nav_c3:
        if st.button(":material/save: 暂存", use_container_width=True):
            _save_current_draft(item, rubric, submit=False)
            st.success(":material/check: 已暂存")
    with nav_c4:
        if st.button(":material/send: 提交", type="primary", use_container_width=True):
            ok, errs = _save_current_draft(item, rubric, submit=True)
            if ok:
                st.success(":material/check: 已提交")
                if idx < len(items) - 1:
                    st.session_state["_eval_item_idx"] = idx + 1
                st.rerun()
            else:
                for e in errs:
                    st.error(e)
    with nav_c5:
        if st.button(":material/folder_open: 返回任务列表", use_container_width=True):
            _set_eval_view("runs")
            st.rerun()

    st.markdown("---")

    preview, right = st.columns([1.35, 1], gap="large")

    existing_scores = item.get("scores") or {}
    existing_tags = set()
    for t in item.get("bad_case_tags") or []:
        if isinstance(t, dict):
            existing_tags.add(t.get("tag", ""))
        else:
            existing_tags.add(str(t))
    existing_note = item.get("note", "") or ""

    with preview:
        st.markdown("#### :material/info: 素材信息")
        asset_type_label = ASSET_TYPE_LABELS.get(item.get("asset_type", ""), item.get("asset_type", ""))
        st.markdown(f"**类型**: {asset_type_label}")
        if item.get("shot_id"):
            st.markdown(f"**镜头ID**: `{item.get('shot_id')[:8]}...`")
        if item.get("model_provider"):
            st.markdown(f"**生成模型**: {item.get('model_provider')} {item.get('model_version', '')}")
        path = item.get("asset_path", "")
        if path:
            st.markdown(f"**文件**: `{Path(path).name}`")

        st.markdown("#### :material/description: Prompt")
        prompt_text = item.get("prompt", "") or "（无Prompt）"
        st.text_area("Prompt内容", value=prompt_text, height=120, disabled=True, label_visibility="collapsed",
                     key=f"_prompt_view_{item['id']}")

        if item.get("reference_images"):
            st.markdown("#### :material/image: 参考图")
            for ref in item["reference_images"][:4]:
                if Path(ref).exists():
                    st.image(ref, use_container_width=True)

    with preview:
        st.markdown("#### :material/image: 待评测内容")
        file_path = Path(item.get("asset_path", ""))
        if not file_path.exists():
            st.error(f":material/error: 文件不存在: {file_path}")
        elif item.get("asset_type") == "video":
            st.video(str(file_path))
        else:
            st.image(str(file_path), use_container_width=True)

    with right:
        st.markdown("#### :material/grade: Rubric 评分")
        st.caption("1 = 严重失败 · 3 = 基本可用 · 5 = 可直接使用")

        scores: dict[str, int] = {}
        for dim in rubric.dimensions:
            existing = existing_scores.get(dim.key, {})
            default_score = None
            if isinstance(existing, dict):
                default_score = existing.get("score")
            else:
                default_score = existing

            score_key = f"_score_{item['id']}_{dim.key}"
            radio_index = None
            if default_score is not None and score_key not in st.session_state:
                try:
                    default_int = int(default_score)
                    if 1 <= default_int <= 5:
                        radio_index = default_int - 1
                except (TypeError, ValueError):
                    pass

            st.markdown(f"**{dim.label}**" + (" `*`" if dim.required else "") + f"  (权重 {dim.weight})")
            score_val = st.radio(
                dim.label,
                options=[1, 2, 3, 4, 5],
                format_func=lambda s: f"{s}分",
                key=score_key,
                index=radio_index,
                horizontal=True,
                label_visibility="collapsed",
            )
            if score_val is not None:
                scores[dim.key] = score_val
                anchor = dim.anchors.get(score_val, "")
                if anchor:
                    st.caption(f":material/anchor: {anchor}")

        st.markdown("#### :material/report: Bad Case 标签")
        available_tags = list(dict.fromkeys([*bad_case_tags, *sorted(existing_tags)]))
        custom_tag = st.text_input(":material/add: 自定义标签", key=f"_custom_tag_{item['id']}", placeholder="输入自定义标签后回车添加")
        tag_columns = st.columns(2)
        for index, tag in enumerate(available_tags):
            tag_key = f"_tag_{item['id']}_{tag}"
            if tag_key not in st.session_state:
                st.session_state[tag_key] = tag in existing_tags
            with tag_columns[index % 2]:
                st.checkbox(tag, key=tag_key)
        if custom_tag and custom_tag.strip():
            st.caption(f"将保存自定义标签：`{custom_tag.strip()}`")

        st.markdown("#### :material/edit: 问题备注")
        note_key = f"_note_{item['id']}"
        if note_key not in st.session_state:
            st.session_state[note_key] = existing_note
        st.text_area("备注", value=existing_note, height=90, label_visibility="collapsed", key=note_key)

        if scores:
            total = calculate_weighted_total(scores, rubric)
            ok, _ = validate_scores(scores, rubric)
            st.markdown(f"#### 当前加权总分: :{('green' if total >= 4 else 'orange' if total >= 3 else 'red')}[{total:.2f} / 5.00]")
            if not ok:
                st.caption(":material/warning: 请先完成所有必选维度评分")


def _collect_bad_case_tags(
    item_id: str,
    state: Mapping[str, Any],
) -> list[dict[str, str]]:
    prefix = f"_tag_{item_id}_"
    names = {
        key[len(prefix):].strip()
        for key, selected in state.items()
        if key.startswith(prefix) and selected and key[len(prefix):].strip()
    }
    custom = str(state.get(f"_custom_tag_{item_id}", "") or "").strip()
    if custom:
        names.add(custom)
    return [{"tag": tag, "note": ""} for tag in sorted(names)]


def _save_current_draft(item: dict[str, Any], rubric: Rubric, submit: bool = False) -> tuple[bool, list[str]]:
    item_id = item["id"]
    scores: dict[str, int] = {}
    for dim in rubric.dimensions:
        key = f"_score_{item_id}_{dim.key}"
        val = st.session_state.get(key)
        if val is not None:
            try:
                scores[dim.key] = int(val)
            except (TypeError, ValueError):
                pass

    ok, errors = validate_scores(scores, rubric)
    if submit and not ok:
        return False, errors

    tags = _collect_bad_case_tags(item_id, st.session_state)

    note_key = f"_note_{item_id}"
    note = st.session_state.get(note_key, "")

    total = calculate_weighted_total(scores, rubric) if ok else None
    eval_repo.save_item_scores(
        item_id=item_id,
        scores=scores,
        bad_case_tags=tags,
        note=note,
        total_score=total,
        submit=submit,
    )
    return True, []


def _render_dashboard() -> None:
    st.subheader(":material/dashboard: 评测看板")

    runs = eval_repo.list_runs()
    if not runs:
        st.info(":material/info: 暂无评测数据。请先创建评测任务并完成评分。")
        return

    run_options = {r["name"] + f" ({r['id'][:8]})": r["id"] for r in runs}
    run_options["（按 Rubric 汇总）"] = "__all__"
    selected = st.selectbox("选择评测任务", list(run_options.keys()), index=len(run_options) - 1)
    run_id = run_options[selected]

    if run_id == "__all__":
        rubric_groups: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            rubric_groups.setdefault(run["rubric_version_id"], []).append(run)
        rubric_labels = {}
        for rubric_id, grouped_runs in rubric_groups.items():
            sample = eval_repo.get_run(grouped_runs[0]["id"])
            rubric_data = sample.get("rubric") if sample else None
            label = (
                f"{rubric_data.name} · {rubric_data.version} · {len(grouped_runs)} 个任务"
                if rubric_data
                else f"未知 Rubric ({rubric_id[:8]})"
            )
            rubric_labels[label] = rubric_id
        selected_rubric_label = st.selectbox(
            "汇总评分标准",
            list(rubric_labels),
            help="不同 Rubric 的维度和权重不可直接合并，因此汇总限定在同一版本。",
        )
        selected_rubric_id = rubric_labels[selected_rubric_label]
        selected_runs = rubric_groups[selected_rubric_id]
        all_items = []
        first_rubric = None
        for r in selected_runs:
            rdata = eval_repo.get_run(r["id"])
            if rdata and not first_rubric:
                first_rubric = rdata.get("rubric")
            all_items.extend(eval_repo.list_run_items(r["id"]))
        rubric = first_rubric or load_default_rubric()
        run_data = {
            "name": f"{rubric.name} 汇总",
            "rubric": rubric,
            "task_type": rubric.task_type,
            "source_run_ids": [run["id"] for run in selected_runs],
        }
        items = all_items
        st.caption(f"当前汇总 {len(selected_runs)} 个使用同一 Rubric 版本的任务。")
    else:
        selected_runs = [next(r for r in runs if r["id"] == run_id)]
        run_data = eval_repo.get_run(run_id)
        if not run_data:
            st.error("任务不存在")
            return
        rubric = run_data.get("rubric") or load_default_rubric()
        items = eval_repo.list_run_items(run_id)

    filter_cols = st.columns(3)
    status_values = sorted({it.get("status") for it in items if it.get("status")})
    asset_values = sorted({it.get("asset_type") for it in items if it.get("asset_type")})
    model_values = sorted({it.get("model_provider") for it in items if it.get("model_provider")})
    with filter_cols[0]:
        selected_statuses = st.multiselect("状态", status_values, placeholder="全部状态")
    with filter_cols[1]:
        selected_assets = st.multiselect(
            "资产类型",
            asset_values,
            format_func=lambda value: ASSET_TYPE_LABELS.get(value, value),
            placeholder="全部类型",
        )
    with filter_cols[2]:
        selected_models = st.multiselect("生成模型", model_values, placeholder="全部模型")
    items = filter_items(
        items,
        statuses=set(selected_statuses),
        asset_types=set(selected_assets),
        model_providers=set(selected_models),
    )

    summary = summarize_items(items, rubric)
    submitted_items = [it for it in items if it.get("status") == "submitted"]

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本总量", summary["total"])
    c2.metric("已评数量", summary["submitted"])
    c3.metric("平均总分", f"{summary['avg_total']:.2f}")
    completion = summary["submitted"] / summary["total"] * 100 if summary["total"] > 0 else 0
    c4.metric("评完率", f"{completion:.0f}%")

    st.progress(min(1.0, completion / 100.0))

    if summary["submitted"] < 5 and summary["total"] > 0:
        st.warning(":material/warning: 样本量过少（<5），统计结论可能不稳定，建议增加样本后再查看。")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/bar_chart: 维度分析",
        ":material/pie_chart: 分数分布",
        ":material/bug_report: Bad Case",
        ":material/thumb_down: 低分样本",
        ":material/psychology: 人机一致性",
    ])

    with tab1:
        st.markdown("#### 各维度均分")
        dim_data = {dim.label: round(summary["dim_avgs"].get(dim.key, 0.0), 2) for dim in rubric.dimensions}
        if dim_data and any(v > 0 for v in dim_data.values()):
            st.bar_chart(dim_data, horizontal=True, height=max(200, len(dim_data) * 35))
        else:
            st.caption("暂无评分数据")

        model_scores: dict[str, list[float]] = {}
        for it in submitted_items:
            mp = it.get("model_provider", "unknown")
            ts = it.get("total_score")
            if ts is not None:
                model_scores.setdefault(mp, []).append(ts)
        if len(model_scores) > 1:
            st.markdown("#### 模型对比")
            model_avgs = {k: round(sum(v) / len(v), 2) for k, v in model_scores.items() if v}
            mc1, mc2 = st.columns([2, 1])
            with mc1:
                st.bar_chart(model_avgs, horizontal=True, height=max(120, len(model_avgs) * 40))
            with mc2:
                for model, avg in sorted(model_avgs.items(), key=lambda x: -x[1]):
                    st.metric(model, f"{avg:.2f}", f"n={len(model_scores[model])}")
        elif model_scores:
            only_model = list(model_scores.keys())[0]
            st.info(f":material/info: 当前仅一个模型（{only_model}），无对比数据。需至少两个模型的评分才能进行模型对比。")

    with tab2:
        st.markdown("#### 总分分布")
        dist = summary["distribution"]
        dist_cols = st.columns(5)
        for i, (label, count) in enumerate(dist.items()):
            with dist_cols[i]:
                st.metric(label, count)
        if any(v > 0 for v in dist.values()):
            st.bar_chart(dist, height=200)

    with tab3:
        st.markdown("#### Bad Case 分布")
        tag_counts = summary["tag_counts"]
        if tag_counts:
            sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
            max_count = max(tag_counts.values())
            tag_data = {tag: cnt for tag, cnt in sorted_tags}
            st.bar_chart(tag_data, horizontal=True, height=max(200, len(tag_data) * 30))
            for tag, cnt in sorted_tags[:15]:
                pct = cnt / max_count
                st.caption(f"**{tag}**: {cnt}次")
                st.progress(pct)
        else:
            st.caption("暂无Bad Case标签")

    with tab4:
        st.markdown("#### 低分样本（总分 < 3.0）")
        low_items = [it for it in submitted_items if it.get("total_score") is not None and it["total_score"] < 3.0]
        if not low_items:
            st.success(":material/check_circle: 没有低分样本，整体质量良好！")
        else:
            st.caption(f"共 {len(low_items)} 个低分样本，按分数升序排列：")
            low_items.sort(key=lambda x: x.get("total_score", 5))
            for it in low_items[:20]:
                with st.container(border=True):
                    lc1, lc2, lc3 = st.columns([1, 1, 3])
                    with lc1:
                        fp = Path(it.get("asset_path", ""))
                        if fp.exists():
                            st.image(str(fp), use_container_width=True)
                    with lc2:
                        ts = it.get("total_score", 0)
                        st.metric("总分", f"{ts:.2f}")
                        st.caption(f"模型: {it.get('model_provider', '?')}")
                    with lc3:
                        prompt = it.get("prompt", "") or "（无Prompt）"
                        st.markdown(f"**Prompt**: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
                        item_tags = []
                        for t in it.get("bad_case_tags") or []:
                            tag_name = t.get("tag", "") if isinstance(t, dict) else str(t)
                            if tag_name:
                                item_tags.append(tag_name)
                        if item_tags:
                            st.caption("Bad Case: " + "、".join(f":red[{t}]" for t in item_tags))
                        note = it.get("note", "")
                        if note:
                            st.caption(f"备注: {note[:150]}")

    with tab5:
        st.markdown("#### 人评 vs 机评 一致性分析")
        agree = eval_repo.analyze_agreement(items)

        n_pairs = agree.get("n_pairs", 0)
        if n_pairs < 2:
            st.info(f":material/info: {agree.get('note', '需要至少2个同时完成人评和机评的样本才能计算一致性指标。请先对同一批素材完成人工评分和自动机评。')}")
        else:
            pr = agree["pearson_r"]
            sr = agree["spearman_rho"]
            ag1, ag2, ag3, ag4, ag5 = st.columns(5)
            ag1.metric("对比样本", n_pairs)
            ag2.metric("Pearson r", f"{pr:.3f}" if pr is not None else "N/A")
            ag3.metric("Spearman ρ", f"{sr:.3f}" if sr is not None else "N/A")
            ag4.metric("MAE", f"{agree['mae']:.2f}")
            ag5.metric("±1分一致率", f"{agree['within_one_rate']*100:.0f}%")

            if pr is not None:
                if pr >= 0.7:
                    st.success(f":material/check_circle: **一致性优秀** (r={pr:.2f})：VLM评分与人评高度一致，可作为自动化初筛或辅助评分工具。")
                elif pr >= 0.5:
                    st.success(f":material/thumb_up: **一致性良好** (r={pr:.2f})：VLM评分与人评整体一致，适合作为参考。")
                elif pr >= 0.3:
                    st.warning(f":material/warning: **一致性中等** (r={pr:.2f})：VLM评分有一定参考价值，但部分维度可能存在偏差，建议人工复核。")
                else:
                    st.error(f":material/error: **一致性较低** (r={pr:.2f})：VLM评分与人评偏差较大，建议检查评分标准(Prompt)或模型质量。")

            dim_label_map = {d.key: d.label for d in rubric.dimensions}
            dim_agree = agree.get("dimension_agreement", {})
            if dim_agree:
                st.markdown("##### 各维度一致性详情")
                rows = []
                for dk, stats in dim_agree.items():
                    label = dim_label_map.get(dk, dk)
                    rows.append({
                        "维度": label,
                        "n": stats["n"],
                        "Pearson r": stats.get("pearson"),
                        "MAE": stats.get("mae"),
                        "完全一致率": f"{stats.get('exact_rate',0)*100:.0f}%" if stats.get('exact_rate') is not None else "N/A",
                        "±1分率": f"{stats.get('within_one_rate',0)*100:.0f}%" if stats.get('within_one_rate') is not None else "N/A",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

            both_scored = [it for it in items if it.get("total_score") is not None and it.get("auto_total_score") is not None]
            if both_scored:
                st.markdown("##### 机评 vs 人评 分差排行（Δ=机评-人评）")
                both_scored.sort(key=lambda x: abs(x["auto_total_score"] - x["total_score"]), reverse=True)
                for it in both_scored[:10]:
                    delta = it["auto_total_score"] - it["total_score"]
                    icon = ":material/keyboard_double_arrow_up:" if delta > 0.5 else (":material/keyboard_double_arrow_down:" if delta < -0.5 else ":material/drag_indicator:")
                    color = "green" if abs(delta) < 0.5 else ("orange" if abs(delta) < 1.0 else "red")
                    with st.expander(f"{icon} {Path(it['asset_path']).name} · 人评 {it['total_score']:.1f} / 机评 {it['auto_total_score']:.1f} · :{color}[Δ{delta:+.1f}]"):
                        fc1, fc2 = st.columns([1, 2])
                        with fc1:
                            fp = Path(it["asset_path"])
                            if fp.exists():
                                st.image(str(fp), use_container_width=True)
                        with fc2:
                            st.caption(f"**Prompt**: {it.get('prompt','')[:200]}")
                            human_dim = it.get("scores", {})
                            auto_dim = it.get("auto_scores", {})
                            h_scores = {}
                            a_scores = {}
                            for dk, sv in human_dim.items():
                                h_scores[dim_label_map.get(dk, dk)] = sv.get("score", "?") if isinstance(sv, dict) else sv
                            for dk, sv in auto_dim.items():
                                a_scores[dim_label_map.get(dk, dk)] = sv.get("score", "?") if isinstance(sv, dict) else sv
                            if h_scores and a_scores:
                                st.markdown("| 维度 | 人评 | 机评 | Δ |")
                                st.markdown("|---|---|---|---|")
                                for dl in dim_label_map.values():
                                    hs = h_scores.get(dl, "-")
                                    aus = a_scores.get(dl, "-")
                                    try:
                                        d = int(aus) - int(hs)
                                        d_str = f"{d:+d}"
                                    except (TypeError, ValueError):
                                        d_str = "-"
                                    st.markdown(f"| {dl} | {hs} | {aus} | {d_str} |")
                            if it.get("auto_explanation"):
                                st.caption(f"**机评评价**: {it['auto_explanation'][:300]}")

    st.markdown("---")
    st.markdown("#### :material/arrow_downward: 导出数据")
    json_data = eval_export.export_run_to_json(run_data, items)
    csv_data = eval_export.export_run_to_csv(run_data, items)
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            ":material/download: 下载当前筛选 JSON",
            data=json_data,
            file_name=f"eval_export_{run_data['name']}.json",
            mime="application/json",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            ":material/table_chart: 下载当前筛选 CSV",
            data=csv_data.encode("utf-8-sig"),
            file_name=f"eval_export_{run_data['name']}.csv",
            mime="text/csv",
            use_container_width=True,
        )
