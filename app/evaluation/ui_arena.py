from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.config import ELO_INITIAL_RATING, ELO_K_FACTOR
from app.evaluation import repository as eval_repo
from app.repository import get_project, list_projects

ARENA_REASON_TAGS = [
    "构图更好", "主体更清晰", "颜色/光影更好", "细节更丰富",
    "动作更自然", "场景更准确", "风格更一致", "Prompt遵循更好",
    "整体更优", "质量接近", "两者均有明显问题", "两者均不可用",
]


def render_arena() -> None:
    st.subheader(":material/sports_martial_arts: A/B 竞技场（盲测对战）")
    st.caption("匿名对比同一 Prompt 的两份产出，支持 A 胜、B 胜、平局或均不可用。模型来源在投票前保持隐藏。")

    arena_view = st.session_state.get("_arena_view", "vote")
    ac1, ac2, ac3, _ = st.columns([1, 1, 1, 4])
    with ac1:
        if st.button(":material/how_to_vote: 开始对战", use_container_width=True,
                     type="primary" if arena_view == "vote" else "secondary"):
            st.session_state["_arena_view"] = "vote"
            st.rerun()
    with ac2:
        if st.button(":material/emoji_events: 排行榜", use_container_width=True,
                     type="primary" if arena_view == "leaderboard" else "secondary"):
            st.session_state["_arena_view"] = "leaderboard"
            st.rerun()
    with ac3:
        if st.button(":material/settings: 发起对战", use_container_width=True,
                     type="primary" if arena_view == "setup" else "secondary"):
            st.session_state["_arena_view"] = "setup"
            st.rerun()

    st.markdown("---")

    if arena_view == "vote":
        _render_arena_vote()
    elif arena_view == "leaderboard":
        _render_arena_leaderboard()
    elif arena_view == "setup":
        _render_arena_setup()


def _render_arena_setup() -> None:
    st.markdown("#### :material/add: 发起新对战组")
    st.caption("仅将同一 Prompt、同一目标位置、不同模型的产出组成对战，左右随机互换并匿名展示。")

    projects = list_projects()
    if not projects:
        st.warning(":material/warning: 暂无项目，请先创建项目并生成资产。")
        return

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

    asset_type = st.selectbox(
        "对战资产类型",
        ["keyframe", "character", "prop", "environment"],
        format_func=lambda x: {"keyframe": "关键帧", "character": "角色设定图", "prop": "道具参考图", "environment": "场景概念图"}[x],
    )

    if st.button(":material/play_arrow: 生成对战组合", type="primary"):
        project = get_project(selected_pid)
        if not project:
            st.error("项目不存在")
            return
        all_assets = eval_repo.discover_project_assets(project)
        filtered = [a for a in all_assets if a["asset_type"] == asset_type]
        if len(filtered) < 2:
            st.warning(f":material/warning: 至少需要 2 个{ { 'keyframe': '关键帧', 'character': '角色设定图', 'prop': '道具参考图', 'environment': '场景概念图' }[asset_type] }才能发起对战，当前仅找到 {len(filtered)} 个。")
            return
        existing = eval_repo.list_pairwise_comparisons(project_id=selected_pid, task_type=asset_type, limit=1)
        if existing:
            st.warning(":material/info: 该项目已有对战记录，将追加新对战。")
        n_pairs = len(eval_repo.build_pairwise_candidates(filtered))
        created = eval_repo.create_pairwise_batch(filtered, task_type=asset_type, project_id=selected_pid)
        if not created:
            st.warning("没有可新增的公平对战组合。请确保同一 Prompt 下存在至少两个不同模型的产出，且组合尚未创建。")
            return
        st.success(f":material/check: 已创建 {len(created)} 组对战（发现 {n_pairs} 个有效组合）")
        st.session_state["_arena_view"] = "vote"
        st.rerun()


def _render_arena_vote() -> None:
    project_id = st.session_state.get("_arena_project_filter")
    pair = eval_repo.get_next_pairwise(project_id=project_id)

    stats = eval_repo.get_pairwise_stats(project_id=project_id)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("已投票", stats["total_votes"])
    sc2.metric("待投票", stats["pending"])
    sc3.metric("参与模型", len(stats["leaderboard"]))
    if stats["leaderboard"]:
        top = stats["leaderboard"][0]
        sc4.metric("当前第一", f"{top['model']} ({top['elo']:.0f})")
    else:
        sc4.metric("当前第一", "-")

    st.markdown("---")

    if not pair:
        st.info(":material/celebration: 当前没有待投票的对战。点击「发起对战」创建新的对战组。")
        if st.button(":material/settings: 去发起对战"):
            st.session_state["_arena_view"] = "setup"
            st.rerun()
        return

    if st.session_state.get("_arena_last_pair_id") != pair["id"]:
        for key in list(st.session_state.keys()):
            if key.startswith("_arena_tag_") or key.startswith("_arena_note_"):
                del st.session_state[key]
        st.session_state["_arena_last_pair_id"] = pair["id"]

    st.markdown("#### :material/description: 共同 Prompt")
    st.info(pair.get("prompt") or "（无 Prompt，不建议投票）")

    a_path = Path(pair["asset_a_path"])
    b_path = Path(pair["asset_b_path"])

    ac1, ac2 = st.columns(2, gap="large")

    with ac1:
        st.markdown("#### :material/visibility_off: 方案 A")
        if a_path.exists():
            st.image(str(a_path), use_container_width=True)
        else:
            st.error(f"文件不存在: {a_path.name}")
        if st.button(":material/thumb_up: A 更好", key="vote_a", type="primary", use_container_width=True):
            _submit_vote(pair, "A")

    with ac2:
        st.markdown("#### :material/visibility_off: 方案 B")
        if b_path.exists():
            st.image(str(b_path), use_container_width=True)
        else:
            st.error(f"文件不存在: {b_path.name}")
        if st.button(":material/thumb_up: B 更好", key="vote_b", type="primary", use_container_width=True):
            _submit_vote(pair, "B")

    st.markdown("##### :material/checklist: 选择理由")
    _render_reason_tags(pair["id"])
    note_key = f"_arena_note_{pair['id']}"
    st.text_area(
        "补充说明",
        key=note_key,
        placeholder="可填写具体选择理由",
        height=70,
    )
    st.caption("投票必须至少选择一个理由标签或填写补充说明。")
    tc1, tc2, tc3, _ = st.columns([2, 2, 2, 3])
    with tc1:
        if st.button(":material/handshake: 平局 / 难分高下", key="vote_tie", use_container_width=True):
            _submit_vote(pair, "tie")
    with tc2:
        if st.button(":material/block: 两者均不可用", key="vote_unusable", use_container_width=True):
            _submit_vote(pair, "both_unusable")
    with tc3:
        if st.button(":material/skip_next: 跳过（不投票）", use_container_width=True):
            st.rerun()


def _render_reason_tags(pair_id: str) -> None:
    tag_cols = st.columns(4)
    for i, tag in enumerate(ARENA_REASON_TAGS):
        tag_key = f"_arena_tag_{pair_id}_{i}"
        if tag_key not in st.session_state:
            st.session_state[tag_key] = False
        with tag_cols[i % 4]:
            st.checkbox(tag, key=tag_key)


def _submit_vote(pair: dict[str, Any], winner: str) -> None:
    pair_id = pair["id"]
    tags: list[str] = []
    note_key = f"_arena_note_{pair_id}"
    note = st.session_state.get(note_key, "")
    for i, tag in enumerate(ARENA_REASON_TAGS):
        tkey = f"_arena_tag_{pair_id}_{i}"
        if st.session_state.get(tkey):
            tags.append(tag)
    try:
        eval_repo.vote_pairwise(
            pair_id,
            winner=winner,
            reason_tags=tags,
            reason_note=note,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    for key in list(st.session_state.keys()):
        if key.startswith(f"_arena_tag_{pair_id}_") or key == note_key or key == "_arena_last_pair_id":
            del st.session_state[key]
    st.success(":material/check: 投票已记录")
    st.rerun()


def _render_arena_leaderboard() -> None:
    stats = eval_repo.get_pairwise_stats()

    if not stats["leaderboard"]:
        st.info(":material/info: 暂无投票数据，请先发起对战并完成投票。")
        return

    st.markdown("#### :material/emoji_events: Elo 排行榜")
    st.caption(
        f"Elo 评分系统：初始分 {ELO_INITIAL_RATING:g}，"
        f"胜者加分、败者扣分，平局各微调。K={ELO_K_FACTOR:g}。"
    )

    leaderboard = stats["leaderboard"]
    medals = ["🥇", "🥈", "🥉"]
    for i, entry in enumerate(leaderboard):
        medal = medals[i] if i < 3 else f"#{i+1}"
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 1, 1])
            with c1:
                st.markdown(f"### {medal}")
            with c2:
                st.markdown(f"**{entry['model']}**")
                win_rate_pct = entry["win_rate"] * 100
                st.caption(
                    f"胜率 {win_rate_pct:.0f}% · {entry['wins']}胜 "
                    f"{entry['losses']}负 {entry['ties']}平 · "
                    f"{entry.get('unusable', 0)}次均不可用"
                )
            with c3:
                st.metric("Elo", f"{entry['elo']:.0f}")
            with c4:
                bar_pct = min(1.0, entry["wins"] / max(1, entry["total"]))
                st.progress(bar_pct)
                st.caption("胜率")
            with c5:
                if entry["total"] > 0:
                    st.metric("场次", entry["total"])

    st.markdown("---")
    st.markdown("#### :material/history: 最近投票记录")
    recent = eval_repo.list_pairwise_comparisons(judged_only=True, limit=20)
    if not recent:
        st.caption("暂无投票记录")
    else:
        for rec in recent[:10]:
            if rec["winner"] == "tie":
                winner_label = "平局"
            elif rec["winner"] == "both_unusable":
                winner_label = "两者均不可用"
            else:
                winner_label = f"{rec['model_a']} (左)" if rec["winner"] == "A" else f"{rec['model_b']} (右)"
            with st.expander(f"{rec['model_a']} vs {rec['model_b']} → {winner_label}", expanded=False):
                ca, cb = st.columns(2)
                with ca:
                    pa = Path(rec["asset_a_path"])
                    if pa.exists():
                        st.image(str(pa), use_container_width=True)
                    st.caption(f"A (左): {rec['model_a']} {rec.get('model_a_version','')}")
                with cb:
                    pb = Path(rec["asset_b_path"])
                    if pb.exists():
                        st.image(str(pb), use_container_width=True)
                    st.caption(f"B (右): {rec['model_b']} {rec.get('model_b_version','')}")
                if rec.get("reason_note"):
                    st.caption(f"备注: {rec['reason_note']}")
