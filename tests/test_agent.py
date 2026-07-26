from __future__ import annotations

# ruff: noqa: I001

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import _test_env  # noqa: F401

from app.agent.checkpoint import create_checkpointer, thread_config
from app.agent.graph import (
    build_video_agent_graph,
    clear_agent_thread,
    get_video_agent_graph,
    resume_agent,
    start_agent,
)
from app.agent.nodes import storyboard_generation_node
from app.agent.policies import build_reflection
from app.agent.routers import route_storyboard_evaluation
from app.agent.tools import render_video, repair_failed_assets
from app.repository import create_project, get_project, save_project


def _initial(project_id: str) -> dict:
    return {
        "project_id": project_id, "thread_id": project_id, "user_goal": "测试",
        "constraints": {}, "retry_counts": {}, "evaluation_results": {},
        "feedback": {}, "failed_items": [], "errors": [], "completed_nodes": [],
        "execution_logs": [], "waiting_for_human": False, "estimated_cost": 0.0,
    }


class TestAgentGraph(unittest.TestCase):
    def setUp(self):
        from app import providers

        self.provider_patches = [
            patch.object(providers, "_LLM_PROVIDER_NAME", "mock"),
            patch.object(providers, "_IMAGE_PROVIDER_NAME", "mock"),
            patch.object(providers, "_VIDEO_PROVIDER_NAME", "mock"),
            patch.object(providers, "_JUDGE_PROVIDER_NAME", "mock"),
        ]
        for provider_patch in self.provider_patches:
            provider_patch.start()
        providers.release_all()

    def tearDown(self):
        from app import providers

        providers.release_all()
        for provider_patch in reversed(self.provider_patches):
            provider_patch.stop()

    def test_graph_compiles_with_expected_nodes_and_conditional_edges(self):
        graph = get_video_agent_graph().get_graph()
        expected = {
            "requirement_analysis", "planner", "storyboard_generation",
            "storyboard_evaluation", "storyboard_reflection",
            "storyboard_human_review", "asset_generation", "render_approval",
            "video_render", "error_handler",
        }
        self.assertTrue(expected.issubset(graph.nodes))
        self.assertTrue(any(edge.conditional for edge in graph.edges))

    def test_project_id_is_stable_thread_id(self):
        self.assertEqual(
            thread_config("project-123")["configurable"]["thread_id"],
            "project-123",
        )

    def test_storyboard_routes_use_real_result_and_retry_count(self):
        self.assertEqual(route_storyboard_evaluation({
            "evaluation_results": {"storyboard": {"passed": True}},
            "retry_counts": {}, "errors": [],
        }), "storyboard_human_review")
        self.assertEqual(route_storyboard_evaluation({
            "evaluation_results": {"storyboard": {"passed": False}},
            "retry_counts": {"storyboard": 0}, "errors": [],
        }), "storyboard_reflection")
        self.assertEqual(route_storyboard_evaluation({
            "evaluation_results": {"storyboard": {"passed": False}},
            "retry_counts": {"storyboard": 2}, "errors": [],
        }), "storyboard_human_review")

    def test_reflection_feedback_reaches_next_generation(self):
        evaluation = {
            "issues": ["镜头 3 包含过多连续动作"],
            "failed_dimensions": ["action_complexity"],
        }
        feedback = build_reflection(evaluation).model_dump(mode="json")
        state = {
            "project_id": "p1", "thread_id": "p1", "analysis": {},
            "feedback": feedback, "retry_counts": {"storyboard": 1},
        }
        with patch(
            "app.agent.nodes.tools.generate_storyboard",
            return_value=("script", [{"id": "shot-01"}]),
        ) as generate:
            storyboard_generation_node(state)
        self.assertEqual(generate.call_args.args[2], feedback)
        self.assertEqual(generate.call_args.args[3], 1)

    def test_checkpoint_can_be_opened_by_new_graph_instance(self):
        project = create_project("checkpoint", "一只猫回家")
        checkpoint_path = Path(tempfile.mkdtemp()) / "agent.sqlite"
        saver1 = create_checkpointer(checkpoint_path)
        graph1 = build_video_agent_graph(saver1)
        with patch("app.providers.mock.time.sleep"):
            graph1.invoke(_initial(project["id"]), thread_config(project["id"]))
        first_snapshot = graph1.get_state(thread_config(project["id"]))
        self.assertEqual(
            first_snapshot.next,
            ("storyboard_human_review",),
            first_snapshot.values.get("errors"),
        )

        saver2 = create_checkpointer(checkpoint_path)
        graph2 = build_video_agent_graph(saver2)
        restored = graph2.get_state(thread_config(project["id"]))
        self.assertEqual(restored.values["project_id"], project["id"])
        self.assertTrue(restored.interrupts)
        saver1.conn.close()
        saver2.conn.close()

    def test_interrupt_resume_preserves_user_storyboard_edits(self):
        project = create_project("edit", "一只猫回家")
        clear_agent_thread(project["id"])
        with patch("app.providers.mock.time.sleep"):
            first = start_agent(project["id"])
            edited = list(first["values"]["storyboard"])
            edited[0] = {**edited[0], "title": "人工修改标题"}
            second = resume_agent(project["id"], {
                "action": "edit_and_continue", "edited_storyboard": edited,
            })
        self.assertEqual(second["values"]["storyboard"][0]["title"], "人工修改标题")
        self.assertEqual(second["interrupts"][0]["review_type"], "render")
        self.assertEqual(get_project(project["id"])["shots"][0]["title"], "人工修改标题")

    def test_mock_provider_end_to_end_agent_flow(self):
        project = create_project("e2e", "雨夜里一只橘猫遇到善良女孩")
        clear_agent_thread(project["id"])
        with patch("app.providers.mock.time.sleep"):
            first = start_agent(project["id"])
            self.assertEqual(first["interrupts"][0]["review_type"], "storyboard")
            second = resume_agent(project["id"], {"action": "approve"})
            self.assertEqual(second["interrupts"][0]["review_type"], "render")
            final = resume_agent(project["id"], {"action": "approve"})
        self.assertFalse(final["next"])
        self.assertEqual(final["values"]["final_video"]["status"], "completed")
        self.assertEqual(final["values"]["current_stage"], "result_packaging")


class TestAgentIdempotency(unittest.TestCase):
    def test_failed_asset_repair_only_regenerates_failed_item(self):
        project = create_project("repair", "测试")
        project["analysis"] = {}
        project["shots"] = [
            {"id": "shot-01", "image_versions": [], "selected_image": None},
            {"id": "shot-02", "image_versions": [], "selected_image": None},
        ]
        save_project(project)
        with (
            patch("app.agent.tools.adapters.generate_keyframe_image_stream") as generator,
            patch("app.agent.tools._drain", return_value=(None, [])),
        ):
            repair_failed_assets(project["id"], [{"type": "keyframe", "id": "shot-02"}])
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(generator.call_args.args[1]["id"], "shot-02")

    def test_completed_video_side_effect_is_not_repeated(self):
        project = create_project("video", "测试")
        project["video"] = {"status": "completed", "path": "done.mp4"}
        save_project(project)
        with patch("app.agent.tools.adapters.generate_final_video") as generate:
            result = render_video(project["id"])
        generate.assert_not_called()
        self.assertEqual(result["path"], "done.mp4")


if __name__ == "__main__":
    unittest.main()
