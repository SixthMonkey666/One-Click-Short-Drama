from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import repository as eval_repo
from tests import _test_env  # noqa: F401


class TestPairwiseArena(unittest.TestCase):
    def test_candidates_require_same_prompt_and_different_models(self):
        assets = [
            {"asset_path": "/a.png", "prompt": "same", "model_provider": "model-a"},
            {"asset_path": "/b.png", "prompt": "same", "model_provider": "model-b"},
            {"asset_path": "/c.png", "prompt": "different", "model_provider": "model-c"},
            {"asset_path": "/d.png", "prompt": "same", "model_provider": "model-a"},
        ]

        candidates = eval_repo.build_pairwise_candidates(assets)
        paths = {
            tuple(sorted((left["asset_path"], right["asset_path"])))
            for left, right in candidates
        }

        self.assertEqual(paths, {("/a.png", "/b.png"), ("/b.png", "/d.png")})

    def test_batch_deduplicates_existing_pairs(self):
        project_id = str(uuid4())
        assets = [
            {"asset_path": "/a.png", "prompt": "same", "model_provider": "model-a"},
            {"asset_path": "/b.png", "prompt": "same", "model_provider": "model-b"},
        ]

        first = eval_repo.create_pairwise_batch(assets, project_id=project_id)
        second = eval_repo.create_pairwise_batch(assets, project_id=project_id)

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(first[0]["prompt"], "same")

    def test_vote_requires_reason_and_supports_both_unusable(self):
        project_id = str(uuid4())
        pair = eval_repo.create_pairwise_comparison(
            "/a.png",
            "/b.png",
            "model-a",
            "model-b",
            project_id=project_id,
            prompt="same",
        )

        with self.assertRaises(ValueError):
            eval_repo.vote_pairwise(pair["id"], "A")

        voted = eval_repo.vote_pairwise(
            pair["id"],
            "both_unusable",
            reason_tags=["两者均不可用"],
        )
        self.assertEqual(voted["winner"], "both_unusable")
        self.assertEqual(voted["reason_tags"], ["两者均不可用"])

        stats = eval_repo.get_pairwise_stats(project_id=project_id)
        self.assertEqual(stats["total_votes"], 1)
        self.assertEqual(stats["leaderboard"][0]["elo"], 1000.0)
        self.assertTrue(all(entry["unusable"] == 1 for entry in stats["leaderboard"]))


if __name__ == "__main__":
    unittest.main()
