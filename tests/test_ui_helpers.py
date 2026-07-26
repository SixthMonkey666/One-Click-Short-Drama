from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.ui_manual import _collect_bad_case_tags
from tests import _test_env  # noqa: F401


class TestEvaluationUiHelpers(unittest.TestCase):
    def test_collects_preset_existing_and_custom_tags(self):
        state = {
            "_tag_item-1_构图异常": True,
            "_tag_item-1_已保存自定义": True,
            "_tag_item-1_未选择": False,
            "_custom_tag_item-1": "  本次自定义  ",
            "_tag_other_无关": True,
        }

        tags = _collect_bad_case_tags("item-1", state)

        self.assertEqual(
            {entry["tag"] for entry in tags},
            {"构图异常", "已保存自定义", "本次自定义"},
        )


if __name__ == "__main__":
    unittest.main()
