from __future__ import annotations

import atexit
import os
import shutil
import tempfile

TEST_DATA_DIR = tempfile.mkdtemp(prefix="storyboard_flow_tests_")
os.environ["APP_DATA_DIR"] = TEST_DATA_DIR
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("IMAGE_PROVIDER", "mock")
os.environ.setdefault("VIDEO_PROVIDER", "mock")
os.environ.setdefault("JUDGE_PROVIDER", "mock")

atexit.register(shutil.rmtree, TEST_DATA_DIR, ignore_errors=True)
