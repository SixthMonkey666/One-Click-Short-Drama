from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from app.config import RUBRICS_DIR


@dataclass
class ScoreAnchor:
    score: int
    description: str


@dataclass
class RubricDimension:
    key: str
    label: str
    weight: float = 1.0
    required: bool = True
    anchors: dict[int, str] = field(default_factory=dict)

    def anchor_text(self, score: int) -> str:
        return self.anchors.get(score, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "required": self.required,
            "anchors": {str(k): v for k, v in self.anchors.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RubricDimension:
        anchors_raw = d.get("anchors", {})
        anchors: dict[int, str] = {}
        for k, v in anchors_raw.items():
            try:
                anchors[int(k)] = str(v)
            except (ValueError, TypeError):
                pass
        return cls(
            key=d["key"],
            label=d["label"],
            weight=float(d.get("weight", 1.0)),
            required=bool(d.get("required", True)),
            anchors=anchors,
        )


@dataclass
class Rubric:
    version: str
    name: str
    task_type: str
    description: str = ""
    dimensions: list[RubricDimension] = field(default_factory=list)
    bad_case_tags: list[str] = field(default_factory=list)

    @property
    def total_weight(self) -> float:
        return sum(d.weight for d in self.dimensions)

    @property
    def required_dimensions(self) -> list[RubricDimension]:
        return [d for d in self.dimensions if d.required]

    @property
    def optional_dimensions(self) -> list[RubricDimension]:
        return [d for d in self.dimensions if not d.required]

    def get_dimension(self, key: str) -> RubricDimension | None:
        for d in self.dimensions:
            if d.key == key:
                return d
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "task_type": self.task_type,
            "description": self.description,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "bad_case_tags": list(self.bad_case_tags),
            "total_weight": self.total_weight,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Rubric:
        dims = [RubricDimension.from_dict(dim) for dim in d.get("dimensions", [])]
        return cls(
            version=d.get("version", "unknown"),
            name=d.get("name", "未命名Rubric"),
            task_type=d.get("task_type", "universal"),
            description=d.get("description", ""),
            dimensions=dims,
            bad_case_tags=list(d.get("bad_case_tags", [])),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Rubric:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_yaml_file(cls, path: Path) -> Rubric:
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to load YAML rubrics. Install with: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_string(cls, yaml_str: str) -> Rubric:
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to load YAML rubrics. Install with: pip install pyyaml")
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)


def load_default_rubric() -> Rubric:
    default_path = RUBRICS_DIR / "default.yaml"
    if default_path.exists():
        return Rubric.from_yaml_file(default_path)
    return _built_in_default_rubric()


def load_rubric(name: str) -> Rubric | None:
    path = RUBRICS_DIR / f"{name}.yaml"
    if path.exists():
        return Rubric.from_yaml_file(path)
    if name == "default":
        return load_default_rubric()
    return None


def list_rubric_files() -> list[dict[str, str]]:
    results = []
    if not RUBRICS_DIR.exists():
        return results
    for f in sorted(RUBRICS_DIR.glob("*.yaml")):
        try:
            r = Rubric.from_yaml_file(f)
            results.append({
                "filename": f.name,
                "version": r.version,
                "name": r.name,
                "task_type": r.task_type,
                "dimensions_count": str(len(r.dimensions)),
            })
        except Exception:
            results.append({
                "filename": f.name,
                "version": "error",
                "name": f.stem,
                "task_type": "unknown",
                "dimensions_count": "0",
            })
    return results


def _built_in_default_rubric() -> Rubric:
    return Rubric(
        version="v1.0-builtin",
        name="默认多模态评测 Rubric (内置)",
        task_type="universal",
        description="内置默认评测标准，当YAML文件不存在时使用",
        bad_case_tags=[
            "指令未遵循", "主体缺失", "主体/角色漂移", "属性错误", "场景跳变",
            "构图异常", "肢体或物理错误", "文字错误", "风格不一致", "画面瑕疵",
            "镜头重复", "动作不连贯", "跨帧闪烁", "Prompt幻觉", "其他",
        ],
        dimensions=[
            RubricDimension("instruction_following", "指令遵循", 1.0, True,
                {1: "严重偏离Prompt核心要求", 2: "明显遗漏核心要素", 3: "基本遵循Prompt", 4: "较好遵循指令", 5: "完美遵循Prompt"}),
            RubricDimension("content_correctness", "内容正确性", 1.0, True,
                {1: "严重事实错误", 2: "多处明显错误", 3: "基本正确", 4: "内容准确", 5: "完全正确"}),
            RubricDimension("composition_quality", "构图与视觉质量", 1.0, True,
                {1: "构图混乱画质极差", 2: "构图有明显问题", 3: "构图基本合理", 4: "构图良好画质清晰", 5: "构图优秀画面精美"}),
            RubricDimension("shot_usability", "镜头可用性", 1.0, True,
                {1: "完全不可用", 2: "需大幅修改", 3: "基本可用", 4: "可直接使用", 5: "完全可用表现优秀"}),
            RubricDimension("overall_usability", "整体可用性", 1.2, True,
                {1: "完全不可用必须重生", 2: "问题较多需大改", 3: "基本可用有瑕疵", 4: "质量较好可进下一环节", 5: "表现优秀可直接交付"}),
        ],
    )
