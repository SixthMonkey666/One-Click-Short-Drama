# Development Guide

本文档描述 Storyboard Flow 当前实现的开发契约。用户安装和功能说明见 [README.md](README.md)。

## 开发环境

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
LLM_PROVIDER=mock \
IMAGE_PROVIDER=mock \
VIDEO_PROVIDER=mock \
JUDGE_PROVIDER=mock \
.venv/bin/streamlit run streamlit_app.py
```

提交改动前运行：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q app streamlit_app.py tests
.venv/bin/python -m pip check
```

测试通过 `tests/_test_env.py` 把数据目录隔离到临时目录。新增测试不得读写仓库的 `data/projects.sqlite`。

## 架构

```text
streamlit_app.py
├── app/workflow.py
│   ├── app/adapters.py
│   ├── app/repository.py
│   └── app/providers/
└── app/evaluation/
    ├── rubric.py
    ├── repository.py
    ├── scoring.py
    ├── export.py
    ├── ui_manual.py
    ├── ui_auto_judge.py
    └── ui_arena.py
```

| 层 | 职责 |
| --- | --- |
| UI | Streamlit 页面、导航、表单和进度展示 |
| Workflow | 创作状态流转和跨阶段编排 |
| Adapter | Provider 调用、Prompt 组装、资产元数据和事件转换 |
| Repository | SQLite CRUD 和资产路径 |
| Provider | LLM、图像、视频和评分后端 |
| Evaluation | Rubric、评分、看板、导出、一致性和 A/B 对战 |

UI 不应直接加载模型或拼接数据库 SQL。Provider 的选择集中在 `app/providers/__init__.py`，路径和环境变量集中在 `app/config.py`。

## 数据与迁移

项目和评测共用 `APP_DATA_DIR/projects.sqlite`。LangGraph checkpoint 使用独立的 `langgraph_checkpoints.sqlite`。

评测相关表：

- `rubric_versions`
- `evaluation_runs`
- `evaluation_items`
- `dimension_scores`
- `bad_case_tags`
- `pairwise_comparisons`

迁移必须向后兼容：

1. 新表使用 `CREATE TABLE IF NOT EXISTS`。
2. 新列先通过 `PRAGMA table_info` 检查，再执行 `ALTER TABLE ... ADD COLUMN`。
3. 不删除或重命名现有列，不在启动时重建用户表。
4. 迁移错误必须向上抛出，不能把真实失败当作“列已存在”吞掉。
5. 测试必须覆盖重复执行迁移的幂等性。

Rubric YAML 只是新版本的来源。创建任务时，Rubric 内容被序列化到 `rubric_versions`，已有任务始终引用当时的不可变版本。跨任务汇总必须限定为同一个 `rubric_version_id`。

## 创作状态

主流程状态如下：

```text
new
  -> analyzing_idea
  -> awaiting_analysis_review
  -> generating_storyboard
  -> awaiting_storyboard_review
  -> generating_assets
  -> generating_video_prompts
  -> awaiting_video_prompt_review
  -> rendering_video
  -> completed
```

状态和中间结果通过 `app/repository.py` 持久化。长操作由生成器发出事件，UI 消费事件并更新进度。

常用事件：

| `type` | 用途 |
| --- | --- |
| `status` | 状态文本 |
| `text_start` / `text_delta` / `text_end` | LLM 流式文本 |
| `phase_start` | 资产阶段开始 |
| `asset_start` / `asset_progress` / `asset_done` | 单个资产生成 |
| `keyframe_start` / `keyframe_progress` / `keyframe_done` | 关键帧生成 |
| `asset_progress_overall` | 批量总进度 |
| `analysis_complete` / `storyboard_complete` / `assets_complete` | 阶段完成 |
| `video_prompts_complete` / `video_complete` | 视频阶段完成 |

生成器的最终结构化结果通过 `return` 传递，即 Python 的 `StopIteration.value`。不要把最终返回值误写成普通事件。

## Provider 契约

基类位于：

- `app/providers/base.py`
- `app/providers/judge_base.py`

所有 Provider 必须提供稳定的 `name`、面向 UI 的 `display_name`、幂等的 `release()` 和无副作用的 `is_available()`。

### LLMProvider

必须实现：

```python
generate_idea_stream() -> Generator[dict, None, list[dict]]
generate_analysis_stream(source_text) -> Generator[dict, None, dict]
generate_characters_stream(source_text) -> Generator[dict, None, list[dict]]
generate_storyboard_stream(source_text, analysis) -> Generator[dict, None, tuple[str, list[dict]]]
generate_video_prompts_stream(source_text, script, shots, characters=None) -> Generator[dict, None, list[dict]]
```

Analysis 至少包含标题、类型、主题、基调、视觉风格、目标时长、叙事摘要，以及 `characters`、`props`、`environments`。

Shot 至少包含：

```text
id, number, title, description, duration_seconds,
camera_angle, camera_movement, character_ids, prop_ids,
environment_id, prompt
```

### ImageProvider

```python
generate_image(
    prompt: str,
    output_path: Path,
    width: int = 1024,
    height: int = 576,
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[dict], None] | None = None,
    seed: int | None = None,
    reference_images: list[Path] | None = None,
) -> None
```

方法必须把最终 PNG 写到 `output_path`。角色视图、道具、场景和关键帧的 Prompt 组装属于 Adapter，不应复制到 Provider。
如果实现支持身份参考图，设置 `supports_reference_images = True`，并实际消费
`reference_images`；禁止静默忽略参考图后声称支持一致性。

### VideoProvider

```python
generate_video(project_id: str, shots: list[dict]) -> dict
```

返回值至少包含 `provider`、`status` 和 `path`。输入镜头可包含 `selected_image`、`duration_seconds` 和 `video_prompt`。

### JudgeProvider

```python
evaluate_image(
    image_path: Path,
    prompt: str,
    task_type: str,
    rubric: dict,
    reference_images: list[Path] | None = None,
) -> JudgeResult
```

视频仍通过这个入口传入；具体 Provider 根据 `task_type == "video"` 抽帧。返回结果应交给 `evaluate_validated()` 校验，不能直接持久化未经验证的模型输出。

`JudgeResult` 包含：

```text
dimension_scores, total_score, bad_case_tags,
explanation, confidence, raw_response
```

校验器会拒绝缺少必选维度、未知维度、非整数或越界分数、未知标签、空解释和越界置信度，并按 Rubric 重新计算总分。实现视频评分时必须在成功和异常路径都清理临时帧。

## 注册与加载 Provider

项目内置 `openai_compatible` 文本和图片 Provider，分别使用标准
`chat/completions`、`images/generations` 和 `images/edits` 接口。它们既可
连接本地模型服务器，也可连接托管 API。

仓库内长期维护的 Provider：

1. 在 `app/providers/` 新建模块并继承对应基类。
2. 在 `app/providers/__init__.py` 导入类。
3. 加入 `_LLM_REGISTRY`、`_IMAGE_REGISTRY`、`_VIDEO_REGISTRY` 或 `_JUDGE_REGISTRY`。
4. 在 `.env.example` 记录新配置，但不要加入真实密钥。
5. 添加接口、不可用状态和错误路径测试。

用户自己的 Provider 不需要修改工厂源码，可以通过模块入口动态加载：

```dotenv
LLM_PROVIDER=custom
CUSTOM_LLM_PROVIDER_CLASS=my_backends.llm:MyLLMProvider

IMAGE_PROVIDER=custom
CUSTOM_IMAGE_PROVIDER_CLASS=my_backends.image:MyImageProvider
```

入口必须使用 `module:Class` 格式，类必须继承对应基类。动态模块拥有当前
Python 进程的完整权限，只能加载受信任代码。

示例：

```python
_JUDGE_REGISTRY: dict[str, type[JudgeProvider]] = {
    "mock": MockJudgeProvider,
    "qwen3_vl": QwenVLJudgeProvider,
    "my_judge": MyJudgeProvider,
}
```

`auto` 只解析到对应的真实 Provider，不会回退到 Mock。未知名称会抛出
`ValueError`，模型或依赖不可用会抛出 `RuntimeError`。Mock Provider
只能在测试配置中显式选择；不要增加任何静默业务降级。

## 评测规则

### 人评和机评

- 人评分数与机评分数必须分开保存。
- 自定义 Bad Case 标签必须和预置标签一样持久化。
- 必选维度未完成时不能提交。
- 可选维度未评分时不参与加权分母。
- 导出只能包含当前筛选后的数据，并过滤敏感字段。

### A/B 竞技场

候选对必须同时满足同 Prompt、同目标位置、不同模型。创建批次时要去重已有组合。投票必须包含理由标签或说明，且允许 `both_unusable`。Elo 参数为初始 1000、`K=32`。

### 一致性

统计只使用同时存在人评和机评的样本。聚合看板应先完成筛选，再对筛选后的完整样本集合计算指标。Pearson 或 Spearman 在样本过少或方差为零时不应被包装成可靠结论。

## Streamlit 约束

- 页面导航统一通过明确的内部 view 状态驱动，选中态必须和主界面一致。
- 后台任务提交时应保存目标页面 context；离开任务页后才能显示返回入口。
- 长任务进度由 `st.fragment(run_every=...)` 读取任务快照，不由页面线程持有推理。
- 需要初始为空的 `st.radio` 必须设置 `index=None`。
- widget 实例化后，不要在同一次运行中写入对应的 `session_state` key。
- 批量选择等需要重置 widget 的场景，使用版本化 key。
- 文件访问统一使用 `Path` 和 Repository 提供的目录函数。

## 依赖与发布

`requirements.txt` 锁定 Docker/Mock 测试所需的轻量依赖，
`requirements-lock.txt` 额外锁定本地 Qwen、FLUX 推理依赖及
Diffusers 提交版本。生产安装统一通过 `setup.sh` 使用完整锁文件。

发布前检查：

1. 运行完整测试、编译检查和 `pip check`。
2. 使用全 Mock Provider 完成一次创建、生成、人工评分、机评、看板和竞技场流程。
3. 确认 `data/`、`.env`、模型权重、日志和压缩包未被纳入发布内容。
4. 确认示例配置不含密钥或本机绝对路径。
5. 检查 Docker 构建和 README 命令仍可执行。
6. 检查 `start.command`、`stop.command` 保持可执行权限并通过 Shell 语法检查。
7. 运行 `./package_release.sh`，检查 ZIP 内容并验证 SHA-256。

安全披露流程见 [SECURITY.md](SECURITY.md)，贡献要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
