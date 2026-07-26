# Storyboard Flow

本地优先的 AI 分镜、视觉资产与多模态评测工作台。

Storyboard Flow 将“一句话创意”组织成一条可恢复、可审核、可评测的生产流程：

```text
创意
  → 需求分析
  → 视觉设定
  → 分镜脚本
  → 角色 / 道具 / 场景资产
  → 镜头关键帧
  → 视频运动提示词
  → FFmpeg 预览视频
  → 人工评测 / VLM 自动评测 / A/B 盲测
```

项目使用 Streamlit 提供交互界面，使用 LangGraph 编排 Agent 流程，通过
Provider 接口接入本地模型、OpenAI-compatible API 和 FFmpeg。内置本地实现
是 Qwen3-VL 与 FLUX；用户也可以连接 Ollama、vLLM、LM Studio、LocalAI、
兼容的云端 API，或加载自己的 Python Provider。

> 当前视频后端根据镜头时长拼接关键帧生成 MP4 预览，并不执行生成式
> Image-to-Video。项目适合验证工作流、资产一致性、任务恢复和评测闭环，
> 不能直接替代成熟的视频生成或剪辑产品。

## 这个项目适合什么

它对以下真实工程场景有直接参考价值：

- 本地大模型和图像模型在统一内存设备上的串行调度与释放。
- Streamlit 页面切换后仍持续运行的后台生成任务。
- LangGraph checkpoint、人工审核中断和恢复。
- 从创作资产到人工评分、VLM 评分和 A/B 盲测的数据闭环。
- 角色身份提示词、稳定 seed 和参考图条件控制。
- 可插拔 LLM、图像、视频与 Judge Provider。
- SQLite 数据迁移、任务快照、评测版本与导出。

它目前不适合直接承担：

- 多用户、高并发或跨机器任务调度。
- 无认证公网服务。
- 商业级非线性剪辑、音频制作或真实镜头运动生成。
- 严格 SLA、分布式存储或 Kubernetes 生产部署。

因此，更准确的定位是：**可运行的本地 AI 视频 Agent 工程样板和评测工作台**。

## 主要能力

### 创作与 Agent

- 自然语言创意输入和智能创意候选。
- 角色、道具、场景、主题、基调、视觉风格和时长分析。
- 2–10 镜结构化分镜，包含时长、景别、运镜和资产引用。
- 分镜结构评测、有限自动反思和定向重生。
- LangGraph `interrupt()` 人工审核与 `Command(resume=...)` 恢复。
- 角色四视图、道具图、场景图和镜头关键帧。
- 缺失资产或失败资产定向补生成。
- 视频运动提示词审核和 FFmpeg MP4 预览。

### 角色与画面一致性

- 角色身份锚点系统提示词。
- 同一角色四视图共享稳定 seed。
- 正面图作为侧面、背面和面部特写的视觉条件。
- 关键帧注入不可变角色身份块。
- 关键帧可使用出场角色的规范参考图。
- `FLUX_MAX_REFERENCE_IMAGES` 控制一致性与峰值内存的平衡。

### 后台任务与进度

- 全应用只允许一个重型生成或评测任务运行。
- 后台线程不依赖当前 Streamlit 页面生命周期。
- SQLite 持久化状态、阶段、当前对象、总体进度、事件和心跳。
- 离开任务页面后显示动态“返回当前任务”入口。
- 自动识别一键成片任务和自动机评任务的目标页面。
- 任务主界面和侧边栏显示同一份实时进度快照。

### 多模态评测

- YAML Rubric 和不可变数据库版本。
- 人工多维评分、权重、必评项和 1–5 分锚点。
- 预置与自定义 Bad Case 标签。
- Qwen3-VL 图片评测和视频抽帧评测。
- Judge 输出结构、维度、分值、标签与置信度校验。
- 人评与机评独立存储。
- Pearson、Spearman、MAE、RMSE 和一致率分析。
- 同 Prompt、同目标、不同模型的 A/B 匿名盲测。
- Elo 排名和 JSON / CSV 导出。

## 技术栈

| 层 | 实现 |
| --- | --- |
| Web UI | Streamlit 1.59 |
| Agent | LangGraph `StateGraph` |
| Schema | Pydantic v2 |
| LLM / VLM | Qwen3-VL；OpenAI-compatible；自定义 Provider |
| 图像生成 | FLUX；OpenAI-compatible；自定义 Provider |
| 视频预览 | FFmpeg |
| 数据 | SQLite + 本地文件系统 |
| 可观测性 | 后台任务快照；可选 Langfuse |
| 测试 | `unittest` + 显式 Mock Provider |

## 架构

```text
Streamlit UI
  │
  ├─ Global Generation Task Manager
  │    ├─ single active-task lock
  │    ├─ background worker
  │    └─ SQLite heartbeat / event snapshots
  │
  ├─ LangGraph Video Agent
  │    ├─ deterministic planner and policies
  │    ├─ checkpoint
  │    └─ interrupt / resume
  │
  ├─ Evaluation Center
  │    ├─ human evaluation
  │    ├─ Qwen VLM judge
  │    ├─ dashboard
  │    └─ A/B arena
  │
  └─ Provider Factory
       ├─ LLM: Qwen3-VL / OpenAI-compatible / Custom / Mock
       ├─ Image: FLUX / OpenAI-compatible / Custom / Mock
       ├─ Video: FFmpeg / Mock
       └─ Judge: Qwen3-VL / Mock
```

Agent 主路径：

```text
START
  → requirement_analysis
  → planner
  → visual_bible
  → storyboard_generation
  → storyboard_evaluation
      ├─ storyboard_reflection → storyboard_generation
      └─ storyboard_human_review [interrupt]
  → asset_planning
  → asset_generation
  → asset_evaluation
      ├─ asset_repair → asset_evaluation
      └─ asset_human_review [interrupt]
  → video_prompt_generation
  → video_prompt_evaluation
  → render_approval [interrupt]
  → video_render
  → final_evaluation
  → result_packaging
  → END
```

Planner 当前为受代码约束的规则规划器，不会执行模型返回的任意节点名。

## 快速开始

### 方案 A：Mock 模式

Mock 模式不下载 Qwen 或 FLUX，适合先验证 UI、Agent、数据库和评测流程。

```bash
git clone <your-repository-url>
cd storyboard-flow

SKIP_MODEL_DEPS=1 ./setup.sh

LLM_PROVIDER=mock \
IMAGE_PROVIDER=mock \
VIDEO_PROVIDER=mock \
JUDGE_PROVIDER=mock \
HOT_TOPIC_SEARCH_ENABLED=false \
./start.sh
```

打开：

```text
http://localhost:8503
```

Mock 图像和视频 manifest 只用于开发和测试，不代表真实生成质量。

### 方案 B：Apple Silicon 本地模型

推荐环境：

- Apple Silicon Mac。
- macOS。
- Python 3.12。
- FFmpeg。
- 建议至少 16GB 统一内存；更高分辨率和更多参考图需要更多内存。
- 安装依赖和下载模型时需要网络；准备完成后核心流程可离线运行。

安装 FFmpeg：

```bash
brew install ffmpeg
```

默认模型目录：

```text
~/ai_models/
├── Qwen3-VL-4B-Instruct/
└── FLUX.2-klein-4B/
```

模型权重不随源码发布。请自行下载，并确认对应模型许可证允许你的使用场景。

安装和启动：

```bash
cp .env.example .env
./setup.sh
./start.sh
```

macOS 完成首次安装后，也可以双击 `start.command` 启动服务，使用
`stop.command` 停止服务并释放模型内存。

自定义模型路径：

```dotenv
LOCAL_QWEN_MODEL_DIR=/absolute/path/to/Qwen3-VL-4B-Instruct
LOCAL_FLUX_MODEL_DIR=/absolute/path/to/FLUX.2-klein-4B
```

## 使用流程

### 一键成片

1. 新建项目并输入创意。
2. 查看 AI 分析，确认或修改结构化需求。
3. 查看分镜，必要时重新生成或人工编辑。
4. 生成角色、道具、场景和关键帧。
5. 确认逐镜头视频提示词。
6. 使用 FFmpeg 输出 MP4 预览。

生成期间可以切换页面。侧边栏保留任务状态、心跳和进度；离开目标页面后，
“返回当前任务”按钮会回到对应的一键成片或自动机评页面。

### 评测中心

1. 创建评测任务并选择项目资产。
2. 使用“人工评测”完成 Rubric 评分。
3. 使用“自动机评”串行执行 Qwen3-VL 评分。
4. 在评测看板查看分布、Bad Case 和人机一致性。
5. 在 A/B 竞技场进行匿名对比和 Elo 排名。
6. 导出当前筛选结果。

## 本地模型内存策略

项目针对 Apple Silicon 统一内存做了保守设计：

- Qwen、FLUX 和 VLM Judge 共用 FIFO 单槽推理队列。
- 所有文本、图像、视频与自动评测任务全局串行。
- 切换模型前释放另一类 Provider。
- FLUX 默认使用分阶段 model CPU offload。
- MPS OOM 时清理缓存并有限重试 sequential CPU offload。
- 支持 VAE tiling。
- Judge 会限制输入像素和参考图数量。
- Judge 可在 MPS OOM 后回退到 CPU。
- Judge 默认每个样本结束后卸载模型并清理缓存。
- 推理前检查可回收内存水位。
- 成功、失败和取消路径都会释放队列槽位和模型缓存。

不要把 `PYTORCH_MPS_HIGH_WATERMARK_RATIO` 设置为 `0.0`。这会取消保护上限，
可能导致系统无响应。

如果仍然 OOM，依次尝试：

1. 使用 `512x288` 或 `768x432`。
2. 设置 `FLUX_MPS_OFFLOAD_MODE=sequential`。
3. 减少 `FLUX_MAX_REFERENCE_IMAGES`。
4. 降低 `QWEN_JUDGE_MAX_PIXELS`。
5. 保持 `QWEN_JUDGE_RELEASE_EACH_ITEM=true`。
6. 关闭其他占用统一内存的应用。

## 配置

复制 `.env.example` 为 `.env`。运行时代码统一从 `app/config.py` 读取配置。

### Provider

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `qwen3_vl` | 创意、分析、分镜与提示词 |
| `IMAGE_PROVIDER` | `flux` | 角色、场景和关键帧 |
| `VIDEO_PROVIDER` | `ffmpeg` | MP4 预览 |
| `JUDGE_PROVIDER` | `qwen3_vl` | 自动机评 |
| `LOCAL_QWEN_MODEL_DIR` | `~/ai_models/Qwen3-VL-4B-Instruct` | Qwen 路径 |
| `LOCAL_FLUX_MODEL_DIR` | `~/ai_models/FLUX.2-klein-4B` | FLUX 路径 |

生产 Provider 不会静默降级成 Mock。模型或依赖不可用时会明确报错。

### 替换文字模型

现有 Qwen 和 FLUX 本地 Provider 是优化后的内置实现，但工作流不会直接依赖
它们。文字工作流只依赖 `LLMProvider` 的结构化生成接口。

连接本地 OpenAI-compatible 服务，以 Ollama 为例：

```dotenv
LLM_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_LLM_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_COMPATIBLE_LLM_MODEL=qwen3:8b
OPENAI_COMPATIBLE_LLM_API_KEY=
```

同样可以把 Base URL、模型 ID 和 API Key 换成 vLLM、LM Studio、LocalAI
或云服务商提供的值。服务必须兼容流式
`POST /v1/chat/completions`，并能稳定遵循项目要求的 JSON Schema。

### 替换图片模型

```dotenv
IMAGE_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_IMAGE_BASE_URL=https://api.example.com/v1
OPENAI_COMPATIBLE_IMAGE_MODEL=provider-image-model-id
OPENAI_COMPATIBLE_IMAGE_API_KEY=replace-me
```

图片服务至少需要实现 `POST /v1/images/generations`，返回 `b64_json` 或
URL。要保持同一角色的一致性，服务还应实现
`POST /v1/images/edits`；项目会把规范角色参考图以 multipart 图片字段发送。
如果服务没有 edits 接口，普通文生图仍可使用，但角色参考图任务会明确失败，
不会悄悄丢弃一致性条件。

修改 `.env` 后重启应用即可切换 Provider。密钥只放在本机 `.env`，不要提交。

### 接入任意自定义模型

非 OpenAI-compatible 的本地推理代码或私有 API 可以作为 Python 插件加载：

```dotenv
LLM_PROVIDER=custom
CUSTOM_LLM_PROVIDER_CLASS=my_backends.llm:MyLLMProvider

IMAGE_PROVIDER=custom
CUSTOM_IMAGE_PROVIDER_CLASS=my_backends.image:MyImageProvider
```

自定义类分别继承 `app.providers.base.LLMProvider` 或 `ImageProvider`。模块只在
用户明确选择 `custom` 后导入。接口契约和最小实现见
[DEVELOPMENT.md](DEVELOPMENT.md)。

### 服务与数据

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_PORT` | `8503` | Streamlit 端口 |
| `APP_DATA_DIR` | `data` | 数据库和资产根目录 |
| `DEFAULT_IMAGE_RESOLUTION` | `768x432 (快速预览)` | 新项目默认分辨率 |
| `HOT_TOPIC_SEARCH_ENABLED` | `true` | 智能创意热点；关闭后可完全离线 |

### 常用内存参数

| 变量 | 默认值 |
| --- | --- |
| `QWEN_MIN_AVAILABLE_GB` | `3.0` |
| `QWEN_JUDGE_MIN_AVAILABLE_GB` | `4.0` |
| `QWEN_JUDGE_MAX_PIXELS` | `147456` |
| `QWEN_JUDGE_MAX_INPUT_IMAGES` | `3` |
| `QWEN_JUDGE_CPU_FALLBACK` | `true` |
| `QWEN_JUDGE_RELEASE_EACH_ITEM` | `true` |
| `FLUX_MIN_AVAILABLE_GB` | `4.0` |
| `FLUX_MPS_OFFLOAD_MODE` | `model` |
| `FLUX_MPS_OOM_RETRIES` | `1` |
| `FLUX_ENABLE_VAE_TILING` | `true` |
| `FLUX_MAX_REFERENCE_IMAGES` | `2` |

其余生成、Agent、Judge、热点和 Langfuse 参数见 `.env.example`。

## 部署

| 方式 | 真实模型 | 推荐用途 |
| --- | --- | --- |
| macOS + `setup.sh` / `start.sh` | 支持，主要测试路径 | 本地创作和开发 |
| Docker Compose | 默认 Mock；可连接兼容 API | 演示、API 部署和 CI |
| Linux / CUDA | 代码包含设备分支，但未作为主要验证环境 | 实验性 |
| 公网裸露 Streamlit | 不推荐 | 无认证、无 TLS |

### Docker Mock 演示

```bash
docker compose up --build
docker compose logs -f
docker compose down
```

Compose 强制使用四类 Mock Provider，并把 `./data` 挂载到 `/app/data`。
Docker 镜像只安装 `requirements.txt`，不包含 PyTorch、Transformers、
Diffusers 或模型权重。

容器也可以连接外部 OpenAI-compatible Provider。若 API 服务运行在宿主机，
macOS/Windows Docker 中应把 Base URL 主机名写成 `host.docker.internal`，
而不是容器自身的 `127.0.0.1`。直接本地加载 Qwen/FLUX 权重仍建议使用
`setup.sh`，因为轻量 Docker 镜像不包含推理依赖。

如果需要远程访问，请在可信反向代理后配置 HTTPS、身份认证和访问控制。

## 数据目录

```text
data/
├── projects.sqlite
├── langgraph_checkpoints.sqlite
└── assets/
    └── <project-id>/
        ├── characters/
        ├── props/
        ├── environments/
        ├── <shot-id>/
        └── final/
```

- `projects.sqlite` 保存项目、Rubric、评测、评分和 A/B 对战。
- `langgraph_checkpoints.sqlite` 保存 Agent checkpoint。
- `data/`、模型、媒体和 `.env` 默认不进入 Git、Docker 镜像或发布包。
- 删除项目前请备份需要保留的素材。

## 项目结构

```text
.
├── streamlit_app.py
├── app/
│   ├── agent/
│   ├── evaluation/
│   ├── providers/
│   ├── rubrics/
│   ├── adapters.py
│   ├── config.py
│   ├── local_models.py
│   ├── repository.py
│   ├── task_manager.py
│   └── workflow.py
├── tests/
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-lock.txt
├── setup.sh
├── start.sh
├── package_release.sh
└── LICENSE
```

模块和 Provider 开发契约见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 开发与测试

轻量开发环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install ruff==0.12.5
```

验证：

```bash
.venv/bin/ruff check app streamlit_app.py tests
.venv/bin/python -m compileall -q app streamlit_app.py tests
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
bash -n start.sh setup.sh clean.sh start.command stop.command
docker compose config --quiet
```

当前代码包含 80 项测试，覆盖 Agent 图与恢复、Provider 严格模式与可替换
本地/API 后端、任务锁和进度快照、角色一致性、MPS OOM 回退、评测存储、
统计、导出与 A/B 竞技场。

GitHub Actions 在 Python 3.12 和全 Mock 环境中执行 lint、编译、依赖检查、
测试和 Docker 构建验证，不下载模型或读取本地运行数据。

## 发布源码包

```bash
./package_release.sh
```

脚本在 `dist/` 生成源码 ZIP 和 SHA-256 文件。发布包只包含代码、测试、配置
模板、CI、脚本和工程文档，不包含：

- `.env` 或密钥。
- `.venv/`。
- `data/`、SQLite、用户素材和生成资产。
- 模型权重与缓存。
- 日志、媒体、Python 缓存或旧压缩包。

## 已知限制

- Planner 当前是规则驱动，不是自由规划 LLM Agent。
- 分镜自动评测以确定性结构规则为主。
- 最终视频评测尚未自动接入完整视频 VLM Judge。
- FFmpeg 输出是关键帧预览，不是真实 I2V。
- 单机后台任务不能在 Python 进程重启后继续推理；旧任务会标记为中断。
- 全局单任务策略提高了 16GB 机器成功率，但牺牲了吞吐量。
- 自动机评 CPU 回退可能非常慢。
- 轻量 Docker 镜像支持 Mock 和兼容 API，不包含直接加载 Qwen/FLUX 的依赖。

## 安全与隐私

- 不要提交 `.env`、`data/`、模型、数据库或私有素材。
- 应把模型输出、上传文件、Prompt 和外部响应视为不可信输入。
- Langfuse 或自定义云端 Provider 可能把 Prompt、元数据或素材发送到第三方。
- 导出敏感字段过滤不能替代发布前人工检查。
- 安全问题请按 [SECURITY.md](SECURITY.md) 私密报告。

## 贡献

提交 Pull Request 前请阅读：

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)
- [SECURITY.md](SECURITY.md)

## License

代码采用 [MIT License](LICENSE)。

模型权重、第三方依赖、外部 API 和生成内容分别受其自身许可证与条款约束。
