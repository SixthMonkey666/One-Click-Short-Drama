#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Storyboard Flow 一键安装 ==="
echo "项目目录: $PROJECT_DIR"
echo ""

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v python3.12 &>/dev/null; then
    PYTHON_BIN="python3.12"
fi

echo ">>> Python: $($PYTHON_BIN --version) ($(which $PYTHON_BIN))"

if [ -d ".venv" ] && [ ! -f ".venv/bin/streamlit" ]; then
    echo ">>> 清理不完整的虚拟环境..."
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo ""
    echo ">>> 创建虚拟环境..."
    "$PYTHON_BIN" -m venv .venv
fi

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
VENV_PIP="$PROJECT_DIR/.venv/bin/pip"

echo ""
echo ">>> 升级 pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo ""
if [ "${SKIP_MODEL_DEPS:-0}" = "1" ]; then
    echo ">>> 安装锁定的 Mock 测试依赖..."
    "$VENV_PIP" install -r requirements.txt
    echo ">>> 已跳过真实模型推理依赖（SKIP_MODEL_DEPS=1）"
else
    echo ">>> 安装锁定的生产依赖..."
    "$VENV_PIP" install -r requirements-lock.txt
    "$VENV_PYTHON" -c "import torch, transformers, diffusers, accelerate"
fi

if [ ! -f ".env" ]; then
    echo ""
    echo ">>> 创建 .env 配置文件..."
    cp .env.example .env
fi

mkdir -p data/assets

echo ""
echo ">>> 环境检查..."

QWEN_DIR=$("$VENV_PYTHON" -c "from app.config import LOCAL_QWEN_MODEL_DIR; print(LOCAL_QWEN_MODEL_DIR)")
if [ -d "$QWEN_DIR" ]; then
    echo "✅ Qwen3-VL-4B 模型目录: $QWEN_DIR"
else
    echo "⚠️  未在默认路径找到 Qwen3-VL-4B-Instruct 模型"
    echo "    请在 .env 中配置 LOCAL_QWEN_MODEL_DIR，缺少真实模型时服务会明确报错。"
fi

FLUX_DIR=$("$VENV_PYTHON" -c "from app.config import LOCAL_FLUX_MODEL_DIR; print(LOCAL_FLUX_MODEL_DIR)")
if [ -d "$FLUX_DIR" ]; then
    echo "✅ FLUX.2-klein-4B 模型目录: $FLUX_DIR"
else
    echo "⚠️  未在默认路径找到 FLUX.2-klein-4B 模型"
    echo "    请在 .env 中配置 LOCAL_FLUX_MODEL_DIR，缺少真实模型时服务会明确报错。"
fi

if command -v ffmpeg &>/dev/null; then
    echo "✅ FFmpeg 已安装: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "⚠️  未安装 FFmpeg（视频合成将仅输出清单文件，运行 brew install ffmpeg 安装）"
fi

echo ""
"$VENV_PYTHON" -c "
import sys
try:
    import torch
    if torch.backends.mps.is_available():
        print('✅ MPS (Metal GPU) 可用，推理将使用 GPU 加速')
    elif torch.cuda.is_available():
        print('✅ CUDA GPU 可用')
    else:
        print('⚠️  GPU 不可用，推理将使用 CPU（较慢）')
except ImportError:
    print('⚠️  PyTorch 未安装，本地模型推理不可用')
" 2>/dev/null || true

echo ""
echo "========================================="
echo "✅ 安装完成！"
echo ""
echo "启动命令："
echo "  cd \"$PROJECT_DIR\""
echo "  ./start.sh"
echo ""
echo "仅测试时显式启用 Mock："
echo "  SKIP_MODEL_DEPS=1 ./setup.sh"
echo "  LLM_PROVIDER=mock IMAGE_PROVIDER=mock VIDEO_PROVIDER=mock JUDGE_PROVIDER=mock ./start.sh"
echo "========================================="
