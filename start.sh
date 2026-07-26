#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "虚拟环境不存在，请先运行 ./setup.sh"
    exit 1
fi

CONFIG_VALUES=$(
    .venv/bin/python -c "from app.config import APP_PORT, MPS_ENABLE_FALLBACK, MPS_HIGH_WATERMARK_RATIO, MPS_LOW_WATERMARK_RATIO; print(f'{APP_PORT}|{MPS_ENABLE_FALLBACK}|{MPS_HIGH_WATERMARK_RATIO}|{MPS_LOW_WATERMARK_RATIO}')"
)
IFS='|' read -r APP_PORT PYTORCH_ENABLE_MPS_FALLBACK PYTORCH_MPS_HIGH_WATERMARK_RATIO PYTORCH_MPS_LOW_WATERMARK_RATIO <<< "$CONFIG_VALUES"
export PYTORCH_ENABLE_MPS_FALLBACK
export PYTORCH_MPS_HIGH_WATERMARK_RATIO
export PYTORCH_MPS_LOW_WATERMARK_RATIO

while true; do
    .venv/bin/streamlit run streamlit_app.py --server.port "$APP_PORT" "$@"
    status=$?

    case "$status" in
        134|137|139)
            echo "模型进程异常退出（状态码 $status），3 秒后自动恢复服务..." >&2
            sleep 3
            ;;
        *)
            exit "$status"
            ;;
    esac
done
