#!/bin/bash

# ==========================================
# Storyboard Flow 一键停止脚本
# 双击此文件即可停止服务并释放内存
# ==========================================

# 获取脚本所在目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear
echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   Storyboard Flow - 停止服务${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

APP_PORT=8503
if [ -x ".venv/bin/python" ]; then
    APP_PORT=$(.venv/bin/python -c "from app.config import APP_PORT; print(APP_PORT)")
fi

PIDS=$(lsof -ti:"$APP_PORT" 2>/dev/null)
if [ -z "$PIDS" ]; then
    echo -e "${YELLOW}当前没有检测到运行中的服务${NC}"
else
    echo -e "${YELLOW}正在停止服务 (PID: $PIDS)...${NC}"
    echo "$PIDS" | xargs kill -15 2>/dev/null

    for i in {1..5}; do
        sleep 1
        if ! lsof -ti:"$APP_PORT" >/dev/null 2>&1; then
            break
        fi
        echo "等待进程退出... ($i/5)"
    done

    if lsof -ti:"$APP_PORT" >/dev/null 2>&1; then
        echo -e "${YELLOW}正常退出超时，正在强制终止本项目服务...${NC}"
        lsof -ti:"$APP_PORT" | xargs kill -9 2>/dev/null
        sleep 1
    fi

    if lsof -ti:"$APP_PORT" >/dev/null 2>&1; then
        echo -e "${RED}服务停止失败，${APP_PORT} 端口仍被占用。${NC}"
    else
        echo -e "${GREEN}服务已停止，进程占用的模型内存已由系统回收。${NC}"
    fi
fi

echo ""
read -n 1 -s -r -p "按任意键关闭窗口..."
