#!/bin/bash

# ==========================================
# Storyboard Flow 一键启动脚本
# 双击此文件即可启动服务
# ==========================================

# 获取脚本所在目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   🎬 Storyboard Flow - 一键成片${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ 错误：虚拟环境不存在${NC}"
    echo ""
    echo -e "${YELLOW}请先在终端运行：${NC}./setup.sh"
    echo ""
    read -n 1 -s -r -p "按任意键关闭窗口..."
    exit 1
fi

APP_PORT=$(.venv/bin/python -c "from app.config import APP_PORT; print(APP_PORT)")
APP_URL="http://localhost:${APP_PORT}/"

# 检查是否已有服务在运行
if lsof -ti:"$APP_PORT" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  检测到 ${APP_PORT} 端口已有服务在运行${NC}"
    echo ""
    read -p "是否先停止旧服务再重启？(y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}正在停止旧服务...${NC}"
        lsof -ti:"$APP_PORT" | xargs kill -15 2>/dev/null
        for _ in {1..5}; do
            lsof -ti:"$APP_PORT" >/dev/null 2>&1 || break
            sleep 1
        done
        if lsof -ti:"$APP_PORT" >/dev/null 2>&1; then
            lsof -ti:"$APP_PORT" | xargs kill -9 2>/dev/null
        fi
        echo -e "${GREEN}✅ 旧服务已停止${NC}"
        echo ""
    else
        echo -e "${GREEN}直接打开浏览器访问现有服务...${NC}"
        open "$APP_URL"
        exit 0
    fi
fi

echo -e "${GREEN}🚀 正在启动服务...${NC}"
echo -e "${BLUE}📍 访问地址：${APP_URL}${NC}"
echo -e "${YELLOW}💡 关闭此终端窗口即可停止服务${NC}"
echo ""

# 启动后自动打开浏览器（延迟 5 秒等待服务就绪）
(
    sleep 5
    open "$APP_URL"
) &

# 服务参数、MPS 内存配置和异常恢复统一由 start.sh 管理。
exec "$PROJECT_DIR/start.sh" "$@"
