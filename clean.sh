#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "正在清理项目缓存和临时文件..."
find app tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find app tests -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf .pytest_cache .ruff_cache

for temp_dir in tmp temp; do
    if [ -d "$temp_dir" ]; then
        find "$temp_dir" -mindepth 1 -delete
    fi
done

echo "清理完成。虚拟环境、项目数据库和生成资产均已保留。"
