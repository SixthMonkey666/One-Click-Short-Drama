#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

VERSION="${1:-$(date +%Y%m%d)}"
case "$VERSION" in
    *[!A-Za-z0-9._-]*)
        echo "版本号只能包含字母、数字、点、下划线和连字符" >&2
        exit 1
        ;;
esac

if ! command -v zip >/dev/null 2>&1; then
    echo "未找到 zip 命令，请先安装 zip" >&2
    exit 1
fi

mkdir -p dist
ARCHIVE="dist/storyboard-flow-${VERSION}-source.zip"
CHECKSUM="${ARCHIVE}.sha256"

INCLUDE_PATHS=(
    app
    tests
    .github
    .streamlit
    .dockerignore
    .env.example
    .gitignore
    CONTRIBUTING.md
    DEVELOPMENT.md
    Dockerfile
    LICENSE
    README.md
    SECURITY.md
    clean.sh
    docker-compose.yml
    package_release.sh
    pyproject.toml
    requirements-lock.txt
    requirements.txt
    setup.sh
    start.command
    start.sh
    stop.command
    streamlit_app.py
)

zip -X -FS -q -r "$ARCHIVE" "${INCLUDE_PATHS[@]}" \
    -x '*/__pycache__/*' '*.pyc' '*.pyo' '.DS_Store' '*/.DS_Store' \
    '.streamlit/secrets.toml' '*.log' '*.sqlite' '*.sqlite-*'

shasum -a 256 "$ARCHIVE" > "$CHECKSUM"

echo "源码包已生成：$ARCHIVE"
echo "校验文件已生成：$CHECKSUM"
