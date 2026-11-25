#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.nvm/versions/node/$(ls -1 "$HOME/.nvm/versions/node" 2>/dev/null | tail -n 1)/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "未检测到 Node.js，请先安装 Node.js (https://nodejs.org)"
  read -r -p "按回车键退出..."
  exit 1
fi

if [ ! -d "$ROOT/node_modules/playwright" ]; then
  echo "首次运行，正在安装依赖..."
  npm install
fi

echo "正在启动 Voice Agent QA Studio..."
node --env-file-if-exists="$ROOT/.env" "$ROOT/src/app.mjs"
