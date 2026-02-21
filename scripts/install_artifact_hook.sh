#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
git -C "$repo_root" config core.hooksPath .githooks
chmod +x "$repo_root/.githooks/pre-commit"

echo "Installed artifact pre-commit hook via core.hooksPath=.githooks"
echo "To disable: git config --unset core.hooksPath"
