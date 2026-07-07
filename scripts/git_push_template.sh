#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/git_push_template.sh git@github.com:<user>/<repo>.git

REMOTE_URL="${1:-}"
if [ -z "$REMOTE_URL" ]; then
  echo "Please provide remote url, e.g. git@github.com:<user>/<repo>.git"
  exit 1
fi

git status
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git branch -M main
git push -u origin main
