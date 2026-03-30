#!/usr/bin/env bash
# Bootstrap any fresh GPU machine for Claudini worker operation.
#
# Tested on Ubuntu 22.04 (RunPod PyTorch template).
# Provider-agnostic: works on RunPod, Lambda Labs, any SSH-accessible machine.
#
# Required environment variables (set before running, or export in ~/.bashrc):
#   CLAUDINI_REPO   — SSH or HTTPS clone URL of this repo
#   GH_TOKEN        — GitHub personal access token (repo + issues scope)
#   GIT_EMAIL       — git commit author email for result commits
#   GIT_NAME        — git commit author name  (default: "Claudini Worker")
#
# Usage:
#   GH_TOKEN=... CLAUDINI_REPO=git@github.com:you/claudini.git bash scripts/bootstrap.sh

set -euo pipefail

REPO_URL="${CLAUDINI_REPO:?Please set CLAUDINI_REPO to the git clone URL}"
REPO_DIR="${HOME}/claudini"
GH_TOKEN="${GH_TOKEN:-}"
GIT_EMAIL="${GIT_EMAIL:?Please set GIT_EMAIL for result commits}"
GIT_NAME="${GIT_NAME:-Claudini Worker}"

# ── uv ────────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Activate for remainder of this script
    source "${HOME}/.local/bin/env" 2>/dev/null || source "${HOME}/.cargo/env" 2>/dev/null || true
fi
echo "uv $(uv --version)"

# ── repo ──────────────────────────────────────────────────────────────────────
echo "==> Setting up repo at ${REPO_DIR}"
if [ -d "${REPO_DIR}/.git" ]; then
    git -C "${REPO_DIR}" pull --ff-only
else
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

echo "==> Installing Python dependencies"
uv sync --extra quantize

# ── GitHub CLI ────────────────────────────────────────────────────────────────
if ! command -v gh &>/dev/null; then
    echo "==> Installing gh CLI"
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
https://cli.github.com/packages stable main" \
        | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt-get update -q && sudo apt-get install -y gh
fi
echo "gh $(gh --version | head -1)"

echo "==> Authenticating gh CLI"
if [ -n "${GH_TOKEN}" ]; then
    echo "${GH_TOKEN}" | gh auth login --with-token
    echo "  gh authenticated via GH_TOKEN"
else
    echo "  WARNING: GH_TOKEN not set. Run 'gh auth login' manually before starting the worker."
fi

# ── git identity ──────────────────────────────────────────────────────────────
echo "==> Configuring git identity"
git config user.email "${GIT_EMAIL}"
git config user.name  "${GIT_NAME}"

# ── done ─────────────────────────────────────────────────────────────────────
echo ""
echo "Bootstrap complete."
echo ""
echo "To start the worker daemon:"
echo "  cd ${REPO_DIR} && uv run -m claudini.pipeline.worker"
echo ""
echo "To run exactly one job and exit:"
echo "  cd ${REPO_DIR} && uv run -m claudini.pipeline.worker --once"
