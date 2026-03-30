"""
Modal app definition for Claudini GPU workers.

Deploy once with:
    modal deploy claudini/modal_app.py

Then use ModalBackend (CLAUDINI_BACKEND=modal) from the submit CLI — no pod IDs,
no host availability issues. Modal allocates GPUs globally across their fleet.

Required Modal secret (create at modal.com/secrets or via `modal secret create`):
    Name: claudini-secrets
    Keys: GH_TOKEN, CLAUDINI_REPO
    Optional: RUNPOD_API_KEY (not needed for Modal)

Required Modal volume (created automatically on first deploy):
    claudini-hf-cache  →  /root/.cache/huggingface
"""

from __future__ import annotations

import os
import modal

# ── Persistent model cache ────────────────────────────────────────────────────

hf_cache = modal.Volume.from_name("claudini-hf-cache", create_if_missing=True)

# ── Container image ───────────────────────────────────────────────────────────
# Deps are baked into the image (Modal caches layers).
# Code is pulled from git at runtime — so new optimizers are picked up
# without a redeploy. Only rebuild when pyproject.toml deps change.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "gpg")
    .run_commands(
        # GitHub CLI
        "mkdir -p /usr/share/keyrings",
        "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg "
        "-o /usr/share/keyrings/githubcli-archive-keyring.gpg",
        'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] '
        'https://cli.github.com/packages stable main" '
        "> /etc/apt/sources.list.d/github-cli.list",
        "apt-get update -q && apt-get install -y gh",
        # uv
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
)

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("claudini", image=image)

# ── Worker function ───────────────────────────────────────────────────────────


@app.function(
    gpu=os.environ.get("CLAUDINI_MODAL_GPU", "A10G"),
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("claudini-secrets")],
    timeout=4 * 3600,  # 4 hours — enough for any realistic benchmark run
)
def run_worker(once: bool = True) -> None:
    """
    Clone/update the repo, configure credentials, and run the worker daemon.

    Called by ModalBackend.start(). Runs on a GPU worker, processes the GitHub
    Issues queue, commits results, and exits when the queue is empty.
    """
    import subprocess

    repo_url = os.environ["CLAUDINI_REPO"]
    repo_dir = "/tmp/claudini"
    uv = "/root/.local/bin/uv"

    # ── repo ──────────────────────────────────────────────────────────────────
    if os.path.isdir(f"{repo_dir}/.git"):
        subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    # Install project and dependencies (ephemeral uv cache, packages take ~1min)
    subprocess.run([uv, "sync", "--extra", "quantize"], cwd=repo_dir, check=True)

    # ── git identity — derived from GH_TOKEN via GitHub API ──────────────────
    import json as _json
    import urllib.request

    gh_token = os.environ["GH_TOKEN"]
    req = urllib.request.Request(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req) as resp:
        profile = _json.loads(resp.read())
    git_name = profile["login"]
    git_email = f"{profile['id']}+{profile['login']}@users.noreply.github.com"
    subprocess.run(["git", "-C", repo_dir, "config", "user.email", git_email], check=True)
    subprocess.run(["git", "-C", repo_dir, "config", "user.name", git_name], check=True)

    # Configure git to use gh as a credential helper so push/pull work over HTTPS.
    subprocess.run(["gh", "auth", "setup-git"], check=True)

    # ── run worker ────────────────────────────────────────────────────────────
    # CLAUDINI_BACKEND is intentionally unset here — the worker exits normally
    # when the queue empties. Modal terminates the container automatically.
    cmd = [uv, "run", "-m", "claudini.pipeline.worker"]
    if once:
        cmd.append("--once")
    subprocess.run(cmd, cwd=repo_dir, check=True)


# ── Local entrypoint (for testing) ───────────────────────────────────────────


@app.local_entrypoint()
def main(once: bool = True) -> None:
    """
    Test the worker locally with:
        modal run claudini/modal_app.py
        modal run claudini/modal_app.py --no-once   # process full queue
    """
    run_worker.remote(once=once)
