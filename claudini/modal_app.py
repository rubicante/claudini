"""
Modal app definition for Claudini GPU workers.

Deploy once with:
    modal deploy claudini/modal_app.py

Then use ModalBackend (CLAUDINI_BACKEND=modal) from the submit CLI — no pod IDs,
no host availability issues. Modal allocates GPUs globally across their fleet.

Required Modal secret (create at modal.com/secrets or via `modal secret create`):
    Name: claudini-secrets
    Keys: GH_TOKEN, GIT_EMAIL, CLAUDINI_REPO
    Optional: GIT_NAME, RUNPOD_API_KEY (not needed for Modal)

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
    )
    .pip_install(
        # Core ML stack — torch pulls in CUDA-enabled wheels from PyPI
        "torch>=2.0",
        "transformers>=4.40",
        "accelerate>=1.13.0",
        "bitsandbytes>=0.43",
        "datasets>=2.14",
        # Project dependencies
        "numpy>=2.0",
        "pyyaml>=6.0",
        "scipy>=1.10",
        "tqdm>=4.60",
        "typer>=0.9",
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
    import sys

    repo_url = os.environ["CLAUDINI_REPO"]
    repo_dir = "/tmp/claudini"

    # ── repo ──────────────────────────────────────────────────────────────────
    if os.path.isdir(f"{repo_dir}/.git"):
        subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    # Install the claudini package itself (deps already in image)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", "-e", repo_dir],
        check=True,
    )

    # ── git identity ──────────────────────────────────────────────────────────
    git_email = os.environ.get("GIT_EMAIL", "worker@claudini")
    git_name = os.environ.get("GIT_NAME", "Claudini Worker")
    subprocess.run(["git", "-C", repo_dir, "config", "user.email", git_email], check=True)
    subprocess.run(["git", "-C", repo_dir, "config", "user.name", git_name], check=True)

    # ── gh auth ───────────────────────────────────────────────────────────────
    subprocess.run(
        ["gh", "auth", "login", "--with-token"],
        input=os.environ["GH_TOKEN"],
        text=True,
        check=True,
    )

    # ── run worker ────────────────────────────────────────────────────────────
    # CLAUDINI_BACKEND is intentionally unset here — the worker exits normally
    # when the queue empties. Modal terminates the container automatically.
    cmd = [sys.executable, "-m", "claudini.pipeline.worker"]
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
