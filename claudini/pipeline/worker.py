"""
Worker daemon — runs on any remote GPU machine.

Usage:
    uv run -m claudini.pipeline.worker          # poll until queue empty, then exit
    uv run -m claudini.pipeline.worker --once   # claim and run exactly one job, then exit
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from claudini.backends import get_backend
from .queue import claim, list_queued, post_failure, post_result

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parents[2]
POLL_INTERVAL = 30  # seconds between queue polls when idle


# ── git helpers ───────────────────────────────────────────────────────────────


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, text=True)


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _pull_latest() -> None:
    _git(["pull", "--ff-only"])


def _commit_and_push_results(issue_number: int) -> str:
    """Stage any new files under results/, commit, push. Returns the new short SHA."""
    _git(["add", "-f", "results/"])

    # Nothing to commit if the index is clean after staging
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if staged.returncode == 0:
        logger.info("no new result files to commit")
        return _git_output(["rev-parse", "--short", "HEAD"])

    msg = f"results: job #{issue_number}"
    _git(["commit", "-m", msg])

    # Discard any uncommitted changes (e.g. uv.lock updates from uv sync)
    # before rebasing — we only want to push the results commit.
    _git(["checkout", "--", "."])

    # Rebase on top of any commits pushed while we were running, then push
    _git(["pull", "--rebase"])
    _git(["push"])

    return _git_output(["rev-parse", "--short", "HEAD"])


# ── job execution ─────────────────────────────────────────────────────────────


def _run_benchmark(spec) -> None:
    """Invoke run_bench as a subprocess, streaming output to the terminal."""
    cmd = ["uv", "run", "-m", "claudini.run_bench"] + spec.to_bench_args()
    logger.info("executing: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"run_bench exited with code {result.returncode}")


def _build_summary(spec) -> str:
    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| Method | `{spec.method}` |",
        f"| Preset | `{spec.preset}` |",
        f"| Samples | `{spec.samples}` |",
        f"| Seeds | `{spec.seeds}` |",
    ]
    if spec.max_flops:
        lines.append(f"| Max FLOPs | `{spec.max_flops:.2e}` |")
    lines.append("")
    lines.append("Full per-step traces are in `results/` — run `git pull` to fetch them.")
    return "\n".join(lines)


# ── main loop ─────────────────────────────────────────────────────────────────


def run(once: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger.info("worker starting  (once=%s)", once)

    backend = get_backend()
    if backend:
        logger.info("backend: %s (will self-stop when queue empties)", backend.name)

    while True:
        _pull_latest()
        jobs = list_queued()

        if not jobs:
            if backend:
                logger.info("queue is empty — stopping backend %s", backend.name)
                backend.stop()
            else:
                logger.info("queue is empty — exiting")
            return

        job = jobs[0]
        logger.info("claiming issue #%d: %s", job.issue_number, job.title)
        claim(job)

        try:
            _run_benchmark(job.spec)
            sha = _commit_and_push_results(job.issue_number)
            post_result(job.issue_number, _build_summary(job.spec), sha)
            logger.info("issue #%d complete", job.issue_number)
        except Exception as exc:
            logger.exception("issue #%d failed", job.issue_number)
            post_failure(job.issue_number, str(exc))

        if once:
            return

        logger.info("polling again in %ds …", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run(once="--once" in sys.argv)
