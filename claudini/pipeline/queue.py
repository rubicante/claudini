from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from .job import JobSpec

# ── GitHub issue labels ───────────────────────────────────────────────────────

LABEL_QUEUED = "queued"
LABEL_RUNNING = "running"
LABEL_DONE = "done"
LABEL_FAILED = "failed"

_LABEL_DEFS = {
    LABEL_QUEUED: ("0075ca", "Job is waiting to be picked up by a worker"),
    LABEL_RUNNING: ("e4e669", "Job is currently executing on a remote worker"),
    LABEL_DONE: ("0e8a16", "Job completed successfully"),
    LABEL_FAILED: ("d73a4a", "Job failed — see issue comments for details"),
}


# ── internal gh wrapper ───────────────────────────────────────────────────────


def _gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


# ── public API ────────────────────────────────────────────────────────────────


@dataclass
class QueuedJob:
    issue_number: int
    spec: JobSpec
    title: str


def ensure_labels() -> None:
    """Create the four pipeline labels in the current repo if they don't exist."""
    existing = {label["name"] for label in json.loads(_gh("label", "list", "--json", "name"))}
    for name, (color, description) in _LABEL_DEFS.items():
        if name not in existing:
            _gh("label", "create", name, "--color", color, "--description", description)


def submit(spec: JobSpec) -> int:
    """Open a GitHub issue for this job. Returns the issue number."""
    url = _gh(
        "issue",
        "create",
        "--title",
        spec.to_issue_title(),
        "--body",
        spec.to_issue_body(),
        "--label",
        LABEL_QUEUED,
    )
    # gh outputs the full URL; issue number is the final path segment
    return int(url.rstrip("/").split("/")[-1])


def list_queued() -> list[QueuedJob]:
    """Return all queued jobs, oldest first (by issue number)."""
    raw = _gh("issue", "list", "--label", LABEL_QUEUED, "--json", "number,title,body", "--limit", "100")
    issues = sorted(json.loads(raw), key=lambda i: i["number"])
    jobs = []
    for issue in issues:
        try:
            jobs.append(QueuedJob(issue["number"], JobSpec.from_issue_body(issue["body"]), issue["title"]))
        except ValueError:
            pass  # issue body didn't contain a valid job spec — skip
    return jobs


def claim(job: QueuedJob) -> None:
    """Atomically move a job from queued → running and post a start comment."""
    _gh("issue", "edit", str(job.issue_number), "--remove-label", LABEL_QUEUED, "--add-label", LABEL_RUNNING)
    _gh("issue", "comment", str(job.issue_number), "--body", "Worker picked up this job and started execution.")


def post_result(issue_number: int, summary: str, git_sha: str) -> None:
    """Post a results summary comment and close the issue as done."""
    body = f"**Results committed** at `{git_sha}`\n\n{summary}"
    _gh("issue", "comment", str(issue_number), "--body", body)
    _gh("issue", "edit", str(issue_number), "--remove-label", LABEL_RUNNING, "--add-label", LABEL_DONE)
    _gh("issue", "close", str(issue_number))


def post_failure(issue_number: int, error: str) -> None:
    """Post an error comment and label the issue as failed (left open for inspection)."""
    body = f"**Job failed**\n\n```\n{error[:3000]}\n```"
    _gh("issue", "comment", str(issue_number), "--body", body)
    _gh("issue", "edit", str(issue_number), "--remove-label", LABEL_RUNNING, "--add-label", LABEL_FAILED)


def get_status(issue_number: int) -> dict:
    """Return raw GitHub issue data including labels and comments."""
    raw = _gh("issue", "view", str(issue_number), "--json", "number,title,labels,state,comments,url")
    return json.loads(raw)
