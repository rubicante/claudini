"""
Local CLI for managing benchmark jobs and compute backends.

Job management:
    uv run -m claudini.pipeline.submit create --method gcg --preset random_valid --sample 0 1 2
    uv run -m claudini.pipeline.submit list
    uv run -m claudini.pipeline.submit status 42
    uv run -m claudini.pipeline.submit watch 42

Backend lifecycle (requires CLAUDINI_BACKEND + provider env vars):
    uv run -m claudini.pipeline.submit backend start
    uv run -m claudini.pipeline.submit backend stop
    uv run -m claudini.pipeline.submit backend status
"""

from __future__ import annotations

import json as _json
import subprocess
import time
from pathlib import Path

import typer

from claudini.backends import get_backend
from .job import JobSpec
from .queue import ensure_labels, get_status, list_queued, submit

REPO_ROOT = Path(__file__).parents[2]

app = typer.Typer(help="Manage Claudini benchmark jobs and compute backends.", no_args_is_help=True)
backend_app = typer.Typer(help="Control the configured compute backend (start / stop / status).", no_args_is_help=True)
app.add_typer(backend_app, name="backend")


# ── job commands ──────────────────────────────────────────────────────────────


@app.command()
def create(
    method: str = typer.Option(..., help="Optimizer method name, e.g. gcg"),
    preset: str = typer.Option(..., help="Config preset name, e.g. random_valid"),
    sample: list[int] = typer.Option([0], help="Sample index (repeat for multiple)"),
    seed: list[int] = typer.Option([0], help="Random seed (repeat for multiple)"),
    max_flops: float = typer.Option(None, help="FLOP budget override"),
    notes: str = typer.Option("", help="Human-readable notes attached to the issue"),
    start_backend: bool = typer.Option(False, "--start-backend", help="Start the compute backend if not running"),
    json: bool = typer.Option(False, "--json", help="Print result as JSON (for scripting)"),
) -> None:
    """Submit a new benchmark job to the queue."""
    ensure_labels()
    spec = JobSpec(
        method=method,
        preset=preset,
        samples=sample,
        seeds=seed,
        max_flops=max_flops or None,
        notes=notes,
    )
    issue_number = submit(spec)

    backend_started = False
    if start_backend:
        backend = get_backend()
        if backend is None:
            typer.echo("Warning: --start-backend set but CLAUDINI_BACKEND is not configured.", err=True)
        elif backend.is_running():
            typer.echo(f"Backend {backend.name!r} is already running.")
        else:
            backend.start()
            backend_started = True
            typer.echo(f"Backend {backend.name!r} started.")

    if json:
        typer.echo(_json.dumps({"issue": issue_number, "backend_started": backend_started}))
    else:
        typer.echo(f"Submitted: issue #{issue_number}  —  {spec.to_issue_title()}")


@app.command("list")
def list_jobs() -> None:
    """List all jobs currently in the queue."""
    jobs = list_queued()
    if not jobs:
        typer.echo("Queue is empty.")
        return
    for job in jobs:
        typer.echo(f"  #{job.issue_number:<6}  {job.title}")


@app.command()
def status(issue_number: int = typer.Argument(..., help="GitHub issue number")) -> None:
    """Show the current status of a job."""
    data = get_status(issue_number)
    labels = [label["name"] for label in data.get("labels", [])]
    state = data.get("state", "unknown")
    url = data.get("url", "")
    typer.echo(f"Issue #{issue_number}  state={state}  labels={labels}")
    if url:
        typer.echo(f"  {url}")
    comments = data.get("comments", [])
    if comments:
        last = comments[-1]
        typer.echo(f"\nLatest comment:\n{last.get('body', '').strip()}")


@app.command()
def watch(
    issue_number: int = typer.Argument(..., help="GitHub issue number to watch"),
    interval: int = typer.Option(15, help="Poll interval in seconds"),
) -> None:
    """
    Stream issue comments until the job closes, then pull results.

    Exits with code 0 on success (done label) or 1 on failure (failed label).
    """
    typer.echo(f"Watching issue #{issue_number} — polling every {interval}s (Ctrl-C to stop) …")
    seen: set[str] = set()

    while True:
        data = get_status(issue_number)
        labels = {label["name"] for label in data.get("labels", [])}

        for comment in data.get("comments", []):
            cid = str(comment.get("id") or comment.get("url", ""))
            if cid not in seen:
                seen.add(cid)
                body = comment.get("body", "").strip()
                if body:
                    typer.echo(f"\n── comment ──────────────────────────────────\n{body}")

        if data.get("state", "").upper() == "CLOSED":
            if "done" in labels:
                typer.echo(f"\nJob #{issue_number} complete. Pulling results …")
                subprocess.run(["git", "pull"], cwd=REPO_ROOT, check=True)
                typer.echo("Done.")
                raise typer.Exit(0)
            else:
                typer.echo(f"\nJob #{issue_number} closed with labels: {labels}")
                raise typer.Exit(1)

        time.sleep(interval)


# ── backend commands ──────────────────────────────────────────────────────────


def _require_backend():
    backend = get_backend()
    if backend is None:
        typer.echo("Error: CLAUDINI_BACKEND is not set.", err=True)
        raise typer.Exit(1)
    return backend


@backend_app.command("start")
def backend_start() -> None:
    """Start (or resume) the configured compute backend."""
    backend = _require_backend()
    if backend.is_running():
        typer.echo(f"Backend {backend.name!r} is already running.")
    else:
        backend.start()
        typer.echo(f"Backend {backend.name!r} started.")


@backend_app.command("stop")
def backend_stop() -> None:
    """Stop the configured compute backend."""
    backend = _require_backend()
    backend.stop()
    typer.echo(f"Backend {backend.name!r} stopped.")


@backend_app.command("status")
def backend_status() -> None:
    """Show whether the configured compute backend is running."""
    backend = _require_backend()
    running = backend.is_running()
    typer.echo(f"Backend {backend.name!r}: {'running' if running else 'stopped'}")


if __name__ == "__main__":
    app()
