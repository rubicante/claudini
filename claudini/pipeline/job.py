from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml


def _current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class JobSpec:
    """Fully describes a single benchmark run submitted to the async pipeline."""

    method: str
    preset: str
    samples: list[int]
    seeds: list[int]
    max_flops: float | None = None
    notes: str = ""
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str = field(default_factory=_current_git_sha)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_issue_title(self) -> str:
        sample_str = ",".join(str(s) for s in self.samples)
        seed_str = ",".join(str(s) for s in self.seeds)
        return f"[job] {self.method} / {self.preset} / samples=[{sample_str}] seeds=[{seed_str}]"

    def to_issue_body(self) -> str:
        """Serialise to a GitHub issue body containing a fenced claudini-job YAML block."""
        data: dict = {
            "method": self.method,
            "preset": self.preset,
            "samples": self.samples,
            "seeds": self.seeds,
        }
        if self.max_flops is not None:
            data["max_flops"] = self.max_flops
        if self.notes:
            data["notes"] = self.notes
        data["submitted_at"] = self.submitted_at
        data["git_sha"] = self.git_sha

        yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False).strip()
        return f"```claudini-job\n{yaml_text}\n```\n"

    @classmethod
    def from_issue_body(cls, body: str) -> JobSpec:
        """Parse a JobSpec from the fenced block in a GitHub issue body."""
        match = re.search(r"```claudini-job\n(.*?)\n```", body, re.DOTALL)
        if not match:
            raise ValueError("No claudini-job block found in issue body")
        data = yaml.safe_load(match.group(1))
        return cls(
            method=data["method"],
            preset=data["preset"],
            samples=list(data["samples"]),
            seeds=list(data["seeds"]),
            max_flops=data.get("max_flops"),
            notes=data.get("notes", ""),
            submitted_at=data.get("submitted_at", ""),
            git_sha=data.get("git_sha", ""),
        )

    # ── execution ────────────────────────────────────────────────────────────

    def to_bench_args(self) -> list[str]:
        """Return the argument list to pass to  uv run -m claudini.run_bench."""
        args = [self.preset, "--method", self.method]
        args += ["--sample"] + [str(s) for s in self.samples]
        args += ["--seed"] + [str(s) for s in self.seeds]
        if self.max_flops is not None:
            args += ["--max-flops", str(self.max_flops)]
        return args
