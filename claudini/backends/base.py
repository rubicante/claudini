from __future__ import annotations

from abc import ABC, abstractmethod


class ComputeBackend(ABC):
    """
    Abstract interface for a remote GPU compute provider.

    Implementations handle provider-specific start/stop mechanics.
    The worker daemon uses this to shut itself down when the queue drains;
    the submit CLI uses it to start the worker before queuing a job.
    """

    name: str  # e.g. "runpod" — used in log messages and CLI output

    @abstractmethod
    def start(self) -> None:
        """Start or resume the compute resource. No-op if already running."""

    @abstractmethod
    def stop(self) -> None:
        """Stop or pause the compute resource."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return True if the compute resource is currently active."""
