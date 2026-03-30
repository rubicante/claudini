from __future__ import annotations

import os

from .base import ComputeBackend
from .runpod import RunPodBackend

_REGISTRY: dict[str, type[ComputeBackend]] = {
    "runpod": RunPodBackend,
}


def get_backend() -> ComputeBackend | None:
    """
    Instantiate the backend named by CLAUDINI_BACKEND.
    Returns None if the env var is not set, so callers can treat backend
    control as optional without special-casing.
    """
    name = os.environ.get("CLAUDINI_BACKEND", "").strip().lower()
    if not name:
        return None
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown backend {name!r}. Available: {list(_REGISTRY)}")
    return cls()
