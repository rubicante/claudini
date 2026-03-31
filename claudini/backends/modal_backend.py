"""
Modal backend — runs workers on Modal's GPU fleet.

No pod IDs. No host availability issues. Modal allocates GPUs globally.

Required env vars:
    (none beyond what's in the Modal secret)

Optional env vars:
    CLAUDINI_MODAL_GPU   — GPU type (default: A10G). Options: L4, A10G, A100-40GB, H100
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import ComputeBackend

# Persists the active call ID across Python processes within the same session.
# Best-effort: lost on machine reboot, but that's fine for our single-threaded loop.
_CALL_ID_FILE = Path("/tmp/claudini_modal_call_id.json")


class ModalBackend(ComputeBackend):
    name = "modal"

    def start(self) -> None:
        """Spawn the Modal worker function asynchronously."""
        import modal

        run_worker = modal.Function.from_name("claudini", "run_worker")
        call = run_worker.spawn(once=True)
        _CALL_ID_FILE.write_text(json.dumps({"call_id": call.object_id}))

    def stop(self) -> None:
        """Cancel a running Modal worker (rarely needed — Modal self-terminates)."""
        if not _CALL_ID_FILE.exists():
            return
        try:
            import modal

            data = json.loads(_CALL_ID_FILE.read_text())
            call = modal.FunctionCall.from_id(data["call_id"])
            call.cancel(terminate_containers=True)
        except Exception:
            pass
        finally:
            _CALL_ID_FILE.unlink(missing_ok=True)

    def is_running(self) -> bool:
        """Return True if a Modal worker call is currently active."""
        if not _CALL_ID_FILE.exists():
            return False
        try:
            import modal

            data = json.loads(_CALL_ID_FILE.read_text())
            call = modal.FunctionCall.from_id(data["call_id"])
            call.get(timeout=0)
            # get() returned — job is done
            _CALL_ID_FILE.unlink(missing_ok=True)
            return False
        except modal.exception.FunctionTimeoutError:
            # Timed out waiting — still running
            return True
        except Exception:
            # Any other error — assume not running
            _CALL_ID_FILE.unlink(missing_ok=True)
            return False
