"""
RunPod backend — controls a dedicated RunPod pod via the GraphQL API.

Required environment variables:
    RUNPOD_API_KEY   — RunPod API key (Settings → API Keys in the RunPod console)
    RUNPOD_POD_ID    — ID of the pod to control (visible in the RunPod console URL)

Optional:
    RUNPOD_GPU_COUNT — number of GPUs to request when resuming (default: 1)
"""

from __future__ import annotations

import json
import os
import urllib.request

from .base import ComputeBackend

_API_URL = "https://api.runpod.io/graphql"


class RunPodBackend(ComputeBackend):
    name = "runpod"

    def __init__(self) -> None:
        self._api_key = os.environ["RUNPOD_API_KEY"]
        self._pod_id = os.environ["RUNPOD_POD_ID"]
        self._gpu_count = int(os.environ.get("RUNPOD_GPU_COUNT", "1"))

    # ── public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Resume a stopped pod. No-op if it is already running."""
        if self.is_running():
            return
        self._mutate(
            """
            mutation PodResume($input: PodResumeInput!) {
                podResume(input: $input) { id desiredStatus }
            }
            """,
            {"input": {"podId": self._pod_id, "gpuCount": self._gpu_count}},
        )

    def stop(self) -> None:
        """Stop a running pod."""
        self._mutate(
            """
            mutation PodStop($input: PodStopInput!) {
                podStop(input: $input) { id desiredStatus }
            }
            """,
            {"input": {"podId": self._pod_id}},
        )

    def is_running(self) -> bool:
        data = self._query(
            """
            query Pod($input: PodFilter!) {
                pod(input: $input) { id desiredStatus }
            }
            """,
            {"input": {"podId": self._pod_id}},
        )
        status = (data.get("pod") or {}).get("desiredStatus", "")
        return status == "RUNNING"

    # ── internal helpers ──────────────────────────────────────────────────────

    def _request(self, gql: str, variables: dict) -> dict:
        payload = json.dumps({"query": gql.strip(), "variables": variables}).encode()
        req = urllib.request.Request(
            f"{_API_URL}?api_key={self._api_key}",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "claudini/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        if "errors" in body:
            raise RuntimeError(f"RunPod API error: {body['errors']}")
        return body.get("data", {})

    def _query(self, gql: str, variables: dict) -> dict:
        return self._request(gql, variables)

    def _mutate(self, gql: str, variables: dict) -> dict:
        return self._request(gql, variables)
