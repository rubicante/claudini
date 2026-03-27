#!/usr/bin/env bash
# Run a benchmark command on the RunPod instance via SSH.
# Usage: ./scripts/run_remote.sh "uv run -m claudini.run_bench random_train --method gcg --sample 0"
#
# The command runs under nohup so it survives SSH disconnection.
# Output is logged to ~/claudini/run.log on the pod.

set -euo pipefail

REMOTE_HOST="${RUNPOD_HOST:-runpod}"
REMOTE_DIR="${RUNPOD_DIR:-~/claudini}"
LOG_FILE="run.log"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"<command>\"" >&2
    echo "Example: $0 \"uv run -m claudini.run_bench random_train --method gcg --sample 0\"" >&2
    exit 1
fi

CMD="$1"

echo "Running on ${REMOTE_HOST}:${REMOTE_DIR}"
echo "Command: ${CMD}"
echo "Log: ${REMOTE_DIR}/${LOG_FILE}"
echo ""

ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && nohup bash -c '${CMD}' > ${LOG_FILE} 2>&1 &"

echo "Job launched in background. Monitor with:"
echo "  ssh ${REMOTE_HOST} \"tail -f ${REMOTE_DIR}/${LOG_FILE}\""
