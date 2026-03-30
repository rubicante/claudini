# RunPod Setup Guide

This is a **one-time setup**. After completing it, you never SSH into the pod manually again.
The pod is started, used, and stopped automatically by the research pipeline.

---

## 1. Create a pod

1. Go to [RunPod](https://www.runpod.io/) and log in
2. Deploy a **Community Cloud** pod:
   - GPU: RTX 4090 (24 GB VRAM)
   - Template: **PyTorch 2.x (Ubuntu)**
   - Disk: 50 GB+ (model weights cache — Qwen 7B NF4 ≈ 6 GB)
3. Note the **pod ID** from the dashboard URL (looks like `abc123xyz`)

---

## 2. Configure SSH access (local machine)

Upload your SSH public key in RunPod Settings → SSH Public Keys, then add to `~/.ssh/config`:

```
Host runpod
    HostName ssh.runpod.io
    User <your-pod-id>
    Port <assigned-port>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
```

Test with `ssh runpod`. You only need SSH for the one-time bootstrap below.

---

## 3. Configure pod environment variables (RunPod console)

In the RunPod console, open your pod → **Edit** → **Environment Variables**. Add:

| Variable | Value |
|---|---|
| `CLAUDINI_BACKEND` | `runpod` |
| `RUNPOD_API_KEY` | Your RunPod API key (Settings → API Keys) |
| `RUNPOD_POD_ID` | Your pod ID (from the dashboard URL) |
| `GH_TOKEN` | GitHub personal access token (scopes: `repo`, `issues`) — git identity derived from this automatically |
| `CLAUDINI_REPO` | SSH or HTTPS clone URL of this repo |

These persist across pod stops/starts — set them once.

---

## 4. Configure the pod startup command (RunPod console)

In the RunPod console, open your pod → **Edit** → **Docker Command** (or **Startup Command**, depending on the template). Set it to:

```bash
cd ~/claudini && git pull --ff-only && uv run -m claudini.pipeline.worker
```

This runs automatically every time the pod boots. It pulls the latest code (including any new
optimizer you just committed) and starts the worker daemon, which processes the job queue and
stops the pod when done.

---

## 5. One-time bootstrap (SSH, done once)

Start the pod manually from the RunPod console, SSH in, and run:

```bash
GH_TOKEN=<your-token> \
CLAUDINI_REPO=<your-clone-url> \
GIT_EMAIL=<your-email> \
bash <(curl -fsSL https://raw.githubusercontent.com/<you>/claudini/main/scripts/bootstrap.sh)
```

Or copy `scripts/bootstrap.sh` to the pod and run it there. This installs `uv`, the `gh` CLI,
clones the repo, installs Python dependencies, and configures git identity.

Stop the pod when bootstrap completes. **You will not SSH in again.**

---

## 6. Configure your local machine

Add to `~/.zshrc` (or equivalent):

```bash
export CLAUDINI_BACKEND=runpod
export RUNPOD_API_KEY=<your RunPod API key>
export RUNPOD_POD_ID=<your pod ID>
```

---

## 7. Normal operation (fully automated)

From this point on, you never touch the pod. The research loop handles everything:

```bash
# Submit a job — starts the pod, queues the work
uv run -m claudini.pipeline.submit create \
  --method <method> --preset <preset> --sample 0 1 2 --seed 0 \
  --start-backend

# Watch progress and pull results when done
uv run -m claudini.pipeline.submit watch <issue-number>
```

**What happens automatically:**
1. Pod powers on (`podResume` via RunPod API)
2. Startup command runs: `git pull` → worker starts
3. Worker claims the job, runs the benchmark, commits results to git, posts summary to the issue
4. Worker stops the pod (`podStop` via RunPod API)
5. `watch` detects the closed issue and runs `git pull` locally

---

## Tips

- **Model cache** — HuggingFace downloads to `~/.cache/huggingface/`. First run of a new model is slow; subsequent runs use the cache (which persists across pod stops, but not terminations).
- **Disk space** — Check with `df -h` over SSH if needed. Keep 20 GB free for model weights.
- **Termination vs. stop** — A *stopped* pod retains its disk. A *terminated* pod does not. Never terminate unless you intend to redo the bootstrap.
- **Multiple jobs** — Queue as many jobs as you like before starting the pod. The worker processes them in submission order and stops only when the queue is empty.
