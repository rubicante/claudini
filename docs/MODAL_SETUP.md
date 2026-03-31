# Modal Setup Guide

Modal is the primary compute backend: serverless GPU, no pod IDs, and a persistent HuggingFace
model cache on a Modal Volume. Setup is a one-time task.

---

## 1. Create a Modal account

Sign up at [modal.com](https://modal.com). Free tier is enough for testing; you'll use credits for
real runs.

---

## 2. Install the Modal CLI and authenticate

```bash
uv add modal          # or: pip install modal
modal setup           # opens browser, links your machine to your workspace
```

---

## 3. Create the `claudini-secrets` secret

In the Modal dashboard go to **Secrets → Create secret → Custom**, or run:

```bash
modal secret create claudini-secrets \
  GH_TOKEN=<github-personal-access-token> \
  CLAUDINI_REPO=<https-clone-url-of-this-repo>
```

| Key | Value |
|---|---|
| `GH_TOKEN` | GitHub personal access token — scopes: `repo`, `issues`. Git identity is derived from this automatically. |
| `CLAUDINI_REPO` | HTTPS clone URL, e.g. `https://github.com/you/claudini.git` |

These are the only secrets needed. RunPod keys are not required for the Modal backend.

---

## 4. Deploy the app

```bash
modal deploy claudini/modal_app.py
```

This bakes Python dependencies into a container image and registers the `claudini` app. The HF
model cache volume (`claudini-hf-cache`) is created automatically on first deploy.

Re-deploy only when `pyproject.toml` dependencies change. New optimizer code is pulled from git
at runtime, so code changes do not require a redeploy.

---

## 5. Configure your local machine

Add to `~/.zshrc` (or equivalent):

```bash
export CLAUDINI_BACKEND=modal
```

The Modal backend needs no other variables locally — it uses the secrets stored in your Modal
workspace.

Optionally, override the GPU type (default: `A10G`):

```bash
export CLAUDINI_MODAL_GPU=A100   # for larger models or faster runs
```

---

## 6. Normal operation

```bash
# Submit jobs — Modal starts a GPU container automatically
uv run -m claudini.pipeline.submit create \
  --method gcg --preset random_valid --sample 0 1 2 --seed 0 \
  --start-backend

# Watch progress and pull results when done
uv run -m claudini.pipeline.submit watch <issue-number>
```

**What happens automatically:**
1. `--start-backend` calls `modal run` to launch a serverless GPU container
2. Container clones/pulls the repo, installs deps, configures git identity from `GH_TOKEN`
3. Worker daemon claims jobs from the GitHub Issues queue, runs benchmarks, commits results
4. Container exits when the queue empties; Modal stops billing
5. `watch` detects the closed issue and runs `git pull` locally

---

## Tips

- **GPU choice** — A10G (24 GB) fits Qwen2.5-7B NF4 comfortably. Use A100 for bf16 or larger
  models.
- **Cold start** — First run of a new model downloads weights to the persistent volume
  (`claudini-hf-cache`); subsequent runs reuse the cache instantly.
- **Volume management** — View and manage the volume at `modal.com/storage`. Qwen2.5-7B NF4
  takes ~6 GB; keep 20 GB free.
- **Logs** — Live container logs are visible in the Modal dashboard or via `modal app logs claudini`.
- **Test locally** — `modal run claudini/modal_app.py` runs one job on a real GPU without
  touching the submit CLI.
