# RunPod Setup Guide

## 1. Create a Pod

1. Go to [RunPod](https://www.runpod.io/) and create an account
2. Deploy a **Community Cloud** pod with:
   - GPU: RTX 4090 (24GB VRAM)
   - Template: PyTorch 2.x (Ubuntu)
   - Disk: 50GB+ (for model weights cache)
3. Note the pod ID and SSH connection info from the dashboard

## 2. SSH Access

RunPod provides SSH via a proxied port. Add to `~/.ssh/config`:

```
Host runpod
    HostName ssh.runpod.io
    User <your-pod-id>
    Port <assigned-port>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
```

Upload your SSH public key in RunPod settings first. Then:

```bash
ssh runpod
```

## 3. Initial Pod Setup

```bash
# On the pod:
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

git clone https://github.com/romovpa/claudini.git
cd claudini
uv sync
```

## 4. Sync Local Changes to Pod

For iterating on code locally and running on the pod:

```bash
# From local machine (repo root):
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'results' \
    --exclude '.git' --exclude '*.pyc' \
    ./ runpod:~/claudini/
```

## 5. Run Benchmarks Remotely

Use the `scripts/run_remote.sh` helper:

```bash
# Smoke test
./scripts/run_remote.sh "uv run -m claudini.run_bench random_train --method gcg --sample 0 --max-flops 1e13 --results-dir /tmp/smoke"

# Full baseline run
./scripts/run_remote.sh "uv run -m claudini.run_bench random_train --method gcg --max-flops 1e15"
```

For long runs, the script uses `nohup` so the job survives SSH disconnection. Check progress with:

```bash
ssh runpod "tail -f ~/claudini/run.log"
```

## 6. Pull Results Back

```bash
# Pull all results
rsync -avz runpod:~/claudini/results/ ./results/

# Pull specific method
rsync -avz runpod:~/claudini/results/gcg/ ./results/gcg/
```

## 7. Tips

- **Model cache**: HuggingFace downloads to `~/.cache/huggingface/`. First run of a new model will be slow.
- **tmux**: Use `tmux` on the pod for persistent sessions: `ssh runpod -t "tmux attach || tmux new"`
- **Disk space**: Check with `df -h`. Model weights can be large (Qwen 7B ~14GB).
- **Stop the pod** when not in use to avoid charges. Results on pod disk persist across stops (not across termination).
