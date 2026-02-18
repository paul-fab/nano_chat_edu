# Experiment Playbook

## Goal
Run two comparable training arms:
- `top16.5` from prebuilt Azure subset
- `random16.5` from HF (`airtrain-ai/fineweb-edu-fortified`)

## Recommended Architecture
Do **not** run HF sharding on an 8x GPU node.

Use:
1. **CPU/storage node** for `prepare_hf_random_subset.py`
2. **GPU node A** for top arm training
3. **GPU node B** for random arm training (from prebuilt shards, `--prepare-mode skip`)

## One-Time Host Setup
On each machine:
```bash
cd ~/bootstrap
bash setup_droplet.sh
```

If shell scripts were copied from Windows, normalize line endings:
```bash
cd ~/nanochat
sed -i 's/\r$//' *.sh
chmod +x *.sh
```

## Random Subset Prep (CPU Node)
```bash
cd ~/nanochat
source .venv/bin/activate
python prepare_hf_random_subset.py \
  --dataset airtrain-ai/fineweb-edu-fortified \
  --split train \
  --ratio 0.165 \
  --seed 42 \
  --data-dir ~/.cache/nanochat/base_data \
  --rows-per-shard 14000
```

Then transfer prepared `shard_*.parquet` to GPU random node (`~/.cache/nanochat/base_data`).

## Top Arm (GPU Node A, Azure prebuilt subset)
```bash
cd ~/nanochat
source .venv/bin/activate
bash run_experiment_arm.sh \
  --arm top16 \
  --container quratingfiltered \
  --prefix <top-prefix-or-empty> \
  --download-mode bulk \
  --depth 26 \
  --run-name exp-top16-d26-8gpu \
  --num-gpus 8 \
  --device-batch-size 32 \
  --total-batch-size 524288
```

## Random Arm (GPU Node B, prebuilt random shards)
```bash
cd ~/nanochat
source .venv/bin/activate
bash run_experiment_hf_random_arm.sh \
  --prepare-mode skip \
  --data-dir ~/.cache/nanochat/base_data \
  --depth 26 \
  --run-name exp-rand16-d26-8gpu \
  --num-gpus 8 \
  --device-batch-size 32 \
  --total-batch-size 524288
```

## Parallel Launcher
`launch_parallel_8gpu_experiments.ps1` supports:
- `-HfPrepareMode prepare` (default)
- `-HfPrepareMode skip` (use prebuilt random shards on random host)

Example:
```powershell
powershell -File .\launch_parallel_8gpu_experiments.ps1 `
  -TopHost 192.0.2.10 `
  -RandomHost 192.0.2.11 `
  -TopPrefix "" `
  -HfPrepareMode skip
```

## Common Failure Checks
- Retry without re-downloading:
```bash
bash run_experiment_arm.sh --skip-download ...
```
- W&B key errors (`No API key configured` or invalid key length):
```bash
cd ~/nanochat && source .venv/bin/activate
wandb login --relogin wandb_v1_xxx
```
- `ModuleNotFoundError: rustbpe` or `psutil`:
```bash
cd ~/nanochat && source .venv/bin/activate
uv pip install rustbpe psutil fastapi ipykernel kernels tabulate uvicorn zstandard
```
- TF32/Inductor crash mentioning mixed legacy/new TF32 APIs:
```bash
cd ~/nanochat && source .venv/bin/activate
python patch_nanochat.py
```
- HF warning about unauthenticated requests:
```bash
echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_TOKEN=hf_xxx' >> ~/.bashrc
source ~/.bashrc
```
- Verify training started:
```bash
ps -ef | grep -E 'torchrun|scripts.base_train' | grep -v grep
nvidia-smi
```
