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

If shards are staged under `~/.cache/nanochat/hf_random_data_staging/base_data`, run:
```bash
cd ~/nanochat
source .venv/bin/activate
bash run_experiment_hf_random_arm.sh \
  --prepare-mode skip \
  --data-dir ~/.cache/nanochat/hf_random_data_staging/base_data \
  --depth 26 \
  --run-name exp-rand16-d26-8gpu \
  --num-gpus 8 \
  --device-batch-size 32 \
  --total-batch-size 524288 \
  --master-port 29501
```

Safety note:
- `run_experiment_hf_random_arm.sh` now refuses to repoint `~/.cache/nanochat/base_data` while training is active.
- Override only if intentional with `--force-link-while-training`.

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

## Feb 18, 2026 Multi-GPU Postmortem (Top16 d26 8x H100)
Root cause that caused the apparent hang:
- DDP dataloader sharding assumed `num_row_groups >= world_size`.
- Our shards commonly have `1` row group per parquet file.
- Ranks `1..7` got no batches and looped over files forever, while rank `0` advanced into optimizer/distributed ops.
- Result was distributed desync and eventual NCCL timeout/stall pattern.

Fixes now applied in this repo:
- `patch_nanochat.py` now patches `nanochat/dataloader.py` with a fallback:
  - if row groups are sparse, shard by file index (`pq_idx % world_size`) instead of row-group index.
- `patch_nanochat.py` now accepts `--data-dir` and uses it for shard counting.
- `run_experiment_arm.sh` and `run_experiment_hf_random_arm.sh` now call:
```bash
python patch_nanochat.py --data-dir "$DATA_DIR"
```
- Both run scripts ensure `~/.cache/nanochat/base_data` points to `DATA_DIR` (symlink) when custom data dirs are used.
- `train.sh` now validates visible GPU count before `torchrun`.
- `train.sh` now parses `TRAIN_EXTRA_ARGS` into a bash array to avoid bad word splitting.
- `launch_parallel_8gpu_experiments.ps1` now uses `RandomMasterPort=29501` by default and hard-fails on same-host/same-port collisions.

Operational mistakes to avoid on the HF run:
- Do not copy `.sh` files from Windows without converting line endings.
```bash
cd ~/nanochat
sed -i 's/\r$//' *.sh
chmod +x *.sh
```
- Always re-run patching before launch:
```bash
cd ~/nanochat && source .venv/bin/activate
python patch_nanochat.py --data-dir ~/.cache/nanochat/base_data
```
- Confirm real progress after launch (not just process existence):
```bash
grep -n "step " ~/train_<run>.log | tail
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```
