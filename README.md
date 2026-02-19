# FabRating — Education Model Training

Pretrain a language model from scratch on quality-filtered education data using [nanochat](https://github.com/karpathy/nanochat) on a single GPU.

## Data

- **Source**: 250GB of filtered education data in Azure Blob Storage (`quratingscoressa/quratingfiltered`)
- **Size**: 3,250 parquet files, ~52M documents
- **Quality scores per document**: pedagogical structure, factual accuracy, lesson engagement, education level
- **Processing**: Embeddings stripped (~75% size reduction), sorted by composite quality score so the best documents are read first during training

## Architecture

```
Azure Blob Storage (248GB raw)
    ↓ download + strip embeddings
GPU instance local disk (~97GB with metadata)
    ↓ sort by quality (pedagogical + factual + engagement)
Sorted text-only shards (~49GB)
    ↓ nanochat dataloader
Training (1x H100/GH200)
```

## Files

| File | Purpose |
|------|---------|
| `setup_droplet.sh` | One-shot GPU instance provisioning (deps, nanochat, venv) |
| `download_azure_data.py` | Download from Azure, strip embeddings, sort by quality, re-shard |
| `patch_nanochat.py` | Patch nanochat for local shards and stable TF32 settings |
| `train.sh` | Train tokenizer + launch pretraining |
| `inspect_blob_store.py` | Utility to inspect Azure blob storage contents |
| `calc_subset_iterations.py` | Compute full-subset token budget and training iterations |
| `run_experiment_arm.sh` | Run one experiment arm from a prebuilt Azure subset prefix |
| `run_experiment_hf_random_arm.sh` | Run HF random arm (`prepare` or `skip` sharding mode) |
| `transfer_shards_parallel.ps1` | Copy shards host-to-host with SSH key + parallel rsync |
| `EXPERIMENT_PLAYBOOK.md` | Recommended end-to-end workflow and troubleshooting |
| `RUN_TRACKING.md` | Tracking sheet for active runs, split runs, and headline metrics |

## Quick Start

```bash
# 1. Provision a GPU instance (H100 or GH200)
# 2. Copy files to instance
scp setup_droplet.sh download_azure_data.py patch_nanochat.py train.sh .env user@<IP>:~/

# 3. SSH in and run
ssh user@<IP>
bash setup_droplet.sh              # install everything (~5 min)
cd ~/nanochat && source .venv/bin/activate
python download_azure_data.py -w 16  # download + sort (~20 min)
python patch_nanochat.py             # patch nanochat
bash train.sh                        # train d20 model (~3 hrs on H100)
bash train.sh 26                     # or d26 (~8 hrs)
```

## Top-vs-Random Experiment (Prebuilt Azure Subsets)

If your `top16.5` and `random16.5` subsets are already prepared in Azure, run one arm per GPU instance:

```bash
cd ~/nanochat && source .venv/bin/activate

bash run_experiment_arm.sh \
  --arm top16 \
  --container <azure-container> \
  --prefix <blob-prefix-for-top16.5/> \
  --download-mode bulk \
  --depth 20 \
  --run-name exp-top16-d20 \
  --workers 16
```

```bash
cd ~/nanochat && source .venv/bin/activate

bash run_experiment_arm.sh \
  --arm rand16 \
  --container <azure-container> \
  --prefix <blob-prefix-for-random16.5/> \
  --depth 20 \
  --run-name exp-rand16-d20 \
  --workers 16
```

Notes:
- This downloads only blobs under `--prefix` and does **not** re-rank/re-sample data.
- `--download-mode bulk` uses `azcopy` first (faster on large Azure subsets) and falls back to Python downloader if unavailable.
- If `token_count` exists in shards, training iterations are set to consume the full subset tokens.
- Both arms can be pinned to the same `--total-batch-size` for fair comparison.
- For cost efficiency, prepare HF random shards on a CPU/storage node, then run GPU training with `--prepare-mode skip`.
- For retries on the same machine, use `--skip-download` to reuse existing `shard_*.parquet` and avoid re-downloading.
- `patch_nanochat.py` also patches TF32 config to avoid torch.compile/inductor TF32 API mismatch crashes.
- If shards do not contain `token_count` (e.g. only `score,text`), set `--num-iterations` explicitly.
- On A100, Flash Attention 3 is unavailable, and `window_pattern='SSSL'` is significantly slower than on H100.

## Host-To-Host Shard Transfer

```powershell
powershell -File .\transfer_shards_parallel.ps1 `
  -SourceHost 192.222.58.132 `
  -DestHost 129.153.201.219 `
  -Workers 8 `
  -KeyPath "$HOME\.ssh\id_ed25519"
```

## Quality Sorting

Documents are sorted by a composite score averaging three metrics:
- **Pedagogical structure** — how well-structured as teaching material
- **Factual accuracy** — correctness of information
- **Lesson engagement** — how engaging as educational content

Nanochat reads shards sequentially and stops at the compute-optimal token count (10.5x model params). Since the best documents are in the first shards, the model trains on the highest-quality slice of the data.

## Training Time Estimates (1x H100/GH200)

| Depth | Params | Tokens | Time | Cost @ $2/hr |
|-------|--------|--------|------|-------------|
| d20 | ~124M | ~1.3B | ~3 hr | ~$6 |
| d24 | ~178M | ~1.9B | ~6 hr | ~$12 |
| d26 | ~215M | ~2.3B | ~8 hr | ~$16 |
| d32 | ~335M | ~3.5B | ~18 hr | ~$36 |

## Monitoring

Training logs to [Weights & Biases](https://wandb.ai) automatically. Set up with:

```bash
wandb login  # paste API key from wandb.ai/authorize
```

Or set a key in your shell and `.env`:

```bash
export WANDB_API_KEY=wandb_v1_xxx
```

If login fails, regenerate a key and relogin:

```bash
wandb login --relogin wandb_v1_xxx
```

## Troubleshooting

- `No API key configured` or `WANDB_API_KEY invalid`:
```bash
wandb login --relogin wandb_v1_xxx
```
- `set: pipefail\r: invalid option name` after copying scripts from Windows:
```bash
sed -i 's/\r$//' *.sh
chmod +x *.sh
```
- 8-GPU run appears alive but never prints `step ...` (often dataloader rank desync on low-row-group parquet shards):
```bash
python patch_nanochat.py --data-dir ~/.cache/nanochat/base_data
```
- To retry quickly without re-downloading:
```bash
bash run_experiment_arm.sh --skip-download ...
```

## Based On

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [FabRating](https://arxiv.org/abs/2402.09739) (ICML 2024) — quality rating methodology
