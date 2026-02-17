# Nanochat Azure Training Pipeline

## Deliverables
- [x] `setup_droplet.sh` -- One-shot droplet provisioning
- [x] `download_azure_data.py` -- Azure → local text-only parquets
- [x] `patch_nanochat.py` -- Minimal nanochat modifications
- [x] `train.sh` -- Tokenizer + pretraining launch

## Deployment Steps
```
1. Create DO H100 droplet
2. scp setup_droplet.sh download_azure_data.py patch_nanochat.py train.sh .env to droplet
3. ssh into droplet
4. bash setup_droplet.sh
5. cd ~/nanochat && source .venv/bin/activate
6. python download_azure_data.py
7. python patch_nanochat.py
8. bash train.sh
```

## Verification Checklist
- [ ] After download: `ls ~/.cache/nanochat/base_data/ | wc -l` → 3250
- [ ] After download: spot-check columns are text-only
- [ ] After patch: MAX_SHARD updated in dataset.py
- [ ] During training: loss decreasing on stdout/wandb
