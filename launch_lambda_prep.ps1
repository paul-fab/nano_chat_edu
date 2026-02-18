param(
    [Parameter(Mandatory = $true)] [string]$Ip,
    [string]$User = "ubuntu",
    [Parameter(Mandatory = $true)] [string]$HfToken,
    [string]$Dataset = "airtrain-ai/fineweb-edu-fortified",
    [string]$Split = "train",
    [double]$Ratio = 0.165,
    [int]$Seed = 42,
    [int]$RowsPerShard = 14000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$target = "$User@$Ip"

Write-Output "Creating bootstrap directory on $target..."
ssh -o StrictHostKeyChecking=accept-new $target "mkdir -p ~/bootstrap"

Write-Output "Copying setup + prep files..."
scp setup_droplet.sh $target`:~/bootstrap/
scp prepare_hf_random_subset.py $target`:~/bootstrap/
scp .env $target`:~/bootstrap/

Write-Output "Running setup on remote host..."
ssh $target "cd ~/bootstrap && bash setup_droplet.sh"

Write-Output "Syncing latest prep script into ~/nanochat..."
scp prepare_hf_random_subset.py $target`:~/nanochat/

Write-Output "Starting HF random subset prep..."
$remoteCmd = @"
set -euo pipefail
cd ~/nanochat
source .venv/bin/activate
nohup env HF_TOKEN=$HfToken HUGGINGFACE_HUB_TOKEN=$HfToken \
  python prepare_hf_random_subset.py \
    --dataset $Dataset \
    --split $Split \
    --ratio $Ratio \
    --seed $Seed \
    --rows-per-shard $RowsPerShard \
  > ~/hf_random_prep.log 2>&1 &
echo \$!
"@

$pid = ssh $target $remoteCmd
Write-Output "Started prep on $target (PID: $($pid.Trim()))."
Write-Output "Monitor with:"
Write-Output "  ssh $target 'tail -f ~/hf_random_prep.log'"
Write-Output "  ssh $target 'ls ~/.cache/nanochat/base_data/shard_*.parquet 2>/dev/null | wc -l'"
