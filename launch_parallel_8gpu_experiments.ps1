param(
    [Parameter(Mandatory = $true)] [string]$TopHost,
    [Parameter(Mandatory = $true)] [string]$RandomHost,
    [string]$TopPrefix = "",
    [string]$User = "ubuntu",
    [string]$Container = "quratingfiltered",
    [string]$TopRunName = "exp-top16-d26-8gpu",
    [string]$RandomRunName = "exp-rand16-d26-8gpu",
    [int]$Depth = 26,
    [int]$Workers = 16,
    [int]$NumGpus = 8,
    [int]$DeviceBatchSize = 32,
    [int]$TotalBatchSize = 524288,
    [string]$TokenCol = "token_count",
    [int]$TopMasterPort = 29500,
    [int]$RandomMasterPort = 29501,
    [string]$NanochatDir = "~/nanochat",
    [string]$HfDataset = "airtrain-ai/fineweb-edu-fortified",
    [string]$HfConfig = "",
    [string]$HfSplit = "train",
    [double]$HfRatio = 0.165,
    [int]$HfSeed = 42,
    [ValidateSet("prepare","skip")] [string]$HfPrepareMode = "prepare"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (($TopHost -eq $RandomHost) -and ($TopMasterPort -eq $RandomMasterPort)) {
    throw "TopHost and RandomHost are the same, but both runs use master port $TopMasterPort. Use different ports."
}

function Start-RemoteArm {
    param(
        [string]$Host,
        [string]$Arm,
        [string]$Prefix,
        [string]$RunName,
        [int]$MasterPort
    )

    $remoteCmd = @"
set -euo pipefail
cd $NanochatDir
chmod +x run_experiment_arm.sh train.sh
nohup bash run_experiment_arm.sh \
  --arm $Arm \
  --container $Container \
  --prefix '$Prefix' \
  --depth $Depth \
  --run-name $RunName \
  --workers $Workers \
  --num-gpus $NumGpus \
  --device-batch-size $DeviceBatchSize \
  --total-batch-size $TotalBatchSize \
  --token-col $TokenCol \
  --master-port $MasterPort \
  > ~/train_${RunName}.log 2>&1 &
echo \$!
"@

    $target = "$User@$Host"
    $pid = ssh $target $remoteCmd
    if (-not $pid) {
        throw "Failed to start remote arm on $target"
    }
    return $pid.Trim()
}

function Start-RemoteHfRandomArm {
    param(
        [string]$Host,
        [string]$RunName,
        [int]$MasterPort
    )

    $hfConfigArg = ""
    if ($HfConfig -ne "") {
        $hfConfigArg = "--config '$HfConfig'"
    }

    $remoteCmd = @"
set -euo pipefail
cd $NanochatDir
chmod +x run_experiment_hf_random_arm.sh train.sh
nohup bash run_experiment_hf_random_arm.sh \
  --dataset $HfDataset \
  $hfConfigArg \
  --split $HfSplit \
  --ratio $HfRatio \
  --seed $HfSeed \
  --prepare-mode $HfPrepareMode \
  --depth $Depth \
  --run-name $RunName \
  --num-gpus $NumGpus \
  --device-batch-size $DeviceBatchSize \
  --total-batch-size $TotalBatchSize \
  --token-col $TokenCol \
  --master-port $MasterPort \
  > ~/train_${RunName}.log 2>&1 &
echo \$!
"@

    $target = "$User@$Host"
    $pid = ssh $target $remoteCmd
    if (-not $pid) {
        throw "Failed to start remote HF random arm on $target"
    }
    return $pid.Trim()
}

# Ensure scripts are present on both hosts.
scp download_azure_data.py "$User@$TopHost`:~/nanochat/"
scp calc_subset_iterations.py "$User@$TopHost`:~/nanochat/"
scp patch_nanochat.py "$User@$TopHost`:~/nanochat/"
scp train.sh "$User@$TopHost`:~/nanochat/"
scp run_experiment_arm.sh "$User@$TopHost`:~/nanochat/"

scp download_azure_data.py "$User@$RandomHost`:~/nanochat/"  # harmless if unused by HF arm
scp calc_subset_iterations.py "$User@$RandomHost`:~/nanochat/"
scp prepare_hf_random_subset.py "$User@$RandomHost`:~/nanochat/"
scp patch_nanochat.py "$User@$RandomHost`:~/nanochat/"
scp train.sh "$User@$RandomHost`:~/nanochat/"
scp run_experiment_hf_random_arm.sh "$User@$RandomHost`:~/nanochat/"

$topPid = Start-RemoteArm -Host $TopHost -Arm "top16" -Prefix $TopPrefix -RunName $TopRunName -MasterPort $TopMasterPort
$randPid = Start-RemoteHfRandomArm -Host $RandomHost -RunName $RandomRunName -MasterPort $RandomMasterPort

Write-Output "Started top arm on $TopHost (PID: $topPid, log: ~/train_${TopRunName}.log)"
Write-Output "Started random HF arm on $RandomHost (PID: $randPid, log: ~/train_${RandomRunName}.log)"
Write-Output "Monitor:"
Write-Output "  ssh $User@$TopHost 'tail -f ~/train_${TopRunName}.log'"
Write-Output "  ssh $User@$RandomHost 'tail -f ~/train_${RandomRunName}.log'"
