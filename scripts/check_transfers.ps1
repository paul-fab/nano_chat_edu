param(
  [string]$SourceHost = "129.158.35.217",
  [string]$DestHost = "192.222.53.38",
  [string]$SourceDir = "~/.cache/nanochat/base_data",
  [string]$DestDir = "~/.cache/nanochat/hf_random_data_staging/base_data",
  [string]$LocalDir = "C:\Users\ather\data\hf_random_base_data_20260218\base_data"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RemoteStats {
  param(
    [Parameter(Mandatory = $true)][string]$RemoteHost,
    [Parameter(Mandatory = $true)][string]$Dir
  )
  $cmd = "ls -1 $Dir/shard_*.parquet 2>/dev/null | wc -l; du -sb $Dir 2>/dev/null | cut -f1"
  $out = ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "ubuntu@$RemoteHost" $cmd
  $lines = @($out -split "`r?`n" | Where-Object { $_ -ne "" })
  $files = 0
  $bytes = 0
  if ($lines.Count -ge 1) { [void][int]::TryParse($lines[0].Trim(), [ref]$files) }
  if ($lines.Count -ge 2) { [void][long]::TryParse($lines[1].Trim(), [ref]$bytes) }
  return [PSCustomObject]@{
    Files = $files
    Bytes = $bytes
  }
}

function Get-LocalStats {
  param([Parameter(Mandatory = $true)][string]$Dir)
  if (-not (Test-Path $Dir)) {
    return [PSCustomObject]@{ Files = 0; Bytes = 0L }
  }
  $files = (Get-ChildItem $Dir -Filter "shard_*.parquet" -File -ErrorAction SilentlyContinue).Count
  $bytes = (Get-ChildItem $Dir -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
  if ($null -eq $bytes) { $bytes = 0L }
  return [PSCustomObject]@{
    Files = [int]$files
    Bytes = [long]$bytes
  }
}

function Format-Bytes {
  param([long]$Bytes)
  if ($Bytes -ge 1TB) { return "{0:N2} TB" -f ($Bytes / 1TB) }
  if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
  if ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
  if ($Bytes -ge 1KB) { return "{0:N2} KB" -f ($Bytes / 1KB) }
  return "$Bytes B"
}

function Percent {
  param([double]$Numerator, [double]$Denominator)
  if ($Denominator -le 0) { return "n/a" }
  return ("{0:N1}%" -f (100.0 * $Numerator / $Denominator))
}

$source = Get-RemoteStats -RemoteHost $SourceHost -Dir $SourceDir
$local = Get-LocalStats -Dir $LocalDir
$dest = Get-RemoteStats -RemoteHost $DestHost -Dir $DestDir

$localFilePct = Percent -Numerator $local.Files -Denominator $source.Files
$localBytePct = Percent -Numerator $local.Bytes -Denominator $source.Bytes
$destFilePct = Percent -Numerator $dest.Files -Denominator $source.Files
$destBytePct = Percent -Numerator $dest.Bytes -Denominator $source.Bytes

$workerStatus = wsl.exe -e bash -lc "LOG=/mnt/c/Users/ather/data/hf_random_base_data_20260218/_transfer_logs; c=0; for p in `$LOG/worker_*.pid; do [ -f `"`$p`" ] || continue; pid=`$(cat `"`$p`"); if kill -0 `"`$pid`" 2>/dev/null; then c=`$((c+1)); fi; done; echo `$c"
$aliveWorkers = 0
[void][int]::TryParse(($workerStatus | Select-Object -First 1), [ref]$aliveWorkers)
$wslRsync = wsl.exe -e bash -lc "ps -ef | grep -E 'rsync .*base_data' | grep -v grep || true"

Write-Host "Transfer Progress"
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Host ""
Write-Host ("Source ({0})  files={1}  bytes={2} ({3})" -f $SourceHost, $source.Files, $source.Bytes, (Format-Bytes $source.Bytes))
Write-Host ("Local          files={0}  bytes={1} ({2})  vs source: files {3}, bytes {4}" -f $local.Files, $local.Bytes, (Format-Bytes $local.Bytes), $localFilePct, $localBytePct)
Write-Host ("8x Staging ({0}) files={1}  bytes={2} ({3})  vs source: files {4}, bytes {5}" -f $DestHost, $dest.Files, $dest.Bytes, (Format-Bytes $dest.Bytes), $destFilePct, $destBytePct)
Write-Host ""
Write-Host ("Active worker PIDs: {0}" -f $aliveWorkers)
if ($aliveWorkers -gt 0) {
  Write-Host $wslRsync
}
