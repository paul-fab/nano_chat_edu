param(
    [Parameter(Mandatory = $true)] [string]$SourceHost,
    [Parameter(Mandatory = $true)] [string]$DestHost,
    [string]$User = "ubuntu",
    [string]$KeyPath = "$HOME\.ssh\id_ed25519",
    [int]$Workers = 8,
    [string]$SourceDir = "/home/ubuntu/.cache/nanochat/base_data",
    [string]$DestDir = "/home/ubuntu/.cache/nanochat/base_data"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$source = "$User@$SourceHost"
$dest = "$User@$DestHost"

if (-not (Test-Path $KeyPath)) {
    throw "Private key not found: $KeyPath"
}
if (-not (Test-Path "$KeyPath.pub")) {
    throw "Public key not found: $KeyPath.pub"
}

Write-Output "Copying SSH key to destination host..."
scp -o StrictHostKeyChecking=accept-new $KeyPath "$KeyPath.pub" "$dest`:~/.ssh/"

$remote = @"
set -euo pipefail
mkdir -p ~/.ssh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/$(Split-Path -Leaf $KeyPath)
chmod 644 ~/.ssh/$(Split-Path -Leaf "$KeyPath.pub")
cat >> ~/.ssh/config <<'EOF'
Host src-transfer
  HostName $SourceHost
  User $User
  IdentityFile ~/.ssh/$(Split-Path -Leaf $KeyPath)
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config
mkdir -p $DestDir
ssh -o BatchMode=yes src-transfer 'echo connected'
ssh src-transfer 'cd $SourceDir && ls shard_*.parquet' | xargs -P $Workers -I {} rsync -a --partial --inplace src-transfer:$SourceDir/{} $DestDir/
ls $DestDir/shard_*.parquet | wc -l
du -sh $DestDir
"@

Write-Output "Starting parallel rsync transfer ($Workers workers)..."
ssh -o StrictHostKeyChecking=accept-new $dest $remote

Write-Output "Transfer complete."
