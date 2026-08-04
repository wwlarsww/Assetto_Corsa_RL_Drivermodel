param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 5000,
    [string]$StorageDirectory = "runs\mlflow"
)

$ErrorActionPreference = "Stop"

$env:MLFLOW_DISABLE_TELEMETRY = "true"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StoragePath = Join-Path $ProjectRoot $StorageDirectory
$DatabasePath = Join-Path $StoragePath "mlflow.db"
$ArtifactPath = Join-Path $StoragePath "artifacts"

New-Item -ItemType Directory -Force -Path $StoragePath | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactPath | Out-Null

$DatabaseUriPath = $DatabasePath.Replace("\", "/")
$ArtifactUriPath = $ArtifactPath.Replace("\", "/")

Write-Host "Starting MLflow"
Write-Host "Dashboard: http://${HostAddress}:${Port}"
Write-Host "Database:  $DatabasePath"
Write-Host "Artifacts: $ArtifactPath"

python -m mlflow server `
    --host $HostAddress `
    --port $Port `
    --backend-store-uri "sqlite:///$DatabaseUriPath" `
    --artifacts-destination "file:///$ArtifactUriPath"
