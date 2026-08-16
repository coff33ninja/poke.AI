# Converts the newest training snapshot into the deployable map_detector.h5
# the app loads (object_detection/keras-retinanet/inference_graphs/map_detector.h5).
# Requires a finished/interrupted training run in training/snapshots_real/.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$snapshot = Get-ChildItem "$root\training\snapshots_real" -Filter *.h5 |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $snapshot) {
    Write-Error "No snapshots found in training/snapshots_real/. Run the training command first (see training/README.md)."
}

$outDir = "$root\object_detection\keras-retinanet\inference_graphs"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outPath = "$outDir\map_detector.h5"

Write-Host "Converting $($snapshot.Name) -> $outPath"
uv run python -m keras_retinanet.bin.convert_model $snapshot.FullName $outPath --backbone resnet50
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  uv run python -c ""import models; models.load_model(r'$outPath', backbone_name='resnet50')"""
