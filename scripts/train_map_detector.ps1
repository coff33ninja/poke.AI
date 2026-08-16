# Trains/resumes the map detector on the 8-class Pokemon overworld dataset.
#
# Usage (from repo root):
#   powershell -File scripts/train_map_detector.ps1                        # fresh start from COCO
#   powershell -File scripts/train_map_detector.ps1 -Resume training/snapshots_real/resnet50_csv_02.h5 -InitialEpoch 2 -Epochs 12
#
# Params:
#   -Resume         path to a snapshot .h5 to continue from (default: none -> COCO init)
#   -InitialEpoch   epoch number to resume numbering at (default: 0)
#   -Epochs         number of epochs to run (default: 12)
#   -GpuIndex       CUDA_VISIBLE_DEVICES to use (default: 1 = the GTX 1060, NOT the display GPU)

param(
    [string]$Resume = "",
    [int]$InitialEpoch = 0,
    [int]$Epochs = 12,
    [int]$GpuIndex = 1
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$args_ = New-Object System.Collections.ArrayList
[void]$args_.Add("-m");        [void]$args_.Add("keras_retinanet.bin.train")
[void]$args_.Add("--backbone"); [void]$args_.Add("resnet50")
if ($Resume) {
    [void]$args_.Add("--snapshot"); [void]$args_.Add($Resume)
    [void]$args_.Add("--initial-epoch"); [void]$args_.Add("$InitialEpoch")
}
[void]$args_.Add("--batch-size");   [void]$args_.Add("1")
[void]$args_.Add("--steps");        [void]$args_.Add("700")
[void]$args_.Add("--epochs");       [void]$args_.Add("$Epochs")
[void]$args_.Add("--lr");           [void]$args_.Add("1e-4")
[void]$args_.Add("--reduce-lr-patience"); [void]$args_.Add("3")
[void]$args_.Add("--reduce-lr-factor");   [void]$args_.Add("0.5")
[void]$args_.Add("--gpu");          [void]$args_.Add("$GpuIndex")
[void]$args_.Add("--workers");      [void]$args_.Add("0")
[void]$args_.Add("--image-min-side"); [void]$args_.Add("400")
[void]$args_.Add("--image-max-side"); [void]$args_.Add("400")
[void]$args_.Add("--random-transform")
[void]$args_.Add("--compute-val-loss")
[void]$args_.Add("--no-evaluation")
[void]$args_.Add("--snapshot-path"); [void]$args_.Add("training/snapshots_real")
[void]$args_.Add("csv")
[void]$args_.Add("training/data/real/train_labels.csv")
[void]$args_.Add("training/data/real/classes.csv")
[void]$args_.Add("--val-annotations"); [void]$args_.Add("training/data/real/val_labels.csv")

$env:TF_GPU_ALLOCATOR = "cuda_malloc_async"
$py = "$root\.venv\Scripts\python.exe"
& $py @args_ *> "$root\training\train_real.log"
exit $LASTEXITCODE
