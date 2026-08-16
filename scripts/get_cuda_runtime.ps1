<#
.SYNOPSIS
    Downloads the CUDA 11.2 + cuDNN 8.1 runtime DLLs into the project-local
    cuda_runtime/ folder.

.DESCRIPTION
    TensorFlow 2.10 (the last TF with native-Windows GPU support) requires the
    CUDA 11.2 and cuDNN 8.1 runtime DLLs. Instead of installing the CUDA
    toolkit into the main system, this script pulls the cudatoolkit 11.2.2
    package from conda-forge and the official NVIDIA cuDNN 8.1.0.77 for CUDA
    11.2 redistributable, then extracts just the DLLs into cuda_runtime/bin.
    sitecustomize.py prepends that folder to PATH at interpreter startup, so
    TF finds the DLLs with zero system changes.

    NOTE: cuDNN MUST be the official NVIDIA build. The conda-forge cudnn
    binaries crash TensorFlow 2.10 with exit code 0xC0000409 on the first
    GPU convolution, even though they report the correct version.

    Idempotent: skips the download if cuda_runtime/bin already has the DLLs.
#>

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$targetDir = Join-Path $root 'cuda_runtime'
$binDir = Join-Path $targetDir 'bin'
$workDir = Join-Path $env:TEMP 'poke_ai_cuda_runtime'

$cudatoolkit = @{ Name = 'cudatoolkit'; File = 'win-64/cudatoolkit-11.2.2-h933977f_8.tar.bz2'; Url = 'https://conda.anaconda.org/conda-forge/win-64/cudatoolkit-11.2.2-h933977f_8.tar.bz2' }
$cudnn = @{ Name = 'cudnn'; File = 'cudnn-11.2-windows-x64-v8.1.0.77.zip'; Url = 'https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.0/cudnn-11.2-windows-x64-v8.1.0.77.zip' }

function Get-Python {
    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) { return $venvPy }
    return 'python'
}

$alreadyPopulated = (Test-Path $binDir) -and ((Get-ChildItem -LiteralPath $binDir -Filter '*.dll' | Measure-Object).Count -ge 8)
if (-not $alreadyPopulated) {
    New-Item -ItemType Directory -Path $targetDir, $workDir -Force | Out-Null

    # --- cudatoolkit 11.2.2 (conda-forge tar.bz2) ---
    $pkg = $cudatoolkit
    $archive = Join-Path $workDir $pkg.File
    if (-not (Test-Path $archive)) {
        Write-Host "Downloading $($pkg.Name) ..."
        $archiveParent = Split-Path -Parent $archive
        New-Item -ItemType Directory -Path $archiveParent -Force | Out-Null
        Invoke-WebRequest -Uri $pkg.Url -OutFile $archive
    }
    $extractDir = Join-Path $workDir $pkg.Name
    $dllDir = Join-Path $extractDir 'Library\bin'
    $alreadyExtracted = (Test-Path $dllDir) -and ((Get-ChildItem -LiteralPath $dllDir -Filter '*.dll' -File | Measure-Object).Count -gt 0)
    if (-not $alreadyExtracted) {
        if (Test-Path $extractDir) { Remove-Item -LiteralPath $extractDir -Recurse -Force }
        New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
        Write-Host "Extracting $($pkg.Name) ..."
        & (Get-Python) -c "import sys, tarfile; tarfile.open(sys.argv[1], 'r:bz2').extractall(sys.argv[2])" $archive $extractDir
        if ($LASTEXITCODE -ne 0) { throw "Failed to extract $($pkg.Name)" }
    } else {
        Write-Host "$($pkg.Name) already extracted, skipping."
    }

    # --- cuDNN 8.1.0.77 (official NVIDIA zip; the conda-forge build crashes TF) ---
    $pkg = $cudnn
    $archive = Join-Path $workDir $pkg.File
    if (-not (Test-Path $archive)) {
        Write-Host "Downloading $($pkg.Name) (665 MB) ..."
        Invoke-WebRequest -Uri $pkg.Url -OutFile $archive
    }
    $extractDir = Join-Path $workDir $pkg.Name
    $dllDir = Join-Path $extractDir 'cuda\bin'
    $alreadyExtracted = (Test-Path $dllDir) -and ((Get-ChildItem -LiteralPath $dllDir -Filter '*.dll' -File | Measure-Object).Count -gt 0)
    if (-not $alreadyExtracted) {
        if (Test-Path $extractDir) { Remove-Item -LiteralPath $extractDir -Recurse -Force }
        Write-Host "Extracting $($pkg.Name) ..."
        Expand-Archive -LiteralPath $archive -DestinationPath $extractDir -Force
    } else {
        Write-Host "$($pkg.Name) already extracted, skipping."
    }

    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    foreach ($pkg in @($cudatoolkit, $cudnn)) {
        $dllDir = if ($pkg.Name -eq 'cudnn') { Join-Path $workDir "$($pkg.Name)\cuda\bin" } else { Join-Path $workDir "$($pkg.Name)\Library\bin" }
        if (Test-Path $dllDir) {
            Get-ChildItem -LiteralPath $dllDir -Filter '*.dll' -File | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $binDir -Force
            }
        }
    }

    $count = (Get-ChildItem -LiteralPath $binDir -Filter '*.dll' | Measure-Object).Count
    Write-Host "Done. Copied $count DLLs into cuda_runtime/bin."
} else {
    Write-Host "cuda_runtime/bin already populated, nothing to do."
}

# Install sitecustomize.py into the venv so it auto-loads at interpreter
# startup (Python only imports sitecustomize from site-packages/PYTHONPATH,
# never from the script's own directory) and prepends cuda_runtime/bin to
# PATH before TensorFlow is imported.
$siteDir = Join-Path $root '.venv\Lib\site-packages'
if (Test-Path $siteDir) {
    Copy-Item -LiteralPath (Join-Path $root 'sitecustomize.py') -Destination (Join-Path $siteDir 'sitecustomize.py') -Force
    Write-Host "Installed sitecustomize.py into $siteDir"
} else {
    Write-Warning "No venv found at $siteDir - re-run after 'uv sync' and re-run this script."
}
