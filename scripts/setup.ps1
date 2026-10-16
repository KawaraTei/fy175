param([switch]$Gpu)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

# The CPU and GPU wheels own the same Python module: never install both.
# NudeNet was only used for its model and depends on the CPU runtime.
$ConflictingRuntime = if ($Gpu) { 'onnxruntime' } else { 'onnxruntime-gpu' }
& '.venv\Scripts\python.exe' -m pip uninstall -y $ConflictingRuntime nudenet
if ($LASTEXITCODE -ne 0) { throw 'Conflicting dependency removal failed.' }
$Requirements = if ($Gpu) { 'requirements-gpu.txt' } else { 'requirements.txt' }
& '.venv\Scripts\python.exe' -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.venv\Scripts\python.exe' scripts\download_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
