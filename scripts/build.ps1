param([switch]$UseExistingEnvironment, [switch]$Gpu)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
}

if (-not $UseExistingEnvironment) {
    & "$PSScriptRoot\setup.ps1" -Gpu:$Gpu
}
$HasCudaRuntime = & '.venv\Scripts\python.exe' -c "import onnxruntime as ort; print(int('CUDAExecutionProvider' in ort.get_available_providers()))"
if ($LASTEXITCODE -ne 0) { throw 'ONNX Runtime inspection failed.' }
if ($Gpu -and $HasCudaRuntime -ne '1') { throw 'GPU build requires scripts\setup.ps1 -Gpu.' }
$GpuArguments = @()
if ($HasCudaRuntime -eq '1') {
    $GpuArguments = @('--collect-binaries', 'nvidia.cu13', '--collect-binaries', 'nvidia.cudnn')
}
& '.venv\Scripts\python.exe' scripts\build_icon.py
if ($LASTEXITCODE -ne 0) { throw 'Application icon generation failed.' }
$FfmpegExecutable = & '.venv\Scripts\python.exe' -c "from auto_mosaic.video_io import tool_path; print(tool_path('ffmpeg'))"
if ($LASTEXITCODE -ne 0) { throw 'The shared FFmpeg executable is missing.' }
$FfprobeExecutable = & '.venv\Scripts\python.exe' -c "from auto_mosaic.video_io import tool_path; print(tool_path('ffprobe'))"
if ($LASTEXITCODE -ne 0) { throw 'The shared ffprobe executable is missing.' }

$originalPath = $env:PATH
try {
    $env:PATH = "$([Environment]::SystemDirectory);$originalPath"
    & '.venv\Scripts\pyinstaller.exe' `
        --noconfirm `
        --clean `
        --windowed `
        --onedir `
        --name FY175AutoMosaic `
        --icon 'assets\app.ico' `
        --add-data 'models;models' `
        --add-data 'assets\app.ico;assets' `
        --add-binary "$FfmpegExecutable;ffmpeg" `
        --add-binary "$FfprobeExecutable;ffmpeg" `
        @GpuArguments `
        auto_mosaic\app.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:PATH = $originalPath
}

Copy-Item -LiteralPath 'README.md' -Destination 'dist\FY175AutoMosaic\README.md' -Force
Copy-Item -LiteralPath 'AGENT_INTERFACE.md' -Destination 'dist\FY175AutoMosaic\AGENT_INTERFACE.md' -Force
Copy-Item -LiteralPath 'LICENSE' -Destination 'dist\FY175AutoMosaic\LICENSE' -Force
Copy-Item -LiteralPath 'NOTICE' -Destination 'dist\FY175AutoMosaic\NOTICE' -Force
Copy-Item -LiteralPath 'THIRD_PARTY_NOTICES.md' -Destination 'dist\FY175AutoMosaic\THIRD_PARTY_NOTICES.md' -Force
Copy-Item -LiteralPath 'THIRD_PARTY_SOURCES.md' -Destination 'dist\FY175AutoMosaic\THIRD_PARTY_SOURCES.md' -Force
Copy-Item -LiteralPath 'MODEL_LICENSES.md' -Destination 'dist\FY175AutoMosaic\MODEL_LICENSES.md' -Force
Copy-Item -LiteralPath 'DISTRIBUTION.md' -Destination 'dist\FY175AutoMosaic\DISTRIBUTION.md' -Force
Copy-Item -LiteralPath 'LICENSES' -Destination 'dist\FY175AutoMosaic\LICENSES' -Recurse -Force
if ($HasCudaRuntime -eq '1') {
    & '.venv\Scripts\python.exe' scripts\copy_gpu_notices.py 'dist\FY175AutoMosaic'
    if ($LASTEXITCODE -ne 0) { throw 'GPU dependency notice collection failed.' }
}
& $FfmpegExecutable -version | Out-File -LiteralPath 'dist\FY175AutoMosaic\FFMPEG_BUILD.txt' -Encoding utf8
Get-FileHash -Algorithm SHA256 -LiteralPath $FfmpegExecutable, $FfprobeExecutable |
    Format-List | Out-File -LiteralPath 'dist\FY175AutoMosaic\FFMPEG_BUILD.txt' -Append -Encoding utf8

Write-Host "Build complete: $ProjectRoot\dist\FY175AutoMosaic\FY175AutoMosaic.exe"
