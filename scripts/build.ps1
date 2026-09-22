param([switch]$UseExistingEnvironment)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
}

if (-not $UseExistingEnvironment) {
    & '.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & '.venv\Scripts\python.exe' scripts\download_models.py
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
}
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
        --add-data 'models;models' `
        --add-binary "$FfmpegExecutable;ffmpeg" `
        --add-binary "$FfprobeExecutable;ffmpeg" `
        auto_mosaic\app.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:PATH = $originalPath
}

Copy-Item -LiteralPath 'README.md' -Destination 'dist\FY175AutoMosaic\README.md' -Force
Copy-Item -LiteralPath 'LICENSE' -Destination 'dist\FY175AutoMosaic\LICENSE' -Force
Copy-Item -LiteralPath 'NOTICE' -Destination 'dist\FY175AutoMosaic\NOTICE' -Force
Copy-Item -LiteralPath 'THIRD_PARTY_NOTICES.md' -Destination 'dist\FY175AutoMosaic\THIRD_PARTY_NOTICES.md' -Force
Copy-Item -LiteralPath 'MODEL_LICENSES.md' -Destination 'dist\FY175AutoMosaic\MODEL_LICENSES.md' -Force
Copy-Item -LiteralPath 'DISTRIBUTION.md' -Destination 'dist\FY175AutoMosaic\DISTRIBUTION.md' -Force
Copy-Item -LiteralPath 'LICENSES' -Destination 'dist\FY175AutoMosaic\LICENSES' -Recurse -Force
& $FfmpegExecutable -version | Out-File -LiteralPath 'dist\FY175AutoMosaic\FFMPEG_BUILD.txt' -Encoding utf8
Get-FileHash -Algorithm SHA256 -LiteralPath $FfmpegExecutable, $FfprobeExecutable |
    Format-List | Out-File -LiteralPath 'dist\FY175AutoMosaic\FFMPEG_BUILD.txt' -Append -Encoding utf8

Write-Host "Build complete: $ProjectRoot\dist\FY175AutoMosaic\FY175AutoMosaic.exe"
