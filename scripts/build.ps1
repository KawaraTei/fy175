$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
}

& '.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.venv\Scripts\python.exe' scripts\download_models.py

& '.venv\Scripts\pyinstaller.exe' `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name AutoMosaic `
    --add-data 'models;models' `
    auto_mosaic\app.py

Copy-Item -LiteralPath 'README.md' -Destination 'dist\AutoMosaic\README.md' -Force
Copy-Item -LiteralPath 'LICENSE' -Destination 'dist\AutoMosaic\LICENSE' -Force
Copy-Item -LiteralPath 'NOTICE' -Destination 'dist\AutoMosaic\NOTICE' -Force
Copy-Item -LiteralPath 'THIRD_PARTY_NOTICES.md' -Destination 'dist\AutoMosaic\THIRD_PARTY_NOTICES.md' -Force
Copy-Item -LiteralPath 'MODEL_LICENSES.md' -Destination 'dist\AutoMosaic\MODEL_LICENSES.md' -Force
Copy-Item -LiteralPath 'DISTRIBUTION.md' -Destination 'dist\AutoMosaic\DISTRIBUTION.md' -Force
Copy-Item -LiteralPath 'LICENSES' -Destination 'dist\AutoMosaic\LICENSES' -Recurse -Force

Write-Host "Build complete: $ProjectRoot\dist\AutoMosaic\AutoMosaic.exe"
