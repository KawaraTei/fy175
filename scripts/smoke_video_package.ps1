$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot 'dist\FY175AutoMosaic\FY175AutoMosaic.exe'

# MainWindowHandle excludes hidden windows. Enumerate only this test process's
# windows so the smoke check can keep the desktop undisturbed.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class VideoSmokeWindow {
    private delegate bool Callback(IntPtr window, IntPtr argument);
    [DllImport("user32.dll")] private static extern bool EnumWindows(Callback callback, IntPtr argument);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] private static extern int GetWindowText(IntPtr window, StringBuilder text, int count);
    [DllImport("user32.dll")] private static extern bool PostMessage(IntPtr window, uint message, IntPtr wParam, IntPtr lParam);
    public static string FindAndClose(int processId) {
        string title = "";
        EnumWindows((window, argument) => {
            uint owner;
            GetWindowThreadProcessId(window, out owner);
            if (owner != processId) return true;
            var buffer = new StringBuilder(512);
            GetWindowText(window, buffer, buffer.Capacity);
            if (!buffer.ToString().Contains("動画処理")) return true;
            title = buffer.ToString();
            PostMessage(window, 0x0010, IntPtr.Zero, IntPtr.Zero);
            return false;
        }, IntPtr.Zero);
        return title;
    }
}
'@

$ModelSmoke = Start-Process -FilePath $Executable -ArgumentList '--smoke-test' -WindowStyle Hidden -Wait -PassThru
if ($ModelSmoke.ExitCode -ne 0) { throw "Packaged model smoke failed: $($ModelSmoke.ExitCode)" }
Write-Output 'PACKAGED_MODEL_SMOKE_OK'

$VideoProcess = Start-Process -FilePath $Executable -ArgumentList '--video' -WindowStyle Hidden -PassThru
try {
    if (-not $VideoProcess.WaitForInputIdle(15000)) { throw 'Video window did not become ready.' }
    $Deadline = [DateTime]::UtcNow.AddSeconds(10)
    do {
        $VideoProcess.Refresh()
        if ($VideoProcess.HasExited) { throw 'Video process exited before opening its window.' }
        $VideoTitle = [VideoSmokeWindow]::FindAndClose($VideoProcess.Id)
        if ($VideoTitle) { break }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $Deadline)
    if (-not $VideoTitle) { throw 'Video window title was not found.' }
    Write-Output "PACKAGED_VIDEO_WINDOW_OK $VideoTitle"
    if (-not $VideoProcess.WaitForExit(10000)) { throw 'Video window did not close normally.' }
    if ($VideoProcess.ExitCode -ne 0) { throw "Video window failed: $($VideoProcess.ExitCode)" }
}
finally {
    if (-not $VideoProcess.HasExited) { $VideoProcess.Kill() }
    $VideoProcess.Dispose()
}

$PackagedProbe = Join-Path $ProjectRoot 'dist\FY175AutoMosaic\_internal\ffmpeg\ffprobe.exe'
& $PackagedProbe -v error -show_entries format=duration -of json (Join-Path $ProjectRoot '.codex-qa\2026-09-20\timeline-fixture.mp4')
if ($LASTEXITCODE -ne 0) { throw 'Packaged ffprobe failed.' }
Write-Output 'PACKAGED_VIDEO_TOOLS_OK'
