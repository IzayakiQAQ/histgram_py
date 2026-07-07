$ErrorActionPreference = "Stop"

$target = Get-Date "2026-06-17 21:05:00"
$now = Get-Date
if ($now -lt $target) {
    $seconds = [Math]::Ceiling(($target - $now).TotalSeconds)
    Write-Output ("[{0}] Waiting {1} seconds until {2}" -f $now.ToString("yyyy-MM-dd HH:mm:ss"), $seconds, $target.ToString("yyyy-MM-dd HH:mm:ss"))
    Start-Sleep -Seconds $seconds
}

$workdir = "E:\lzy\crosscorrelation compensation\repositories\histgram_py"
Set-Location -LiteralPath $workdir

Write-Output ("[{0}] Starting 0km 5nm single-peak batch" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
python -u .\run_singlepeak_batch.py --job-file .\singlepeak_0km_5nm_20260616_12_34.json
Write-Output ("[{0}] Finished 0km 5nm single-peak batch" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
