$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path "$PSScriptRoot\..").Path
$WslPath = (wsl.exe wslpath -a $RepoPath).Trim()
Write-Host "Running target-environment verification in WSL: $WslPath"
wsl.exe bash -lc "cd '$WslPath' && bash scripts/setup_wsl.sh && source .venv/bin/activate && python scripts/smoke_qwen.py --device cuda --dtype fp16"
if ($LASTEXITCODE -ne 0) { throw "WSL verification failed with exit code $LASTEXITCODE" }
Write-Host "WSL verification passed. Start with: wsl.exe bash -lc \"cd '$WslPath' && bash scripts/run_wsl.sh\""
