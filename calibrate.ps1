$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
conda run -n flybrain python -m app.calibrate --device cuda --flies 8

