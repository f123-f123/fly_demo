$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
conda run -n flybrain python -m app.server --device cuda

