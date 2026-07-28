$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python -m pip install -r requirements-physioatlas.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m physioatlas doctor --repo .
python -m pytest -o "addopts=" tests/physioatlas -q
python -m physioatlas household-smoke-test --output outputs/physioatlas/household-smoke
python -m physioatlas household-fault-test --output outputs/physioatlas/household-fault
Write-Host "`nResearch hub demo:"
Write-Host "  python -m physioatlas serve-research-hub --state outputs/physioatlas/household-smoke/research-hub-state.json --port 8770"
