$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m compileall -q physioatlas
python -m physioatlas list-adapters | Out-Null
python -m physioatlas doctor --repo .
python -m pytest -o "addopts=" tests/physioatlas -q
python -m physioatlas smoke-test --output outputs/physioatlas/innerloop-smoke
python -m physioatlas verify-run outputs/physioatlas/innerloop-smoke/run
python -m physioatlas household-smoke-test --output outputs/physioatlas/household-smoke
python -m physioatlas household-fault-test --output outputs/physioatlas/household-fault
python -m physioatlas fault-test --output outputs/physioatlas/fault-test
