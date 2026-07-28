$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = if ($env:PHYSIOATLAS_VENV) { $env:PHYSIOATLAS_VENV } else { ".venv-physioatlas" }

if (-not (Test-Path $VenvDir)) {
    & $PythonBin -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r requirements-physioatlas.txt
& $VenvPython -m pip install -e . --no-deps --no-build-isolation

& $VenvPython -m compileall -q physioatlas
& $VenvPython -m physioatlas list-adapters | Out-Null
& $VenvPython -m physioatlas doctor --repo .
& $VenvPython -m pytest -o "addopts=" tests/physioatlas -q
& $VenvPython -m physioatlas smoke-test --output outputs/physioatlas/innerloop-smoke
& $VenvPython -m physioatlas verify-run outputs/physioatlas/innerloop-smoke/run
& $VenvPython -m physioatlas fault-test --output outputs/physioatlas/fault-test

Write-Host "PhysioAtlas InnerLoop bootstrap passed."
Write-Host "Report: outputs/physioatlas/innerloop-smoke/smoke-report.json"
