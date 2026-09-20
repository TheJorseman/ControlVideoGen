# Crea el venv e instala dependencias (PyTorch CUDA para RTX serie 50: cu128).
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
Write-Host "`nListo. Ejecuta .\run.ps1 para lanzar la app." -ForegroundColor Green
