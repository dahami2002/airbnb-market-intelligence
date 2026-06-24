Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

if (!(Test-Path ".venv")) {
    Write-Host "Virtual environment not found. Please run .\setup_env.ps1 first."
    exit 1
}

.\.venv\Scripts\Activate.ps1

python -m py_compile dashboard/app.py

streamlit run dashboard/app.py