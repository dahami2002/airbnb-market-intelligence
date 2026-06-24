Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

if (!(Test-Path ".venv")) {
    Write-Host "Virtual environment not found. Please run .\setup_env.ps1 first."
    exit 1
}

.\.venv\Scripts\Activate.ps1

python -c "import numpy, pandas, scipy, duckdb, pyarrow, fastparquet, streamlit, plotly, sklearn, statsmodels; print('Packages OK')"

Remove-Item -Force "data\warehouse\airbnb_bangkok.duckdb" -ErrorAction SilentlyContinue
Remove-Item -Force "data\warehouse\airbnb_bangkok.duckdb.wal" -ErrorAction SilentlyContinue
Remove-Item -Force "data\warehouse\airbnb_bangkok.duckdb.tmp" -ErrorAction SilentlyContinue

python -m py_compile dashboard/app.py

python -m src.airbnb_pipeline.pipeline --city bangkok

pytest