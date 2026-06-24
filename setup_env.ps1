Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

python -m pip install `
numpy==1.26.4 `
pandas==2.2.2 `
scipy==1.11.4 `
duckdb==0.10.3 `
pyarrow==15.0.2 `
fastparquet==2024.2.0 `
streamlit==1.35.0 `
plotly==5.22.0 `
matplotlib==3.8.4 `
seaborn==0.13.2 `
scikit-learn==1.4.2 `
statsmodels==0.14.2 `
pyyaml==6.0.1 `
requests==2.32.3 `
tqdm==4.66.4 `
pytest==8.2.2

python -c "import numpy, pandas, scipy, duckdb, pyarrow, fastparquet, streamlit, plotly, sklearn, statsmodels; print('Environment ready')"