# Fix DuckDB Version Mismatch Error
# This script clears the old DuckDB file and rebuilds it

Write-Host "Fixing DuckDB version mismatch error..." -ForegroundColor Yellow

# Step 1: Clear old DuckDB files
Write-Host "`n[Step 1/3] Clearing old DuckDB files..."
Remove-Item "data\warehouse\*.duckdb*" -Force -ErrorAction SilentlyContinue
Write-Host "✓ Old DuckDB files cleared" -ForegroundColor Green

# Step 2: Activate virtual environment
Write-Host "`n[Step 2/3] Activating virtual environment..."
if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "✗ Virtual environment not found. Run .\setup_env.ps1 first" -ForegroundColor Red
    exit 1
}

# Step 3: Rebuild DuckDB warehouse only
Write-Host "`n[Step 3/3] Rebuilding DuckDB warehouse..."
Write-Host "This will take 1-2 minutes..." -ForegroundColor Cyan

python -m src.airbnb_pipeline.pipeline --city bangkok --stages model

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ DuckDB warehouse rebuilt successfully!" -ForegroundColor Green
    Write-Host "`nYou can now run the dashboard:" -ForegroundColor Cyan
    Write-Host "  .\run_dashboard.ps1" -ForegroundColor White
} else {
    Write-Host "`n✗ Failed to rebuild warehouse" -ForegroundColor Red
    Write-Host "`nThe dashboard will still work using Parquet files as fallback." -ForegroundColor Yellow
    Write-Host "You can run the dashboard with:" -ForegroundColor Cyan
    Write-Host "  .\run_dashboard.ps1" -ForegroundColor White
}
