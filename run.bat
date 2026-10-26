@echo off
title Eliteserien Fantasy Predictor ^& Transfer Optimizer
cd /d "%~dp0"

echo ==============================================================================
echo   Eliteserien Fantasy Predictor ^& Transfer Optimizer
echo ==============================================================================
echo.
echo [1/2] Oppdaterer data og nullstiller lokal cache...
python -m streamlit cache clear

echo.
echo [2/2] Starter applikasjonen med ferske API-data...
python -m streamlit run app.py

pause
