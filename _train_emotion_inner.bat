@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Versioned full-train candidate route. Nothing here promotes a model.
set "PYTHON=%~dp0.venv/Scripts/python.exe"
set "PROFILE=tools/emotion-data/one-click-profile.json"

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found: %PYTHON%
    echo         Run: uv sync --extra emotion_train
    exit /b 2
)
if not exist "%PROFILE%" (
    echo [ERROR] Training profile not found: %PROFILE%
    exit /b 2
)

echo ============================================
echo   Readest Emotion One-Click Candidate Train
echo ============================================
echo  Profile: %PROFILE%
echo  Resume:  automatic when data, code and method match
echo  Publish: never automatic
echo.

"%PYTHON%" -X utf8 tools/emotion-data/run_one_click.py --profile "%PROFILE%"
set "FINAL_EXIT=%ERRORLEVEL%"
echo.
if not "%FINAL_EXIT%"=="0" (
    echo [ERROR] Candidate pipeline stopped with exit code %FINAL_EXIT%.
    echo         Fix the cause, then run this BAT again to resume.
) else (
    echo [OK] Candidate training and frozen evaluation finished.
    echo      See the versioned output directory from the profile.
)
exit /b %FINAL_EXIT%
