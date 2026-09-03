@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM === Stable FP32 MacBERT emotion training configuration ===
set "PYTHON=%~dp0.venv/Scripts/python.exe"
set "NOVEL_TRAIN=outputs/emotion-data/training-ready-v2/train.jsonl"
set "NOVEL_DEV=outputs/emotion-data/training-ready-v2/dev.jsonl"
set "NOVEL_TEST=outputs/emotion-data/training-ready-v2/test.jsonl"
set "BRIGHTER_TRAIN=outputs/emotion-data/brighter/brighter-chn-train.jsonl"
set "BRIGHTER_DEV=outputs/emotion-data/brighter/brighter-chn-dev.jsonl"
set "BRIGHTER_TEST=outputs/emotion-data/brighter/brighter-chn-test.jsonl"
set "OUTPUT_DIR=outputs/emotion-data/macbert-training-v2"
set "PREFLIGHT_REPORT=%OUTPUT_DIR%/preflight.json"
set "THRESHOLD_CALIBRATION=%OUTPUT_DIR%/threshold-calibration.json"
set "THRESHOLD_VALUE=%OUTPUT_DIR%/neutral-threshold.txt"
set "NOVEL_TEST_METRICS=%OUTPUT_DIR%/novel-test-metrics.json"
set "TEST_METRICS=%OUTPUT_DIR%/test-metrics.json"
set "EPOCHS=8"
set "BATCH_SIZE=12"
set "GRADIENT_ACCUMULATION=2"
set "LEARNING_RATE=2e-5"
set "HEAD_LEARNING_RATE=1e-4"
set "INTENSITY_LOSS_WEIGHT=0.7"
set "NEUTRAL_LOSS_WEIGHT=3.0"
set "MAX_LENGTH=256"
set "SEED=20260829"
set "CHECKPOINT_STEPS=200"
set "KEEP_CHECKPOINTS=2"
set "MIN_FREE_VRAM_GIB=9"
set "MIN_FREE_DISK_GIB=10"
set "FINAL_EXIT=0"

REM === Reproducible local/offline environment ===
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "TOKENIZERS_PARALLELISM=false"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"

echo ============================================
echo   Readest MacBERT Emotion Training - FP32
echo ============================================
echo.
echo  Output:       %OUTPUT_DIR%
echo  Epochs:       %EPOCHS%
echo  Batch Size:   %BATCH_SIZE%
echo  Grad Accum:   %GRADIENT_ACCUMULATION%
echo  Encoder LR:   %LEARNING_RATE%
echo  Head LR:      %HEAD_LEARNING_RATE%
echo  Gate Loss:    %INTENSITY_LOSS_WEIGHT%
echo  Neutral Loss: %NEUTRAL_LOSS_WEIGHT%
echo  Max Length:   %MAX_LENGTH%
echo  Seed:         %SEED%
echo  Resume:       auto
echo  Checkpoints:  every %CHECKPOINT_STEPS% optimizer steps, keep %KEEP_CHECKPOINTS%
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found: %PYTHON%
    echo         Run: uv sync --extra emotion_train
    set "FINAL_EXIT=2"
    goto :end
)

for %%F in ("%NOVEL_TRAIN%" "%NOVEL_DEV%" "%NOVEL_TEST%" "%BRIGHTER_TRAIN%" "%BRIGHTER_DEV%" "%BRIGHTER_TEST%") do (
    if not exist "%%~F" (
        echo [ERROR] Training input not found: %%~F
        set "FINAL_EXIT=2"
        goto :end
    )
)

echo [1/5] Environment and data preflight...
echo.
"%PYTHON%" -m indextts.emotion.cli preflight-training ^
    --train "%NOVEL_TRAIN%" ^
    --train "%BRIGHTER_TRAIN%" ^
    --dev "%NOVEL_DEV%" ^
    --dev "%BRIGHTER_DEV%" ^
    --test "%NOVEL_TEST%" ^
    --test "%BRIGHTER_TEST%" ^
    --output "%PREFLIGHT_REPORT%" ^
    --max-length %MAX_LENGTH% ^
    --verify-base-weights ^
    --require-cuda ^
    --minimum-free-vram-gib %MIN_FREE_VRAM_GIB% ^
    --minimum-free-disk-gib %MIN_FREE_DISK_GIB%
if errorlevel 1 (
    echo.
    echo [ERROR] Preflight failed. Close GPU-heavy applications or repair the environment.
    set "FINAL_EXIT=2"
    goto :end
)

echo.
echo [2/5] Training with automatic resume...
echo.
"%PYTHON%" -m indextts.emotion.cli train ^
    --train "%NOVEL_TRAIN%" ^
    --train "%BRIGHTER_TRAIN%" ^
    --dev "%NOVEL_DEV%" ^
    --dev "%BRIGHTER_DEV%" ^
    --output "%OUTPUT_DIR%" ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --gradient-accumulation %GRADIENT_ACCUMULATION% ^
    --learning-rate %LEARNING_RATE% ^
    --head-learning-rate %HEAD_LEARNING_RATE% ^
    --intensity-loss-weight %INTENSITY_LOSS_WEIGHT% ^
    --neutral-loss-weight %NEUTRAL_LOSS_WEIGHT% ^
    --max-length %MAX_LENGTH% ^
    --seed %SEED% ^
    --device cuda ^
    --resume auto ^
    --checkpoint-steps %CHECKPOINT_STEPS% ^
    --keep-checkpoints %KEEP_CHECKPOINTS% ^
    --progress
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" (
    echo.
    if "%TRAIN_EXIT%"=="130" (
        echo [INFO] Training interrupted safely. Run this BAT again to resume.
    ) else (
        echo [ERROR] Training failed with exit code %TRAIN_EXIT%.
    )
    set "FINAL_EXIT=%TRAIN_EXIT%"
    goto :end
)

if not exist "%OUTPUT_DIR%/training-complete.json" (
    echo [ERROR] Training returned success without a completion marker.
    set "FINAL_EXIT=2"
    goto :end
)
if not exist "%OUTPUT_DIR%/best/model.safetensors" (
    echo [ERROR] Best checkpoint is missing.
    set "FINAL_EXIT=2"
    goto :end
)

echo.
echo [3/5] Calibrating the neutral gate on dev data only...
echo.
"%PYTHON%" -m indextts.emotion.cli calibrate-threshold ^
    --checkpoint "%OUTPUT_DIR%/best" ^
    --data "%NOVEL_DEV%" ^
    --data "%BRIGHTER_DEV%" ^
    --output "%THRESHOLD_CALIBRATION%" ^
    --threshold-output "%THRESHOLD_VALUE%" ^
    --minimum-active-recall 0.8 ^
    --batch-size 32 ^
    --max-length %MAX_LENGTH% ^
    --device cuda ^
    --progress
if errorlevel 1 (
    echo.
    echo [ERROR] Dev-only neutral threshold calibration failed.
    set "FINAL_EXIT=2"
    goto :end
)

set /p NEUTRAL_THRESHOLD=<"%THRESHOLD_VALUE%"
if not defined NEUTRAL_THRESHOLD (
    echo [ERROR] Calibrated neutral threshold is missing.
    set "FINAL_EXIT=2"
    goto :end
)
echo  Calibrated neutral threshold: %NEUTRAL_THRESHOLD%

echo.
echo [4/5] Evaluating the held-out novel test set...
echo.
"%PYTHON%" -m indextts.emotion.cli evaluate ^
    --checkpoint "%OUTPUT_DIR%/best" ^
    --data "%NOVEL_TEST%" ^
    --output "%NOVEL_TEST_METRICS%" ^
    --batch-size 32 ^
    --max-length %MAX_LENGTH% ^
    --neutral-threshold %NEUTRAL_THRESHOLD% ^
    --device cuda ^
    --progress
if errorlevel 1 (
    echo.
    echo [ERROR] Novel test evaluation failed.
    set "FINAL_EXIT=2"
    goto :end
)

echo.
echo [5/5] Evaluating the combined novel and BRIGHTER test sets...
echo.
"%PYTHON%" -m indextts.emotion.cli evaluate ^
    --checkpoint "%OUTPUT_DIR%/best" ^
    --data "%NOVEL_TEST%" ^
    --data "%BRIGHTER_TEST%" ^
    --output "%TEST_METRICS%" ^
    --batch-size 32 ^
    --max-length %MAX_LENGTH% ^
    --neutral-threshold %NEUTRAL_THRESHOLD% ^
    --device cuda ^
    --progress
set "EVAL_EXIT=%ERRORLEVEL%"
if not "%EVAL_EXIT%"=="0" (
    echo.
    echo [ERROR] Evaluation failed with exit code %EVAL_EXIT%.
    set "FINAL_EXIT=%EVAL_EXIT%"
    goto :end
)

echo.
echo ============================================
echo   Training and evaluation completed
echo   Best model:   %OUTPUT_DIR%/best
echo   Train report: %OUTPUT_DIR%/training-report.json
echo   Calibration:  %THRESHOLD_CALIBRATION%
echo   Novel test:   %NOVEL_TEST_METRICS%
echo   Test metrics: %TEST_METRICS%
echo ============================================

:end
echo.
if not "%FINAL_EXIT%"=="0" echo Process stopped with exit code %FINAL_EXIT%.
exit /b %FINAL_EXIT%
