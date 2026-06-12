@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run connectome_analysis.py for every edge weight.
rem Required env vars: GPICKLE_DIR, BASE_OUTPUT_DIR
rem Optional env var: BASE_CACHE_DIR (default: .\data\null_cache)
rem
rem Usage:
rem   set GPICKLE_DIR=.\data\takahashi\na\cmp-v3.2.0
rem   set BASE_OUTPUT_DIR=.\results
rem   run_all_weights.bat [extra args passed to each run]
rem
rem Each weight gets its own --output-dir and --cache-dir to avoid collisions.

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if not defined PYTHON set "PYTHON=python"
if not defined GPICKLE_DIR (
    echo Error: GPICKLE_DIR is not set. Set it before running the script. 1>&2
    exit /b 1
)
if not defined BASE_OUTPUT_DIR (
    echo Error: BASE_OUTPUT_DIR is not set. Set it before running the script. 1>&2
    exit /b 1
)
if not defined BASE_CACHE_DIR set "BASE_CACHE_DIR=.\data\null_cache"

set "WEIGHTS=fiber_density fiber_number fiber_length FA"

for %%W in (%WEIGHTS%) do (
    echo ========================================
    echo  Edge weight: %%W
    echo ========================================
    "%PYTHON%" "%SCRIPT_DIR%\code\connectome_analysis.py" ^
        "%GPICKLE_DIR%" ^
        --edge-weight "%%W" ^
        --output-dir "%BASE_OUTPUT_DIR%\%%W" ^
        --cache-dir "%BASE_CACHE_DIR%\%%W" ^
        %*
    if errorlevel 1 exit /b 1
    echo.
)

echo All weights complete.