@echo off
echo [%date% %time%] Starting simulation wrapper... > debug_log.txt
echo Current directory: %CD% >> debug_log.txt

echo [%date% %time%] Checking Python... >> debug_log.txt
python --version >> debug_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found or failed to run! >> debug_log.txt
    exit /b 1
)

echo [%date% %time%] Checking ADDA... >> debug_log.txt
if exist "..\adda\win64\adda_ocl.exe" (
    echo ADDA found. >> debug_log.txt
) else (
    echo [ERROR] ADDA executable not found at ..\adda\win64\adda_ocl.exe >> debug_log.txt
    dir ..\adda\win64 >> debug_log.txt
)

echo [%date% %time%] Running simulation script... >> debug_log.txt
python -u run_rough_surface_simulation.py >> debug_log.txt 2>&1

echo [%date% %time%] Simulation script finished with exit code %ERRORLEVEL% >> debug_log.txt

