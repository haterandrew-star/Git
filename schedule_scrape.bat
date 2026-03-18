@echo off
REM ============================================================
REM  schedule_scrape.bat
REM  Registers a Windows Task Scheduler task that runs the
REM  prediction pipeline every 6 hours, automatically.
REM
REM  Run once as Administrator to register the task.
REM  After that it runs silently in the background.
REM ============================================================

SET TASK_NAME=CCA_Chess_Predictor_Scrape
SET PYTHON=%~dp0run_pipeline_task.bat
SET PROJECT_DIR=%~dp0

echo.
echo  CCA Entry Predictor — Scheduler Setup
echo  ======================================
echo  Task name: %TASK_NAME%
echo  Script:    %PYTHON%
echo  Directory: %PROJECT_DIR%
echo.

REM Create the runner batch file that the scheduler will call
echo @echo off > "%PROJECT_DIR%run_pipeline_task.bat"
echo cd /d "%PROJECT_DIR%" >> "%PROJECT_DIR%run_pipeline_task.bat"
echo REM Scrape all upcoming tournaments + inject entry counts into HTML >> "%PROJECT_DIR%run_pipeline_task.bat"
echo python tools\scrape_upcoming.py >> "%PROJECT_DIR%run_pipeline_task.bat"
echo REM Run prediction pipeline for WO2026 (active tournament) >> "%PROJECT_DIR%run_pipeline_task.bat"
echo python tools\run_pipeline.py --id WO2026 >> "%PROJECT_DIR%run_pipeline_task.bat"

echo  Created: run_pipeline_task.bat
echo.

REM Delete existing task if it exists
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

REM Register the task — runs every 6 hours, starting now
schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "\"%PROJECT_DIR%run_pipeline_task.bat\"" ^
  /sc HOURLY ^
  /mo 6 ^
  /st 00:00 ^
  /ru "%USERNAME%" ^
  /f

IF %ERRORLEVEL% EQU 0 (
    echo  [OK] Task registered: %TASK_NAME%
    echo  [OK] Runs every 6 hours
    echo.
    echo  To view in Task Scheduler: taskschd.msc
    echo  To run immediately:        schtasks /run /tn "%TASK_NAME%"
    echo  To remove:                 schtasks /delete /tn "%TASK_NAME%" /f
) ELSE (
    echo  [ERR] Failed to create task.
    echo        Try running this script as Administrator:
    echo        Right-click schedule_scrape.bat → Run as administrator
)

echo.
pause
