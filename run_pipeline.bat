@echo off
echo Activating venv...
call venv\Scripts\activate.bat

echo Running pipeline...
python scripts\automated_pipeline.py --auto

echo.
echo Pipeline completed.
pause
