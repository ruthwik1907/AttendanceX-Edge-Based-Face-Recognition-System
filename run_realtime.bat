@echo off
echo Activating venv...
call venv\Scripts\activate.bat

echo Starting realtime inference...
python scripts\realtime_tflite_infer_with_excel.py

pause
