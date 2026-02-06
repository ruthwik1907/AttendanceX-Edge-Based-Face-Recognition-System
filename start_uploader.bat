@echo off
echo Activating venv...
call venv\Scripts\activate.bat

echo Starting uploader at http://127.0.0.1:5000 ...
python scripts\serve_uploads_polished.py --host 0.0.0.0 --port 5000

pause
