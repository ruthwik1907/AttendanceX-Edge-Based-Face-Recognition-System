@echo off
echo === Creating venv ===
python -m venv venv

echo === Activating venv ===
call venv\Scripts\activate.bat

echo === Upgrading pip ===
python -m pip install --upgrade pip

echo === Installing requirements ===
pip install -r requirements.txt

echo.
echo === Setup complete! ===
pause
