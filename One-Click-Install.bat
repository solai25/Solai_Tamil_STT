@echo off

echo Installing and launching Tamil STT...

:: Check if the virtual environment folder already exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists, skipping creation...
)

call .venv\Scripts\activate
python setup.py
python app_gui.py
pause