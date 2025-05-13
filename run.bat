@echo off
echo Activating virtual environment and running camera detection...

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run setup_venv.bat first to create the virtual environment.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Verify activation
echo Virtual environment activated: %VIRTUAL_ENV%

REM Check if the app is present
if not exist app.py (
    echo Error: app.py not found!
    echo Please check the file path and try again.
    deactivate
    exit /b 1
)

REM Run the camera detection script
echo Running app.py
python app.py

REM Deactivate virtual environment when done
deactivate
echo Virtual environment deactivated.