@echo off
echo Setting up virtual environment and installing dependencies...

REM Check if Python is installed and is version 3.11
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Detected Python version: %PYTHON_VERSION%

REM Check if the version starts with 3.11
echo %PYTHON_VERSION% | findstr /b "3.11" > nul
if %errorlevel% neq 0 (
    echo Python 3.11.x is required. Your version is %PYTHON_VERSION%
    echo Please install Python 3.11 and try again.
    exit /b 1
)

echo Python 3.11 detected, proceeding with setup...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Verify activation
echo Virtual environment activated: %VIRTUAL_ENV%

REM Install mediapipe
echo Installing mediapipe...
pip install mediapipe

REM Uninstall jax
pip uninstall -y jax

REM Install requirements.txt if it exists
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt file not found in current directory.
)

echo.
echo Setup complete! The virtual environment is now active.
echo To deactivate the virtual environment later, run: deactivate
echo.