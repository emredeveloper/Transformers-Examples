@echo off
REM Transformers Examples dependency installer for Windows

IF NOT DEFINED VIRTUAL_ENV (
    echo Warning: No virtual environment detected. Consider creating one before installing dependencies.
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

