@echo off
setlocal
cd /d "%~dp0" || exit /b 1
if not exist "venv\Scripts\python.exe" (
    echo Project environment is missing. See docs/setup/local.md.
    exit /b 1
)
set "PATH=%CD%\venv\Scripts;%PATH%"
set "PYTHONUNBUFFERED=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
python -u -c "from dotenv import load_dotenv; import runpy; load_dotenv('.env'); runpy.run_path('dashboard/server.py', run_name='__main__')"
exit /b %ERRORLEVEL%
