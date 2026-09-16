@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Run from a temporary copy because Git will replace setup.bat during checkout.
if /i "%~1"=="--bootstrap" goto :bootstrap
set "BOOTSTRAP_FILE=%TEMP%\arxiv-dashboard-setup-%RANDOM%-%RANDOM%.bat"
copy /y "%~f0" "%BOOTSTRAP_FILE%" >nul
call "%BOOTSTRAP_FILE%" --bootstrap "%CD%"
set "SETUP_EXIT=%ERRORLEVEL%"
del /q "%BOOTSTRAP_FILE%" >nul 2>&1
exit /b %SETUP_EXIT%

:bootstrap
set "PROJECT_DIR=%~2"
set "REPOSITORY_URL=https://github.com/YHFTF/arxiv-conversational-recommender.git"

cd /d "%PROJECT_DIR%" || goto :directory_error

echo [1/4] Checking Git and Docker...
where git >nul 2>&1 || goto :git_missing
where docker >nul 2>&1 || goto :docker_missing
docker compose version >nul 2>&1 || goto :compose_missing
docker info >nul 2>&1 || goto :docker_not_running

echo [2/4] Getting the main branch...
if not exist ".git\" (
    git init || goto :failed
    git remote add origin "%REPOSITORY_URL%" || goto :failed
    git fetch origin main || goto :failed
    git checkout -B main origin/main --force || goto :failed
) else (
    git remote get-url origin >nul 2>&1 || git remote add origin "%REPOSITORY_URL%" || goto :failed
    git remote set-url origin "%REPOSITORY_URL%" || goto :failed
    git fetch origin main || goto :failed
    git checkout main || goto :failed
    git merge --ff-only origin/main || goto :update_conflict
)
git branch --set-upstream-to=origin/main main >nul 2>&1

echo [3/4] Pulling the shared Docker image...
docker compose -f docker-compose.dashboard.yml pull dashboard || goto :pull_failed

echo [4/4] Starting the dashboard...
docker compose -f docker-compose.dashboard.yml up -d --no-build dashboard || goto :failed

echo.
echo Setup complete. Dashboard: http://localhost:8080
start "" "http://localhost:8080"
exit /b 0

:git_missing
echo [ERROR] Git is not installed or is not in PATH.
goto :failed_pause

:docker_missing
echo [ERROR] Docker Desktop is not installed or docker is not in PATH.
goto :failed_pause

:compose_missing
echo [ERROR] Docker Compose v2 is unavailable. Update Docker Desktop.
goto :failed_pause

:docker_not_running
echo [ERROR] Docker is not running. Start Docker Desktop and run this file again.
goto :failed_pause

:directory_error
echo [ERROR] Cannot open the target folder: %PROJECT_DIR%
goto :failed_pause

:update_conflict
echo [ERROR] The existing repository cannot be fast-forwarded to origin/main.
echo Commit or back up local changes, then run setup.bat again.
goto :failed_pause

:pull_failed
echo [ERROR] Could not pull the dashboard image.
echo The GHCR package must exist and be public, or Docker must be logged in to ghcr.io.
goto :failed_pause

:failed
echo [ERROR] Setup failed. Review the message above and try again.

:failed_pause
pause
exit /b 1
