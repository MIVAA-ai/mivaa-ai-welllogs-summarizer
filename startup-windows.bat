@echo off

:: Check if one argument is passed
if "%~1"=="" (
  echo Usage: %~nx0 ^<base_directory^>
  exit /b 1
)

set BASE_DIR=%~1

:: Ensure base directory exists
if not exist "%BASE_DIR%" (
  echo Base directory does not exist. Creating it: %BASE_DIR%
  mkdir "%BASE_DIR%"
)

:: Define folder paths based on the .env file structure
setlocal enabledelayedexpansion
set FOLDERS="%BASE_DIR%\processed" "%BASE_DIR%\uploads" "%BASE_DIR%\logs" "%BASE_DIR%\worker\data\in" "%BASE_DIR%\worker\data\results" "%BASE_DIR%\worker\data\summary"

:: Create the folders
for %%F in (%FOLDERS%) do (
  if not exist %%F (
    echo Creating folder: %%F
    mkdir %%F
  ) else (
    echo Folder already exists: %%F
  )
)

:: Update .env file with the new folder paths
if not exist .env (
  echo .env file not found in the current directory.
  exit /b 1
)

(for /f "tokens=* delims=" %%A in (.env) do (
  set "line=%%A"
  if "!line!"=="" (
    echo.>> updated_env.tmp
  ) else (
    echo !line! | findstr /b "PROCESSED_VOLUME=" >nul && (
      echo PROCESSED_VOLUME=%BASE_DIR%\processed>> updated_env.tmp
    ) || echo !line! | findstr /b "UPLOADS_VOLUME=" >nul && (
      echo UPLOADS_VOLUME=%BASE_DIR%\uploads>> updated_env.tmp
    ) || echo !line! | findstr /b "LOGS_VOLUME=" >nul && (
      echo LOGS_VOLUME=%BASE_DIR%\logs>> updated_env.tmp
    ) || echo !line! | findstr /b "DATA_IN_VOLUME=" >nul && (
      echo DATA_IN_VOLUME=%BASE_DIR%\worker\data\in>> updated_env.tmp
    ) || echo !line! | findstr /b "DATA_RESULTS_VOLUME=" >nul && (
      echo DATA_RESULTS_VOLUME=%BASE_DIR%\worker\data\results>> updated_env.tmp
    ) || echo !line! | findstr /b "SUMMARY_VOLUME=" >nul && (
      echo SUMMARY_VOLUME=%BASE_DIR%\worker\data\summary>> updated_env.tmp
    ) || (
      echo %%A>> updated_env.tmp
    )
  )
))

move /y updated_env.tmp .env >nul

:: Run docker-compose command
echo Starting Docker container using docker-compose...
docker-compose --env-file .env up --build
