@echo off
REM Build script for Musical Key CNN
REM Builds the standalone executable using PyInstaller

echo ======================================================================
echo Building Musical Key CNN Standalone Application
echo ======================================================================
echo.

REM Check if pipenv is available
where pipenv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: pipenv not found
    echo Install it with: pip install pipenv
    exit /b 1
)

REM Check if pyinstaller is available in pipenv environment
pipenv run pyinstaller --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: pyinstaller not found in pipenv environment
    echo Install it with: pipenv install --dev
    exit /b 1
)

REM Clean previous build artifacts (optional, PyInstaller will handle this)
if exist "build" (
    echo Cleaning build\ directory...
    rmdir /s /q build
)

echo Starting PyInstaller build...
echo.

REM Run PyInstaller with --noconfirm to skip prompts (via pipenv)
pipenv run pyinstaller --noconfirm openkeyscan_analyzer.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ======================================================================
    echo Build Failed!
    echo ======================================================================
    exit /b 1
)

echo.
echo ======================================================================
echo Build Complete!
echo ======================================================================
echo.
echo Output:
echo   Executable: dist\openkeyscan-analyzer\openkeyscan-analyzer.exe
echo   Archive:    dist\openkeyscan-analyzer.zip
echo.

REM Detect architecture (Windows is typically x64)
REM PROCESSOR_ARCHITECTURE can be AMD64, ARM64, or x86
set ARCH_DIR=x64
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set ARCH_DIR=arm64
if "%PROCESSOR_ARCHITECTURE%"=="x86" set ARCH_DIR=ia32

REM Set destination directory
set DEST_DIR=%USERPROFILE%\workspace\openkeyscan\OpenKeyScan-app\build\lib\win\%ARCH_DIR%
set ZIP_FILE=dist\openkeyscan-analyzer.zip

echo Installing to distribution directory...
echo   Architecture: %ARCH_DIR%
echo   Destination:  %DEST_DIR%
echo.

REM Create destination directory if it doesn't exist
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

REM Copy the zip file, replacing any existing file
if exist "%ZIP_FILE%" (
    copy /Y "%ZIP_FILE%" "%DEST_DIR%\openkeyscan-analyzer.zip"
    echo Success: Installed: %DEST_DIR%\openkeyscan-analyzer.zip
) else (
    echo Error: ZIP file not found at %ZIP_FILE%
    exit /b 1
)

echo.
echo ======================================================================
echo Installation Complete!
echo ======================================================================
echo.
echo Test the build:
echo   dist\openkeyscan-analyzer\openkeyscan-analyzer.exe
echo.
echo Distribution package:
echo   %DEST_DIR%\openkeyscan-analyzer.zip
echo.
