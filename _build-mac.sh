#!/bin/bash

# Build script for Musical Key CNN
# Builds the standalone executable using PyInstaller

set -e  # Exit on error

# Parse architecture argument
if [ -z "$1" ]; then
    echo "Error: Architecture argument required (arm64 or x64)"
    echo "Usage: $0 <arm64|x64>"
    exit 1
fi

TARGET_ARCH="$1"
if [ "$TARGET_ARCH" != "arm64" ] && [ "$TARGET_ARCH" != "x64" ]; then
    echo "Error: Invalid architecture '$TARGET_ARCH'"
    echo "Must be 'arm64' or 'x64'"
    exit 1
fi

# Convert x64 to x86_64 for PyInstaller
if [ "$TARGET_ARCH" = "x64" ]; then
    PYINSTALLER_ARCH="x86_64"
else
    PYINSTALLER_ARCH="arm64"
fi

echo "======================================================================"
echo "Building Musical Key CNN Standalone Application"
echo "Architecture: $TARGET_ARCH"
echo "======================================================================"
echo ""

# Check if pipenv is available
if ! command -v pipenv &> /dev/null; then
    echo "Error: pipenv not found"
    echo "Install it with: pip install pipenv"
    exit 1
fi

# Update code from git
echo "Pulling latest code from git..."
git pull
echo ""

# Install/update dependencies
echo "Installing/updating dependencies..."
pipenv install --dev
echo ""

# Check if pyinstaller is available in pipenv environment
if ! pipenv run which pyinstaller &> /dev/null; then
    echo "Error: pyinstaller not found in pipenv environment"
    echo "This should not happen after pipenv install --dev"
    exit 1
fi

# Clean previous build artifacts (optional, PyInstaller will handle this)
if [ -d "build" ]; then
    echo "Cleaning build/ directory..."
    rm -rf build
fi

echo "Starting PyInstaller build..."
echo ""

# Run PyInstaller with --noconfirm to skip prompts (via pipenv)
pipenv run pyinstaller --noconfirm --target-arch $PYINSTALLER_ARCH openkeyscan_analyzer.spec

echo ""
echo "======================================================================"
echo "Build Complete!"
echo "======================================================================"
echo ""
echo "Output:"
echo "  Executable: dist/openkeyscan-analyzer/openkeyscan-analyzer"
echo "  Archive:    dist/openkeyscan-analyzer.zip"
echo ""

# Use the target architecture for directory naming
ARCH_DIR="$TARGET_ARCH"

# Move zip file to distribution directory
DIST_DIR="$HOME/workspace/openkeyscan/build/lib/mac/$ARCH_DIR"
ZIP_FILE="dist/openkeyscan-analyzer.zip"

echo "Installing to distribution directory..."
echo "  Architecture: $ARCH_DIR"
echo "  Destination:  $DIST_DIR"
echo ""

# Create destination directory if it doesn't exist
mkdir -p "$DIST_DIR"

# Move the zip file, replacing any existing file
if [ -f "$ZIP_FILE" ]; then
    cp "$ZIP_FILE" "$DIST_DIR/openkeyscan-analyzer.zip"
    echo "✓ Installed: $DIST_DIR/openkeyscan-analyzer.zip"
else
    echo "Error: ZIP file not found at $ZIP_FILE"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Test the build:"
echo "  ./dist/openkeyscan-analyzer/openkeyscan-analyzer"
echo ""
echo "Distribution package:"
echo "  $DIST_DIR/openkeyscan-analyzer.zip"
echo ""
