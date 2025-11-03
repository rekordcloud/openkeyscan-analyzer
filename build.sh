#!/bin/bash

# Build script for Musical Key CNN
# Builds the standalone executable using PyInstaller

set -e  # Exit on error

echo "======================================================================"
echo "Building Musical Key CNN Standalone Application"
echo "======================================================================"
echo ""

# Check if pipenv is available
if ! command -v pipenv &> /dev/null; then
    echo "Error: pipenv not found"
    echo "Install it with: pip install pipenv"
    exit 1
fi

# Check if pyinstaller is available in pipenv environment
if ! pipenv run which pyinstaller &> /dev/null; then
    echo "Error: pyinstaller not found in pipenv environment"
    echo "Install it with: pipenv install --dev"
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
pipenv run pyinstaller --noconfirm openkeyscan_analyzer.spec

echo ""
echo "======================================================================"
echo "Build Complete!"
echo "======================================================================"
echo ""
echo "Output:"
echo "  Executable: dist/openkeyscan-analyzer/openkeyscan-analyzer-server"
echo "  Archive:    dist/openkeyscan-analyzer.zip"
echo ""

# Detect architecture (using Node.js naming convention)
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    ARCH_DIR="arm64"
elif [ "$ARCH" = "x86_64" ]; then
    ARCH_DIR="x64"
else
    echo "Warning: Unknown architecture '$ARCH', using as-is"
    ARCH_DIR="$ARCH"
fi

# Move zip file to distribution directory
DIST_DIR="$HOME/openkeyscan/build/lib/mac/$ARCH_DIR"
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
echo "  ./dist/openkeyscan-analyzer/openkeyscan-analyzer-server"
echo ""
echo "Distribution package:"
echo "  $DIST_DIR/openkeyscan-analyzer.zip"
echo ""
