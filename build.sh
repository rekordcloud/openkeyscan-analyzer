#!/bin/bash

# Build script for Musical Key CNN
# Builds the standalone executable using PyInstaller

set -e  # Exit on error

echo "======================================================================"
echo "Building Musical Key CNN Standalone Application"
echo "======================================================================"
echo ""

# Check if pyinstaller is available
if ! command -v pyinstaller &> /dev/null; then
    echo "Error: pyinstaller not found"
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

# Run PyInstaller with --noconfirm to skip prompts
pyinstaller --noconfirm openkeyscan_analyzer.spec

echo ""
echo "======================================================================"
echo "Build Complete!"
echo "======================================================================"
echo ""
echo "Output:"
echo "  Executable: dist/openkeyscan-analyzer/openkeyscan-analyzer-server"
echo "  Archive:    dist/openkeyscan-analyzer.zip"
echo ""
echo "Test the build:"
echo "  ./dist/openkeyscan-analyzer/openkeyscan-analyzer-server"
echo ""
echo "Or extract and distribute the zip file:"
echo "  dist/openkeyscan-analyzer.zip"
echo ""
