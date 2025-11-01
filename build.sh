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
pyinstaller --noconfirm predict_keys.spec

echo ""
echo "======================================================================"
echo "Build Complete!"
echo "======================================================================"
echo ""
echo "Output:"
echo "  Executable: dist/predict_keys/predict_keys_server"
echo "  Archive:    dist/predict_keys.zip"
echo ""
echo "Test the build:"
echo "  ./dist/predict_keys/predict_keys_server"
echo ""
echo "Or extract and distribute the zip file:"
echo "  dist/predict_keys.zip"
echo ""
