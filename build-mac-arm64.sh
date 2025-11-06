#!/bin/bash

# Build script for ARM64 (Apple Silicon) architecture
# This script must be run from a native ARM64 terminal (not Rosetta 2)

set -e  # Exit on error

# Verify we're running on ARM64
CURRENT_ARCH=$(uname -m)
if [ "$CURRENT_ARCH" != "arm64" ]; then
    echo "========================================================================"
    echo "ERROR: Wrong Terminal Architecture"
    echo "========================================================================"
    echo ""
    echo "This script builds for ARM64 (Apple Silicon) and must be run from"
    echo "a native ARM64 terminal."
    echo ""
    echo "Current architecture: $CURRENT_ARCH"
    echo "Required architecture: arm64"
    echo ""
    echo "Please run this script from a native (non-Rosetta 2) terminal."
    echo "========================================================================"
    exit 1
fi

# Call the main build script with arm64 parameter
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_build-mac.sh" arm64
