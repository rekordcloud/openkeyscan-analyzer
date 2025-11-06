#!/bin/bash

# Build script for x64 (Intel) architecture
# This script must be run from a Rosetta 2 terminal

set -e  # Exit on error

# Verify we're running on x86_64 (Rosetta 2)
CURRENT_ARCH=$(uname -m)
if [ "$CURRENT_ARCH" != "x86_64" ]; then
    echo "========================================================================"
    echo "ERROR: Wrong Terminal Architecture"
    echo "========================================================================"
    echo ""
    echo "This script builds for x64 (Intel) and must be run from a Rosetta 2"
    echo "terminal."
    echo ""
    echo "Current architecture: $CURRENT_ARCH"
    echo "Required architecture: x86_64"
    echo ""
    echo "To run in Rosetta 2 mode:"
    echo "  1. Open Terminal app"
    echo "  2. Right-click on Terminal in Applications"
    echo "  3. Select 'Get Info'"
    echo "  4. Check 'Open using Rosetta'"
    echo "  5. Restart Terminal"
    echo ""
    echo "Or use: arch -x86_64 /bin/bash"
    echo "========================================================================"
    exit 1
fi

# Call the main build script with x64 parameter
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/_build-mac.sh" x64
