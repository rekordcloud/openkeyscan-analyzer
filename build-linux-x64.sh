#!/bin/bash

# Build script for Linux x64, run from macOS (or Linux) with Docker.
# PyInstaller cannot cross-compile, so the build runs inside an x64 Debian
# container. Debian 11 (glibc 2.31) keeps the binary compatible with
# Ubuntu 20.04+ and other distros of that age or newer.

set -e

if ! command -v docker &> /dev/null; then
    echo "Error: docker not found"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$HOME/workspace/openkeyscan/openkeyscan-app/build/lib/linux/x64"
IMAGE="python:3.12-slim-bullseye"

echo "======================================================================"
echo "Building openkeyscan-analyzer for Linux x64 in Docker ($IMAGE)"
echo "======================================================================"
echo ""

docker run --rm --platform linux/amd64 \
    -v "$SCRIPT_DIR:/src" \
    -v openkeyscan-pip-cache:/root/.cache/pip \
    -w /src \
    "$IMAGE" \
    bash -c '
        set -e
        apt-get update -qq
        apt-get install -y -qq binutils > /dev/null

        # CPU-only torch: the default Linux wheel pulls in ~2GB of CUDA libraries
        pip install -q torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
        # Versions match Pipfile.lock
        pip install -q "numpy==1.26.4" "librosa==0.11.0" "soundfile==0.13.1" "av==16.1.0" \
            "scipy==1.17.1" "numba==0.64.0" "llvmlite==0.46.0" "scikit-learn==1.8.0" \
            "soxr==1.0.0" "audioread==3.1.0" tqdm \
            "pyinstaller==6.19.0" "pyinstaller-hooks-contrib==2026.2"

        rm -rf build/linux dist/linux
        pyinstaller --noconfirm --distpath dist/linux --workpath build/linux openkeyscan_analyzer.spec

        # No cleanup_video_deps.py here: on Linux, libavcodec links the video codec
        # libraries directly, so removing them would break PyAV audio decoding
    '

echo ""
echo "Copying build to distribution directory..."
mkdir -p "$DIST_DIR"
rm -rf "$DIST_DIR/openkeyscan-analyzer"
# -R keeps PyInstaller's symlinks instead of duplicating the libraries
cp -R "$SCRIPT_DIR/dist/linux/openkeyscan-analyzer" "$DIST_DIR/"
echo "✓ Copied to: $DIST_DIR/openkeyscan-analyzer"
