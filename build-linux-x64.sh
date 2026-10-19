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

        # CPU-only torch: the default Linux wheel pulls in ~2GB of CUDA libraries.
        # torchaudio is left out: the server never imports it, and its bundled libraries
        # link against system libtorch/libsox, which makes the .deb depend on -dev packages
        pip install -q torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
        # Versions match Pipfile.lock
        pip install -q "numpy==1.26.4" "librosa==0.11.0" "soundfile==0.13.1" "av==16.1.0" \
            "scipy==1.17.1" "numba==0.64.0" "llvmlite==0.46.0" "scikit-learn==1.8.0" \
            "soxr==1.0.0" "audioread==3.1.0" tqdm \
            "pyinstaller==6.19.0" "pyinstaller-hooks-contrib==2026.2"

        rm -rf build/linux dist/linux
        pyinstaller --noconfirm --distpath dist/linux --workpath build/linux openkeyscan_analyzer.spec

        # No cleanup_video_deps.py here: on Linux, libavcodec links the video codec
        # libraries directly, so removing them would break PyAV audio decoding

        INTERNAL=dist/linux/openkeyscan-analyzer/_internal

        # Optional numba threading layers that need system libtbb/libgomp; numba falls
        # back to its built-in workqueue layer
        rm -f $INTERNAL/numba/np/ufunc/tbbpool.*.so
        rm -f $INTERNAL/numba/np/ufunc/omppool.*.so

        # torch ships ~120 MB of C++ test programs and test libraries
        rm -rf $INTERNAL/torch/test
        find $INTERNAL/torch/bin -type f ! -name torch_shm_manager -delete
        rm -f $INTERNAL/torch/lib/libjitbackend_test.so $INTERNAL/torch/lib/libtorchbind_test.so \
            $INTERNAL/torch/lib/libbackend_with_compiler.so

        # Conveyor drops ELF files it cannot identify as Linux binaries (no GLIBC_ symbol
        # versions and no .note.ABI-tag), e.g. _contextvars, which breaks Python startup.
        # Give those files the standard GNU ABI tag (Linux 3.2.0) as a non-loaded section
        printf "\x04\x00\x00\x00\x10\x00\x00\x00\x01\x00\x00\x00GNU\x00\x00\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00" > /tmp/abitag
        find dist/linux -type f | while read f; do
            [ "$(head -c 4 "$f" | tail -c 3)" = "ELF" ] || continue
            objdump -T "$f" 2>/dev/null | grep -q GLIBC_ && continue
            readelf -n "$f" 2>/dev/null | grep -q NT_GNU_ABI_TAG && continue
            objcopy --add-section .note.ABI-tag=/tmp/abitag --set-section-flags .note.ABI-tag=noload,readonly "$f"
            echo "Added ABI tag: $f"
        done
    '

echo ""
echo "Copying build to distribution directory..."
mkdir -p "$DIST_DIR"
rm -rf "$DIST_DIR/openkeyscan-analyzer"
# -L resolves PyInstaller's symlinks: Conveyor disables ASAR packaging when the app
# contains symlinks, and the app expects resources/app.asar.unpacked/lib
cp -RL "$SCRIPT_DIR/dist/linux/openkeyscan-analyzer" "$DIST_DIR/"
echo "✓ Copied to: $DIST_DIR/openkeyscan-analyzer"
