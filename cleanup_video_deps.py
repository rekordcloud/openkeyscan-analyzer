#!/usr/bin/env python3
"""
Cleanup script to remove video and image dependencies from PyInstaller dist.

This script removes video codec libraries, image codec libraries, and video-related
Python modules from the built distribution, reducing size by ~56 MB while preserving
all audio functionality.

Platform Support:
    - macOS: Removes .dylib files from _internal/ and _internal/av/.dylibs/
    - Windows: Removes .dll files from _internal/ and _internal/av/dlls/
    - Linux: Removes .so files from _internal/ and _internal/av/.libs/

Usage:
    python cleanup_video_deps.py [dist_path]

    If dist_path is not provided, defaults to: dist/openkeyscan-analyzer/

Examples:
    # macOS/Linux
    python3 cleanup_video_deps.py

    # Windows
    python cleanup_video_deps.py

    # Custom path
    python cleanup_video_deps.py path/to/dist/openkeyscan-analyzer/

    # Dry run (preview changes)
    python cleanup_video_deps.py --dry-run
"""

import os
import sys
from pathlib import Path
import shutil
import platform


# Platform detection
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# File extensions by platform
if IS_WINDOWS:
    LIB_EXT = '.dll'
    PYTHON_EXT = '.pyd'
    DYLIB_DIR = 'av/dlls'  # Windows uses different directory structure
elif IS_MACOS:
    LIB_EXT = '.dylib'
    PYTHON_EXT = '.so'
    DYLIB_DIR = 'av/.dylibs'
else:  # Linux
    LIB_EXT = '.so'
    PYTHON_EXT = '.so'
    DYLIB_DIR = 'av/.libs'  # Linux may use .libs

# Video codec libraries to remove (base names without extensions)
# Extensions will be added based on platform
VIDEO_CODEC_BASES = [
    # AV1 codecs
    ('SvtAv1Enc', ['3.1.0', '3']),                    # AV1 encoder
    ('aom', ['3.11.0', '3']),                         # AV1 codec
    ('dav1d', ['7', '']),                             # AV1 decoder
    # VP8/VP9
    ('vpx', ['11', '']),                              # VP8/VP9
    # H.264/H.265
    ('x264', ['165', '']),                            # H.264 encoder
    ('x265', ['215', '']),                            # H.265 encoder
    ('openh264', ['7', '']),                          # H.264 Cisco
    # Video processing
    ('swscale', ['9.1.100', '9']),                    # Video scaling
    ('avfilter', ['11.4.100', '11']),                 # Audio/video filtering
]

# Image codec libraries to remove (base names)
IMAGE_CODEC_BASES = [
    ('webp', ['7.1.10', '7']),                        # WebP images
    ('webpmux', ['3.1.1', '3']),                      # WebP muxer
    ('sharpyuv', ['0.1.1', '0']),                     # WebP YUV
]

def generate_library_names(base_name, versions):
    """Generate all possible library names for different platforms."""
    names = []

    # Generate with 'lib' prefix and without
    prefixes = ['lib', ''] if IS_WINDOWS else ['lib']

    for prefix in prefixes:
        for version in versions:
            if version:
                # With version
                if IS_WINDOWS:
                    # Windows: libname-X.dll or name-X.dll
                    names.append(f"{prefix}{base_name}-{version}{LIB_EXT}")
                else:
                    # Unix: libname.so.X or libname.X.dylib
                    if IS_MACOS:
                        names.append(f"{prefix}{base_name}.{version}{LIB_EXT}")
                    else:
                        names.append(f"{prefix}{base_name}{LIB_EXT}.{version}")
            else:
                # Without version
                names.append(f"{prefix}{base_name}{LIB_EXT}")

    return names

# Generate full library lists for current platform
VIDEO_CODECS = []
for base, versions in VIDEO_CODEC_BASES:
    VIDEO_CODECS.extend(generate_library_names(base, versions))

IMAGE_CODECS = []
for base, versions in IMAGE_CODEC_BASES:
    IMAGE_CODECS.extend(generate_library_names(base, versions))

# Video Python modules to remove (directories and files)
# Use platform-specific extensions
VIDEO_MODULES = [
    'av/video',                                       # Entire video module directory
    'av/subtitles',                                   # Entire subtitles module directory
    f'av/sidedata/motionvectors.cpython-312-{platform.system().lower()}{PYTHON_EXT}',  # Motion vectors
]

# Audio libraries to preserve (safeguard - will NOT remove these)
AUDIO_LIBRARIES_SAFEGUARD = [
    'libmp3lame',
    'libogg',
    'libvorbis',
    'libopus',
    'libspeex',
    'libtwolame',
    'libopencore-amr',
    'libavcodec',      # Contains both audio and video, but needed for audio
    'libavformat',     # Container formats
    'libavutil',       # Utilities
    'libswresample',   # Audio resampling
]


def get_file_size(path):
    """Get file size in bytes, return 0 if file doesn't exist."""
    try:
        return path.stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def format_size(bytes_size):
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def is_audio_library(filename):
    """Check if filename matches any audio library safeguard."""
    return any(audio_lib in filename for audio_lib in AUDIO_LIBRARIES_SAFEGUARD)


def remove_file_safely(file_path, dry_run=False):
    """Remove a file if it exists and is not an audio library."""
    if not file_path.exists():
        return 0, False

    if file_path.is_file():
        # Safety check: don't remove audio libraries
        if is_audio_library(file_path.name):
            print(f"  ⚠️  SKIPPED (audio library): {file_path.name}")
            return 0, False

        size = get_file_size(file_path)
        if dry_run:
            print(f"  [DRY RUN] Would remove: {file_path.name} ({format_size(size)})")
        else:
            print(f"  ✓ Removing: {file_path.name} ({format_size(size)})")
            file_path.unlink()
        return size, True

    return 0, False


def remove_directory_safely(dir_path, dry_run=False):
    """Remove a directory if it exists."""
    if not dir_path.exists():
        return 0, False

    if dir_path.is_dir():
        # Calculate directory size
        size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())

        if dry_run:
            print(f"  [DRY RUN] Would remove directory: {dir_path.name}/ ({format_size(size)})")
        else:
            print(f"  ✓ Removing directory: {dir_path.name}/ ({format_size(size)})")
            shutil.rmtree(dir_path)
        return size, True

    return 0, False


def cleanup_video_deps(dist_path, dry_run=False):
    """
    Remove video and image dependencies from PyInstaller distribution.

    Args:
        dist_path: Path to the distribution directory (e.g., dist/openkeyscan-analyzer/)
        dry_run: If True, only print what would be removed without actually removing

    Returns:
        Total bytes removed
    """
    dist_path = Path(dist_path)

    if not dist_path.exists():
        print(f"❌ Error: Distribution path does not exist: {dist_path}")
        return 0

    internal_path = dist_path / '_internal'
    av_dylibs_path = internal_path / DYLIB_DIR

    if not internal_path.exists():
        print(f"❌ Error: _internal directory not found: {internal_path}")
        return 0

    print("=" * 70)
    print("Cleaning up video and image dependencies from distribution")
    print("=" * 70)
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be actually removed")
        print("=" * 70)

    total_removed = 0
    total_files = 0

    # 1. Remove video codec libraries from _internal/
    print(f"\n📦 Removing video codec libraries from _internal/...")
    for lib in VIDEO_CODECS:
        size, removed = remove_file_safely(internal_path / lib, dry_run)
        if removed:
            total_removed += size
            total_files += 1

    # 2. Remove video codec libraries from av library directory
    if av_dylibs_path.exists():
        print(f"\n📦 Removing video codec libraries from _internal/{DYLIB_DIR}/...")
        for lib in VIDEO_CODECS:
            size, removed = remove_file_safely(av_dylibs_path / lib, dry_run)
            if removed:
                total_removed += size
                total_files += 1

    # 3. Remove image codec libraries from _internal/
    print(f"\n🖼️  Removing image codec libraries from _internal/...")
    for lib in IMAGE_CODECS:
        size, removed = remove_file_safely(internal_path / lib, dry_run)
        if removed:
            total_removed += size
            total_files += 1

    # 4. Remove image codec libraries from av library directory
    if av_dylibs_path.exists():
        print(f"\n🖼️  Removing image codec libraries from _internal/{DYLIB_DIR}/...")
        for lib in IMAGE_CODECS:
            size, removed = remove_file_safely(av_dylibs_path / lib, dry_run)
            if removed:
                total_removed += size
                total_files += 1

    # 5. Remove video Python modules
    print(f"\n🐍 Removing video Python modules...")
    for module in VIDEO_MODULES:
        module_path = internal_path / module
        if module_path.is_dir():
            size, removed = remove_directory_safely(module_path, dry_run)
        else:
            size, removed = remove_file_safely(module_path, dry_run)

        if removed:
            total_removed += size
            total_files += 1

    # Summary
    print("\n" + "=" * 70)
    if dry_run:
        print(f"🔍 DRY RUN SUMMARY")
    else:
        print(f"✅ CLEANUP COMPLETE")
    print("=" * 70)
    print(f"Total items removed: {total_files}")
    print(f"Total space saved: {format_size(total_removed)}")
    print("=" * 70)

    return total_removed


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Remove video/image dependencies from PyInstaller distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean up default distribution
  python cleanup_video_deps.py

  # Clean up specific distribution
  python cleanup_video_deps.py dist/openkeyscan-analyzer/

  # Dry run (preview what would be removed)
  python cleanup_video_deps.py --dry-run
        """
    )

    parser.add_argument(
        'dist_path',
        nargs='?',
        default='dist/openkeyscan-analyzer',
        help='Path to distribution directory (default: dist/openkeyscan-analyzer/)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing files'
    )

    args = parser.parse_args()

    try:
        bytes_removed = cleanup_video_deps(args.dist_path, dry_run=args.dry_run)

        if args.dry_run:
            print(f"\n💡 Run without --dry-run to actually remove files")
            return 0

        if bytes_removed > 0:
            print(f"\n✨ Successfully cleaned up distribution!")
            return 0
        else:
            print(f"\n⚠️  No files were removed (already clean or not found)")
            return 1

    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
