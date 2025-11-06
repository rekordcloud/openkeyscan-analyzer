# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

# Read target architecture from environment variable (set by build script)
# This ensures PyInstaller validates that the terminal arch matches the target
target_arch = os.environ.get('TARGET_ARCH', None)  # 'arm64', 'x86_64', or None (auto-detect)

# Determine the base path
base_path = Path.cwd()

# Data files to bundle
datas = [
    (str(base_path / 'checkpoints' / 'keynet.pt'), 'checkpoints'),
    # Bundle ffmpeg binaries for fast MP3/M4A/AAC decoding (fixes 25x slowdown)
    # The ffmpeg.exe (4.3MB) is a minimal audio-only build
]

# Add ffmpeg binaries if they exist (platform-specific)
ffmpeg_windows = base_path / 'ffmpeg.exe'
ffmpeg_unix = base_path / 'ffmpeg'
if ffmpeg_windows.exists():
    datas.append((str(ffmpeg_windows), '.'))
if ffmpeg_unix.exists():
    datas.append((str(ffmpeg_unix), '.'))

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'sklearn.utils._weight_vector',
    'librosa',
    'numba',
    'soundfile',
    'cffi',
]

a = Analysis(
    ['openkeyscan_analyzer_server.py'],
    pathex=[str(base_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_server = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='openkeyscan-analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,  # Set from environment variable (arm64, x86_64, or None)
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_server,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='openkeyscan-analyzer',
)

# Post-build: Dereference symlinks for distribution compatibility
import os
import shutil
import zipfile

def dereference_symlinks(dist_path):
    """Replace all symlinks with actual files/directories."""
    print("\n" + "="*70)
    print("Post-build: Dereferencing symlinks for distribution compatibility")
    print("="*70)

    symlinks_found = []

    # Find all symlinks
    for root, dirs, files in os.walk(dist_path):
        root_path = Path(root)
        for name in files + dirs:
            item_path = root_path / name
            if item_path.is_symlink():
                symlinks_found.append(item_path)

    if not symlinks_found:
        print("No symlinks found")
        return

    print(f"Found {len(symlinks_found)} symlinks to dereference")

    # Replace each symlink with actual file/directory
    for symlink_path in symlinks_found:
        try:
            target = symlink_path.resolve()

            if not target.exists():
                print(f"  ⚠️  Warning: Target does not exist: {symlink_path}")
                continue

            # Remove the symlink
            symlink_path.unlink()

            # Copy the actual file/directory
            if target.is_dir():
                shutil.copytree(target, symlink_path)
                print(f"  [+] Copied directory: {symlink_path.name}")
            else:
                shutil.copy2(target, symlink_path)
                print(f"  [+] Copied file: {symlink_path.name}")

        except Exception as e:
            print(f"  [-] Error: {symlink_path.name}: {e}")

    print(f"Successfully dereferenced {len(symlinks_found)} symlinks")
    print("="*70 + "\n")

def create_zip_archive(dist_path, output_name):
    """Create a zip archive of the distribution folder."""
    print("\n" + "="*70)
    print("Post-build: Creating zip archive for distribution")
    print("="*70)

    zip_path = dist_path.parent / f"{output_name}.zip"

    # Remove existing zip if it exists
    if zip_path.exists():
        zip_path.unlink()
        print(f"Removed existing archive: {zip_path.name}")

    print(f"Creating archive: {zip_path.name}")
    print(f"Compressing: {dist_path.name}/")

    file_count = 0
    total_size = 0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # Walk through all files in the distribution folder
        for root, dirs, files in os.walk(dist_path):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path from dist folder
                arcname = file_path.relative_to(dist_path.parent)
                zipf.write(file_path, arcname)
                file_count += 1
                total_size += file_path.stat().st_size

    zip_size = zip_path.stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0

    print(f"  [+] Added {file_count} files")
    print(f"  [+] Original size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  [+] Compressed size: {zip_size / 1024 / 1024:.1f} MB")
    print(f"  [+] Compression ratio: {compression_ratio:.1f}%")
    print(f"  [+] Archive saved: {zip_path}")
    print("="*70 + "\n")

# Run the dereferencing
dist_folder = Path(DISTPATH) / 'openkeyscan-analyzer'
if dist_folder.exists():
    dereference_symlinks(dist_folder)
    create_zip_archive(dist_folder, 'openkeyscan-analyzer')
