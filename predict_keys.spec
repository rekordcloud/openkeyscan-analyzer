# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Determine the base path
base_path = Path.cwd()

# Data files to bundle
datas = [
    (str(base_path / 'checkpoints' / 'keynet.pt'), 'checkpoints'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'sklearn.utils._weight_vector',
    'librosa',
    'numba',
    'soundfile',
    'cffi',
]

a = Analysis(
    ['predict_keys_server.py'],
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
    name='predict_keys_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
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
    name='predict_keys',
)

# Post-build: Dereference symlinks for distribution compatibility
import os
import shutil

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
                print(f"  ✓ Copied directory: {symlink_path.name}")
            else:
                shutil.copy2(target, symlink_path)
                print(f"  ✓ Copied file: {symlink_path.name}")

        except Exception as e:
            print(f"  ✗ Error: {symlink_path.name}: {e}")

    print(f"Successfully dereferenced {len(symlinks_found)} symlinks")
    print("="*70 + "\n")

# Run the dereferencing
dist_folder = Path(DISTPATH) / 'predict_keys'
if dist_folder.exists():
    dereference_symlinks(dist_folder)
