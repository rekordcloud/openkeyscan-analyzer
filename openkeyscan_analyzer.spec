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
    'av',  # PyAV for optimized audio loading
    # Comprehensive scipy imports to prevent lazy loading issues
    'scipy._lib',
    'scipy._lib.messagestream',
    'scipy.special',
    'scipy.special._cdflib',
    'scipy.special._ufuncs',
    'scipy.special._ufuncs_cxx',
    'scipy.stats',
    'scipy.stats._distn_infrastructure',
    'scipy.stats.distributions',
    'scipy.stats._continuous_distns',
    'scipy.stats._discrete_distns',
    'scipy.stats._stats_py',
    'scipy.stats._stats',
    'scipy.signal',
    'scipy.signal.windows',
    'scipy.signal._peak_finding',
    'scipy.fft',
    'scipy.fftpack',
    'scipy.linalg',
    'scipy.linalg.blas',
    'scipy.linalg.lapack',
]

# Runtime hooks to fix scipy initialization
runtime_hooks = ['runtime_hook_scipy.py']

a = Analysis(
    ['openkeyscan_analyzer_server.py'],
    pathex=[str(base_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
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
    python_options=['X utf8_mode=1'],  # Force UTF-8 mode for proper Unicode handling on Windows
    codesign_identity='Developer ID Application: Rekordcloud B.V. (2B7KR8BSYR)',
    entitlements_file='./analyzer.entitlements',
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
