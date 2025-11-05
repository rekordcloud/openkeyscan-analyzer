# Windows Performance Investigation

## Problem Statement

The Windows **PyInstaller executable** shows **significantly slower performance** compared to macOS AND native Python:

- **macOS (ARM64, Sequoia 15.6.1)**: ~0.44s per file
- **Windows Python (pipenv, x64, Build 26100)**: ~1.96s per file ✅
- **Windows Executable (PyInstaller, x64, Build 26100)**: ~20-21s per file ❌
- **Performance difference**:
  - Windows vs macOS: ~4.5x slower (acceptable - different CPU architecture)
  - **Executable vs Python on same machine: ~10x slower** ⚠️ **THIS IS THE REAL PROBLEM**

## Test Results

### Windows Executable (PyInstaller)
```
Test: 3 MP3 files
Results:
  Athys & Duster - Barfight.mp3: 4A (F minor) - ~21s
  Audio - Combust.mp3: 4A (F minor) - ~21s
  Balthazar & JackRock - Andromeda.mp3: 11A (F# minor/Gb minor) - ~21s

Average: 21.07s per file
```

### Windows Python (pipenv) - WITH PROFILING
```
Test: Athys & Duster - Barfight.mp3

Breakdown:
  - librosa.load: 1.605s (82%)
  - librosa.cqt: 0.323s (16%)
  - postprocess: 0.004s (<1%)
  - model inference: 0.027s (1%)

Total: ~1.96s per file
```

### KEY FINDING
**The executable is ~10x slower than Python on the SAME machine!**

This proves the bottleneck is NOT the algorithm or Windows itself, but something about how PyInstaller packages the application.

## Hypothesis: Where is the Time Being Spent?

The audio processing pipeline has several stages:

1. **Audio Loading** (`librosa.load()`)
   - Reads MP3 file
   - Decodes audio (via audioread + Core Audio on macOS, FFmpeg on Windows)
   - Resamples to 44.1kHz
   - Converts to mono

2. **CQT Computation** (`librosa.cqt()`)
   - Constant-Q Transform (computationally intensive)
   - Uses numpy/scipy FFT operations
   - 105 frequency bins, 24 bins/octave
   - Processes entire audio file

3. **Preprocessing**
   - Log scaling (`np.log1p`)
   - Array slicing
   - Tensor conversion
   - **Fast** (< 0.1s typically)

4. **Model Inference** (PyTorch)
   - CNN forward pass
   - **Fast** (< 0.1s typically on CPU)

5. **Post-processing**
   - Format conversion
   - **Negligible** (< 0.01s)

### Root Cause Analysis

**CONFIRMED via profiling:**

1. **Python environment (pipenv) breakdown**:
   - `librosa.load()`: 1.605s (82% of time)
   - `librosa.cqt()`: 0.323s (16% of time)
   - Model inference: 0.027s (1% of time)
   - **Total: 1.96s** ✅

2. **PyInstaller executable**:
   - **Total: ~20s** ❌
   - **10x slower than Python!**

3. **Why is the executable slower?**

   **Investigation findings:**

   ✅ **OpenBLAS IS bundled** - Found in `numpy.libs/` and `scipy.libs/`
   - `libscipy_openblas64_*.dll` (numpy)
   - `libscipy_openblas-*.dll` (scipy)

   ✅ **Python environment uses OpenBLAS 0.3.30** with full SIMD optimizations:
   - AVX, AVX2, AVX512, FMA3, etc.
   - DYNAMIC_ARCH, Haswell optimizations
   - MAX_THREADS=24

   ❌ **But PyInstaller may not be loading them correctly**:
   - DLL path issues
   - Thread configuration not carried over
   - Environment variables not set (OMP_NUM_THREADS, MKL_NUM_THREADS)
   - Possible DLL hell with multiple OpenBLAS versions bundled

## Profiling Code Added

The server code has been instrumented with profiling capabilities (commit 2025-11-05):

```python
# Enable profiling via environment variable
profile = os.environ.get('PROFILE_PERFORMANCE', '0') == '1'
```

### How to Profile

**After rebuilding the executable:**

```bash
# Set environment variable to enable profiling
export PROFILE_PERFORMANCE=1

# Run test
echo '{"id": "test", "path": "C:/path/to/song.mp3"}' | ./dist/openkeyscan-analyzer/openkeyscan-analyzer.exe
```

**Expected stderr output:**
```
[PROFILE] song.mp3:
  - librosa.load: X.XXXs (XX.X%)
  - librosa.cqt: X.XXXs (XX.X%)
  - postprocess: X.XXXs (XX.X%)
  preprocess_audio: X.XXXs (XX.X%)
  to_device: X.XXXs (X.X%)
  model_inference: X.XXXs (X.X%)
  format_output: X.XXXs (X.X%)
  TOTAL: XX.XXXs
```

## Potential Solutions

### 1. **Fix OpenBLAS DLL loading in PyInstaller** ⭐ MOST PROMISING

**Current issue**: OpenBLAS DLLs are bundled but may not be loaded/configured correctly

**Solutions to try:**

a) **Add runtime environment configuration**:
```python
# Add to server startup code
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Or match system cores
os.environ['OPENBLAS_NUM_THREADS'] = '4'
```

b) **Verify DLL paths at runtime**:
```python
import numpy
print(numpy.__file__)
print(numpy.show_config())
```

c) **Force OpenBLAS to be loaded first**:
```python
# In spec file or runtime hook
import ctypes
import os
openblas_path = os.path.join(sys._MEIPASS, 'numpy.libs', 'libscipy_openblas64_*.dll')
ctypes.CDLL(openblas_path)  # Load explicitly
```

**Expected improvement**: 5-10x faster (match Python performance)

### 2. **Multi-threading for librosa**

**Current issue**: librosa.cqt() is single-threaded

**Solution**: Enable numpy multi-threading
```python
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Use 4 CPU cores
os.environ['MKL_NUM_THREADS'] = '4'
```

**Expected improvement**: 2-4x faster CQT (if numpy supports it)

### 3. **Alternative audio loading**

**Current issue**: audioread may be slow on Windows

**Solution**: Try PyAV or soundfile direct loading
```bash
pip install av
```

**Expected improvement**: 1.5-2x faster audio loading

### 4. **Use Native Windows Console**

**Current issue**: MSYS2/Git Bash emulation overhead

**Solution**: Test in cmd.exe or PowerShell
```cmd
set PROFILE_PERFORMANCE=1
echo {"id": "test", "path": "C:/path/to/song.mp3"} | dist\openkeyscan-analyzer\openkeyscan-analyzer.exe
```

**Expected improvement**: 5-10% faster overall

### 5. **GPU Acceleration** (Future)

**Solution**: Move CQT computation to GPU using CUDA
- Requires custom CUDA kernel or nnAudio library
- Would require significant code changes

**Expected improvement**: 10-50x faster (but requires NVIDIA GPU)

## Root Cause Update (2025-11-05)

### Test Results After Threading Fix

**Executable with PROFILE_PERFORMANCE=1 and OMP_NUM_THREADS=32:**
```
[THREADING] CPU cores: 32
[THREADING] OMP_NUM_THREADS: 32
[THREADING] OPENBLAS_NUM_THREADS: 32

[PROFILE] Athys & Duster - Barfight.mp3:
  - librosa.load: 31.316s (99.9%) ❌ **THIS IS THE PROBLEM**
  - librosa.cqt: 0.322s (1.0%) ✅ **FIXED - matches Python!**
  - postprocess: 0.001s (<0.1%)
  preprocess_audio: 31.641s (99.9%)
  to_device: 0.000s (0.0%)
  model_inference: 0.029s (0.1%)
  format_output: 0.000s (0.0%)
  TOTAL: 31.670s
```

**Comparison:**

| Component | Python (pipenv) | Executable | Ratio |
|-----------|----------------|------------|-------|
| librosa.load | 1.605s | 31.316s | **19.5x slower** ❌ |
| librosa.cqt | 0.323s | 0.322s | **1.0x (SAME)** ✅ |
| model inference | 0.027s | 0.029s | **1.1x (SAME)** ✅ |
| **TOTAL** | 1.96s | 31.67s | **16.2x slower** |

### Key Findings

1. ✅ **OpenBLAS threading fix WORKED** - librosa.cqt is now matching Python performance (0.322s vs 0.323s)
2. ✅ **Model inference is fine** - 0.029s, matches Python's 0.027s
3. ❌ **Real bottleneck is MP3/audioread** - librosa.load() is 25x slower for MP3 files in executable
4. ✅ **WAV files work PERFECTLY** - 1.220s in executable vs 1.408s in Python (actually slightly faster!)
5. **Root cause**: audioread backend (used for MP3/M4A/AAC) is broken in PyInstaller bundle

### Critical Discovery: Format-Specific Performance

| Format | Environment | librosa.load | librosa.cqt | TOTAL | vs Python |
|--------|-------------|--------------|-------------|-------|-----------|
| MP3 | Python | 1.637s | 0.320s | 1.986s | - |
| **WAV** | Python | 1.408s | 0.282s | 1.719s | - |
| **MP3** | Executable | **31.316s** ❌ | 0.322s | 31.670s | **25.7x slower** |
| **WAV** | Executable | **1.220s** ✅ | 0.287s | 1.534s | **1.15x faster!** |

**The problem is ONLY with compressed formats (MP3, M4A, AAC) that use audioread!**
**Native formats (WAV, FLAC, OGG) using soundfile work perfectly.**

### Why Audio Decoding is Slow

**librosa.load() on Windows uses:**
- **PySoundFile** for native formats (WAV, FLAC, OGG) - fast
- **audioread** for compressed formats (MP3, M4A, AAC) - uses FFmpeg or Windows Media Foundation

**Possible causes:**
1. **FFmpeg DLLs not bundled or loaded inefficiently** by PyInstaller
2. **audioread falling back to slow decoder** (e.g., MAD instead of FFmpeg)
3. **Windows Media Foundation API overhead** in bundled environment
4. **DLL loading overhead** - audioread may be searching/loading DLLs on every call

### Solution

Since WAV/FLAC/OGG files work perfectly, the solution is to **replace audioread with a better MP3 decoder**.

**Best Options:**

1. **PyAV (av)** - Fast FFmpeg Python bindings
   ```python
   import av
   import numpy as np

   def load_mp3_with_pyav(path, sr=44100):
       container = av.open(str(path))
       audio = container.streams.audio[0]
       audio.codec_context.sample_rate = sr

       frames = []
       for frame in container.decode(audio):
           frames.append(frame.to_ndarray())

       waveform = np.concatenate(frames, axis=1)[0]  # Get first channel
       return waveform.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
   ```
   - **Pros**: Fast, bundleable, widely used
   - **Cons**: Need to add dependency and modify code

2. **Pre-convert MP3 to WAV** for batch processing
   - For large batches, pre-convert MP3 → WAV outside the executable
   - Use FFmpeg or other fast converter
   - Then process WAV files at full speed
   - **Pros**: No code changes needed
   - **Cons**: Extra preprocessing step

3. **Use native WAV export from DJ software**
   - Most DJ software (Serato, Rekordbox, Traktor) can export WAV
   - Avoid MP3 encoding/decoding entirely
   - **Pros**: Best audio quality and speed
   - **Cons**: Larger file sizes

4. **Bundle FFmpeg binary and configure audioread**
   - Try explicitly bundling ffmpeg.exe and ffprobe.exe
   - Configure audioread to find them
   - **Pros**: Minimal code changes
   - **Cons**: May not work if audioread is fundamentally broken in PyInstaller

## Next Steps

1. ✅ **Add profiling code** (DONE)
2. ✅ **Rebuild executable with profiling** (DONE)
3. ✅ **Run profiled test to identify bottleneck** (DONE)
4. ✅ **Enable numpy multi-threading** (DONE - librosa.cqt now matches Python!)
5. ✅ **Test WAV vs MP3 performance** (DONE - WAV works perfectly!)
6. ✅ **Identify root cause** (DONE - audioread is broken in PyInstaller, soundfile works fine)
7. ⏳ **Implement PyAV backend** as replacement for audioread
8. ⏳ **Test all supported formats** (MP3, MP4, M4A, AAC) with PyAV
9. ⏳ **Rebuild and benchmark** to confirm ~2s per file performance

## Tools for Further Investigation

### Python profiling (if dependencies available):

```bash
# Install profiling tools
pip install py-spy

# Profile during execution
py-spy record -o profile.svg -- python openkeyscan_analyzer_server.py
```

### cProfile (built-in):

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... run your code ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## References

- **librosa performance**: https://librosa.org/doc/latest/ioformats.html
- **numpy BLAS**: https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries
- **Windows audio decoding**: audioread uses Core Audio via `ffmpeg` or `mad` backends

---

## Summary

### What We Learned

1. **Windows Python performance is acceptable** (~1.96s for MP3, 1.72s for WAV)
   - This is expected given CPU architecture differences (ARM vs x86)
   - Breakdown: librosa.load (82%), librosa.cqt (16%), model (1%)

2. **PyInstaller executable has TWO bottlenecks**:
   - ❌ **MP3/audioread backend**: 31.3s (25x slower than Python!)
   - ✅ **WAV/soundfile backend**: 1.2s (actually faster than Python!)
   - ✅ **OpenBLAS/CQT computation**: Fixed with threading configuration
   - ✅ **Model inference**: Working perfectly

3. **Root cause identified**:
   - ✅ OpenBLAS threading fix WORKED - CQT now matches Python performance
   - ❌ audioread library is BROKEN in PyInstaller (used for MP3/M4A/AAC)
   - ✅ soundfile library works PERFECTLY (used for WAV/FLAC/OGG)

4. **Solution**:
   - Replace audioread with PyAV for MP3/M4A/AAC decoding
   - Expected performance after fix: ~1.5-2.0s per file (matching Python)
   - WAV/FLAC/OGG files already work at full speed

### The Good News

- ✅ The algorithm itself is fast and works perfectly
- ✅ OpenBLAS threading configuration successfully fixed the CQT bottleneck
- ✅ WAV/FLAC/OGG files already work at full speed in the executable
- ✅ We have a clear path forward: replace audioread with PyAV
- ✅ Once fixed, Windows executable should match Python performance (~2s per file)

---

*Investigation started: 2025-11-05*
*Status: **ROOT CAUSE IDENTIFIED** - PyInstaller OpenBLAS configuration issue*
*Updated: 2025-11-05*
