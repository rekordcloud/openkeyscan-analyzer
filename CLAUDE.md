# Musical Key CNN - Project Documentation

This file contains technical documentation and important context about this project for future reference.

---

## Project Overview

**Musical Key CNN** is a convolutional neural network-based system for musical key detection and classification. It predicts the musical key of audio files (MP3) using the Camelot Wheel notation system (1A-12A for minor keys, 1B-12B for major keys).

**Key Features:**
- Predicts musical keys for individual MP3 files or entire folders
- Uses CNN architecture based on Korzeniowski & Widmer (2018)
- Outputs both Camelot notation (e.g., "9A") and traditional key notation (e.g., "E minor")
- Can be packaged as a standalone executable for distribution

**Based on Research:**
- Korzeniowski & Widmer. "Genre-Agnostic Key Classification With Convolutional Neural Networks" (ISMIR 2018)
- Training dataset: GiantSteps MTG Key Dataset
- Evaluation dataset: GiantSteps Key Dataset

---

## Project Architecture

### Core Components

1. **predict_keys.py** - Main entry point for key prediction
   - Command-line interface for predicting keys from MP3 files
   - Uses librosa for audio loading (modified from original torchaudio)
   - Preprocesses audio to CQT spectrograms
   - Outputs formatted results with Camelot notation

2. **model.py** - Neural network architecture
   - `KeyNet` class: CNN with 9 convolutional layers
   - Uses batch normalization, ELU activation, and dropout
   - Global average pooling for variable-length inputs
   - 24 output classes (12 keys × 2 modes)

3. **dataset.py** - Dataset handling and Camelot mapping
   - `CAMELOT_MAPPING` dictionary: maps key strings to indices (0-23)
   - `KeyDataset` class for training data loading

4. **eval.py** - Model evaluation utilities
   - `load_model()` function for loading trained weights
   - MIREX key evaluation metrics implementation

5. **train.py** - Training script (not modified in recent work)

6. **predict_keys_server.py** - Long-running server mode (NEW)
   - stdin/stdout JSON protocol for IPC
   - Loads model once, keeps in memory for efficiency
   - ThreadPoolExecutor for concurrent audio preprocessing
   - Ideal for Electron/desktop app integration
   - Supports high-throughput analysis (20+ files/min)

7. **test_server.py** - Server test harness
   - Spawns server as subprocess
   - Tests with random MP3 files
   - Validates protocol and performance

### Audio Processing Pipeline

1. **Load audio**: librosa.load() with mono conversion and resampling to 44.1kHz
2. **Compute CQT**: Constant-Q Transform with 105 bins, 24 bins/octave
3. **Apply log scaling**: log1p() for magnitude compression
4. **Slice spectrogram**: Remove last 2 time frames and last frequency bin
5. **Batch and predict**: Pass through CNN model
6. **Output**: Argmax to get class index, map to Camelot notation

---

## Important Modifications Made

### 1. Audio Loading Backend Change

**File:** `predict_keys.py:66-68`

**Original:**
```python
waveform, sr = torchaudio.load(mp3_path)
# ... stereo to mono conversion
# ... resampling with torchaudio.transforms.Resample
```

**Modified:**
```python
# Use librosa to load and resample audio (avoids torchcodec dependency)
waveform, sr = librosa.load(mp3_path, sr=sample_rate, mono=True)
waveform = waveform.astype(np.float32)
```

**Reason:**
- torchaudio.load() requires torchcodec which has FFmpeg dependency issues
- librosa.load() uses native audio backends (Core Audio on macOS) and is more reliable
- Simplifies code by handling mono conversion and resampling in one call

### 2. Resource Path Resolution for PyInstaller

**File:** `predict_keys.py:12-29`

**Added:**
```python
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)  # PyInstaller temp folder
    except AttributeError:
        base_path = Path(__file__).parent  # Normal execution
    return base_path / relative_path
```

**Usage:** Default model path now uses `get_resource_path('checkpoints/keynet.pt')`

**Reason:** Allows the bundled executable to find the model file in PyInstaller's temporary extraction directory

### 3. PyInstaller Spec File with Auto-Dereferencing

**File:** `predict_keys.spec:72-127`

**Key Features:**
- Bundles `checkpoints/keynet.pt` model file
- Includes hidden imports for sklearn, librosa, numba, soundfile, cffi
- **Post-build hook** automatically replaces all symlinks with actual files
- Ensures distribution is portable when zipped/copied

**Why Symlink Dereferencing:**
- PyInstaller creates 15 symlinks for torch, scipy, and Python framework libraries
- Symlinks break when zipping or copying to other systems
- Post-build hook runs automatically after COLLECT stage

### 4. Standalone Dereference Script

**File:** `dereference_symlinks.py`

Standalone utility for manually dereferencing symlinks if needed. Not required for normal build process since spec file handles it automatically.

---

## Dependencies

### Production Dependencies
- **torch** (>=2.0): PyTorch deep learning framework
- **torchaudio**: Audio I/O (not actively used due to librosa change)
- **librosa**: Audio processing and CQT computation
- **numpy**: Numerical computing
- **tqdm**: Progress bars
- **scipy**: Scientific computing (transitive via librosa)
- **scikit-learn**: Machine learning utilities (transitive via librosa)
- **numba**: JIT compilation for librosa
- **soundfile**: Audio I/O backend for librosa

### Development Dependencies
- **pyinstaller**: Executable packaging

### Dependency Management

**Pipenv (recommended):**
```sh
pipenv install          # Production dependencies
pipenv install --dev    # Add PyInstaller
```

**Traditional (legacy):**
```sh
pip install -r requirements.txt
```

---

## Building Standalone Executable

### Prerequisites
- macOS (tested on macOS 15.6.1, ARM64)
- Python 3.13
- All dependencies installed

### Build Process

```sh
# Install dev dependencies
pipenv install --dev

# Build executable
pyinstaller predict_keys.spec
```

### What Happens During Build

1. **Analysis**: PyInstaller analyzes dependencies and imports
2. **Collection**: Gathers all required Python modules and binaries
3. **Bundling**: Creates `dist/predict_keys/` folder with:
   - Executable binary
   - Python runtime and libraries
   - All dependencies (PyTorch, librosa, etc.)
   - Trained model (`checkpoints/keynet.pt`)
4. **Post-processing**: Automatically dereferences 15 symlinks to actual files

### Build Output

- **Location**: `dist/predict_keys/`
- **Size**: ~780MB (uncompressed), ~224MB (zipped)
- **Portability**: Fully self-contained, can run on any macOS system

### Symlinks Replaced During Build

The following symlinks are automatically replaced with actual files:
- `libtorch.dylib`, `libtorch_cpu.dylib`, `libtorch_python.dylib`
- `libc10.dylib`, `libshm.dylib`, `libomp.dylib`
- `libtorchaudio.so`
- `libc++.1.0.dylib`
- `libgfortran.5.dylib`, `libquadmath.0.dylib`, `libgcc_s.1.1.dylib`
- Python framework symlinks (Python, Resources, Current)

---

## Platform-Specific Notes

### macOS
- **Audio Backend**: Uses Core Audio (native) via librosa/audioread
- **FFmpeg**: NOT required for MP3 decoding on macOS
- **Distribution**: Executable is fully self-contained
- **Tested On**: macOS 15.6.1 (Sequoia), ARM64 architecture

### Linux/Windows
- **Audio Backend**: Would require FFmpeg for MP3 decoding
- **FFmpeg**: Must be bundled or installed on target system
- **Distribution**: Not currently configured, would need platform-specific builds

---

## File Structure

```
MusicalKeyCNN-main/
├── predict_keys.py          # Main CLI entry point
├── predict_keys_server.py   # Server mode (stdin/stdout JSON protocol)
├── test_server.py           # Server test harness
├── model.py                 # CNN architecture
├── dataset.py               # Dataset and Camelot mapping
├── eval.py                  # Evaluation utilities
├── train.py                 # Training script
├── preprocess_data.py       # Dataset preprocessing
├── predict_keys.spec        # PyInstaller configuration (both CLI and server)
├── dereference_symlinks.py  # Manual symlink dereferencing utility
├── Pipfile                  # Pipenv dependencies
├── requirements.txt         # Pip dependencies (legacy)
├── README.md               # User documentation
├── CLAUDE.md               # This file (technical documentation)
├── checkpoints/
│   └── keynet.pt           # Trained model weights (1.8MB)
└── dist/                   # Build output (gitignored)
    └── predict_keys/       # Standalone executable distribution
        ├── predict_keys           # CLI executable
        └── predict_keys_server    # Server executable
```

---

## Common Workflows

### Predict Key for Single File
```sh
python predict_keys.py -f path/to/song.mp3
```

### Predict Keys for Folder
```sh
python predict_keys.py -f path/to/music/folder/
```

### Using Custom Model
```sh
python predict_keys.py -f song.mp3 -m path/to/custom_model.pt
```

### Force CPU Usage
```sh
python predict_keys.py -f song.mp3 --device cpu
```

### Build and Distribute
```sh
# Build
pyinstaller predict_keys.spec

# Test
./dist/predict_keys/predict_keys -f test.mp3

# Package for distribution
cd dist
zip -r predict_keys.zip predict_keys/
```

### Server Mode (Electron/IPC Integration)

```sh
# Start server in development
python predict_keys_server.py

# Test server with 10 random files
python test_server.py
```

---

## Server Mode Architecture

**Purpose**: Long-running process for integration with Electron/JavaScript applications. Avoids model reload overhead by keeping model in memory.

### Protocol: Line-Delimited JSON (NDJSON)

**Communication**: stdin (requests) / stdout (responses)

**Request Format:**
```json
{"id": "uuid-1234", "path": "/absolute/path/to/song.mp3"}
```

**Success Response:**
```json
{
  "id": "uuid-1234",
  "status": "success",
  "camelot": "9A",
  "openkey": "1m",
  "key": "E minor",
  "class_id": 8,
  "filename": "song.mp3"
}
```

**Error Response:**
```json
{
  "id": "uuid-1234",
  "status": "error",
  "error": "File not found",
  "filename": "song.mp3"
}
```

**System Messages:**
```json
{"type": "ready"}       // Sent on startup
{"type": "heartbeat"}   // Sent every 30s
```

### Server Features

1. **Thread-Safe**: Each worker thread loads its own model instance (default: 1 worker)
2. **Memory Efficient**: Lazy model loading per thread (~200MB per worker)
3. **Configurable Concurrency**: `-w` flag controls worker count (1-8 threads)
4. **Fast**: ~0.41s per file average (tested with 10 files)
5. **Reliable**: Auto-restart capability via parent process
6. **Async**: Non-blocking - returns results as they complete
7. **Simple IPC**: Direct pipe communication, no network overhead

### Electron Integration Example

```javascript
const { spawn } = require('child_process');
const readline = require('readline');

// Spawn the server
const server = spawn('./dist/predict_keys/predict_keys_server');

// Set up line reader for responses
const rl = readline.createInterface({
  input: server.stdout,
  crlfDelay: Infinity
});

// Handle responses
rl.on('line', (line) => {
  const response = JSON.parse(line);

  if (response.type === 'ready') {
    console.log('Server ready!');
  } else if (response.id) {
    // Match response to request by ID
    handleResult(response);
  }
});

// Send request
function analyzeFile(filePath) {
  const request = {
    id: generateUUID(),
    path: filePath
  };

  server.stdin.write(JSON.stringify(request) + '\n');
}

// Auto-restart on crash
server.on('exit', (code) => {
  console.log(`Server exited with code ${code}`);
  // Implement restart logic
});
```

### Performance Characteristics

**Throughput (Tested):**
- Single file: ~440ms average
- 10 files concurrent: 4.37s total (437ms avg per file)
- Expected: 20-40 files/min single-threaded
- With batching: 100-200 files/min potential

**Memory Profile:**
- Model: ~80-100MB (loaded once)
- Per-file spectrogram: ~5-10MB (temporary)
- Thread pool: ~10MB
- Total baseline: ~100-150MB

**Startup Time:**
- Model loading: ~1-2s
- Server initialization: ~0.5s
- Total: ~1.5-2.5s ready time

### Testing the Server

```sh
# Run test harness (analyzes 10 random files from ~/Music/spotify)
python test_server.py
```

**Expected Output:**
```
Starting key detection server...
[SERVER] Loading model...
[SERVER] Model loaded on cpu
Server ready!

Finding 10 random MP3 files from ~/Music/spotify...
Found 10 files to analyze
Sending 10 requests...

Results:
----------------------------------------------------------------------
Track1.mp3: 9A (E minor)
Track2.mp3: 4A (F minor)
...
Track10.mp3: 6A (G minor)
----------------------------------------------------------------------

Processed 10 files in 4.37s
Success: 10, Failed: 0
Average: 0.44s per file
```

### Command-Line Options

```sh
# Start server with custom model
python predict_keys_server.py -m path/to/model.pt

# Force CPU usage
python predict_keys_server.py --device cpu

# Adjust worker threads (default: 1, each loads own model)
python predict_keys_server.py -w 2  # 2 workers = 2 model instances

# Note: Each worker loads its own model instance to avoid thread safety issues
# Memory usage = ~200MB per worker + preprocessing buffers
# Workers=1: ~1GB peak | Workers=2: ~1.2GB peak | Workers=4: ~1.6GB peak (estimated)
```

**Threading Model:**
- **Default: 1 worker** (safest, lowest memory)
- Each worker thread has its own model instance (lazy-loaded on first request)
- No locks needed - true parallel processing if workers > 1
- Increase workers for higher throughput, but monitor memory usage

### Error Handling

**File Errors:**
- File not found → Returns error JSON
- Invalid format → Returns error JSON
- Corrupted audio → Returns error JSON

**Protocol Errors:**
- Malformed JSON → Logged to stderr, skipped
- Missing fields → Logged to stderr, skipped

**Process Errors:**
- Model inference failure → Returns error JSON
- Out of memory → Process exits (parent should restart)

### Advantages Over API Server

✅ **Lower latency**: No HTTP overhead
✅ **Simpler**: No port management or CORS
✅ **Process isolation**: Easy to restart on crash
✅ **Resource efficient**: Single model instance
✅ **Built-in IPC**: Node.js child_process handles pipes

### When to Use Server Mode

- ✅ Electron/desktop app integration
- ✅ High-volume batch processing
- ✅ Need to analyze 10+ files without reload
- ✅ Want async/streaming results
- ✅ Memory-constrained environments

### When to Use CLI Mode

- ✅ One-off analysis
- ✅ Shell scripting
- ✅ Manual testing
- ✅ Simple use cases

---

## Technical Details

### Model Architecture
- **Input**: CQT spectrogram (1 channel, 104 frequency bins, variable time frames)
- **Layers**: 9 convolutional layers with increasing feature maps (20 → 160)
- **Pooling**: 3 MaxPool2d layers + Global Average Pooling
- **Regularization**: Dropout2d (p=0.5) after each pooling stage
- **Output**: 24-dimensional logits (one per key class)

### Key Notation Systems

**Camelot Wheel Mapping:**
- **Indices 0-11**: Minor keys (A, e.g., 1A = G# minor)
- **Indices 12-23**: Major keys (B, e.g., 1B = B major)
- **Calculation**: `idx = (class % 12) + 1`, `mode = "A" if class < 12 else "B"`

**Open Key Notation:**
- Alternative notation system used by some DJ software (Traktor, etc.)
- Minor keys: 1m-12m (m = minor)
- Major keys: 1d-12d (d = dur/major, from German)
- **Conversion** (both use the same offset):
  - `openkey_num = ((camelot_num - 8) % 12) + 1`
  - `openkey_mode = "m" if minor else "d"`
- **Examples**:
  - 8A (A minor) = 1m
  - 8B (C major) = 1d
  - 9A (E minor) = 2m
  - 12A (C# minor) = 5m
  - 6A (G minor) = 11m

Both notations are provided in server responses for compatibility with different DJ applications.

### CQT Parameters
- **Sample Rate**: 44,100 Hz
- **Hop Length**: 8,820 samples (~200ms)
- **Bins**: 105 frequency bins
- **Bins per Octave**: 24
- **Frequency Min**: 65 Hz (C2)

### Performance Metrics
The included model (`keynet.pt`) achieves:
- **Weighted Score**: 73.51% (MIREX metric)
- **Correct**: 66.72%
- **Fifth**: 8.11%
- **Relative**: 6.79%
- **Parallel**: 3.48%
- **Other**: 14.90%

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install dependencies with `pipenv install` or `pip install -r requirements.txt`

### Issue: "Could not load libtorchcodec" (if using torchaudio.load)
**Solution:** This is resolved by using librosa.load() instead (already implemented)

### Issue: Executable doesn't work on other machines
**Solution:** Ensure symlinks were dereferenced (automatic in spec file). Verify by running `find dist/predict_keys -type l` (should return nothing)

### Issue: "FileNotFoundError" when running executable
**Solution:** Use absolute paths, not tilde (~) expansion. The executable doesn't expand `~` like shells do.

### Issue: Slow first-time execution
**Solution:** Normal. PyTorch and model loading take a few seconds on first run.

---

## Future Enhancements

### Potential Improvements
1. **Cross-platform builds**: Configure PyInstaller for Linux and Windows
2. **Bundle FFmpeg**: For non-macOS platforms
3. **GPU acceleration**: Add CUDA support for faster inference
4. **Batch processing**: Optimize for processing many files at once
5. **Web interface**: Add Flask/FastAPI wrapper for web-based predictions
6. **Real-time processing**: Stream audio for live key detection
7. **Confidence scores**: Output prediction confidence alongside key

### Build Optimization
- Exclude unused scipy/sklearn modules to reduce size
- Use UPX compression more aggressively
- Consider single-file executable (may increase startup time)

---

## Development Notes

### Testing Changes
Always test after modifications:
```sh
# Test in development environment
python predict_keys.py -f test.mp3

# Test built executable
./dist/predict_keys/predict_keys -f test.mp3

# Test distribution portability
cd /tmp
unzip path/to/predict_keys.zip
./predict_keys/predict_keys -f test.mp3
```

### Code Style
- Follow existing patterns in codebase
- Use type hints where applicable
- Document functions with docstrings
- Keep predict_keys.py compatible with both dev and PyInstaller environments

### Git Ignore
Ensure these are gitignored:
- `dist/`
- `build/`
- `*.spec` (if generating custom ones)
- `venv/`
- `.venv/`
- `__pycache__/`
- `*.pyc`

---

## References

### Papers
- [Korzeniowski & Widmer, 2018 - Genre-Agnostic Key Classification](https://arxiv.org/abs/1808.05340)
- [Korzeniowski & Widmer, 2017 - End-to-End Musical Key Estimation](https://arxiv.org/abs/1706.02921)

### Datasets
- [GiantSteps MTG Key Dataset](https://github.com/GiantSteps/giantsteps-mtg-key-dataset)
- [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset)

### Tools
- [PyTorch](https://pytorch.org/)
- [librosa](https://librosa.org/)
- [PyInstaller](https://pyinstaller.org/)
- [Camelot Wheel (Mixed In Key)](https://mixedinkey.com/camelot-wheel/)

---

## Changelog

### Recent Modifications (2025-11-01)
1. ✅ Changed audio loading from torchaudio to librosa (FFmpeg independence)
2. ✅ Added resource path resolution for PyInstaller bundling
3. ✅ Created PyInstaller spec file with model bundling
4. ✅ Implemented automatic symlink dereferencing in spec file
5. ✅ Migrated to Pipenv for dependency management
6. ✅ Updated README with Pipenv instructions and build process
7. ✅ Uncommented torch/torchaudio in requirements.txt for legacy support
8. ✅ Created standalone dereference_symlinks.py utility
9. ✅ Verified executable portability on macOS ARM64
10. ✅ **Implemented server mode (predict_keys_server.py)** for Electron/IPC integration
11. ✅ Created stdin/stdout JSON protocol (NDJSON) for async communication
12. ✅ Added ThreadPoolExecutor for concurrent audio preprocessing
13. ✅ Created test_server.py for validation and testing
14. ✅ Updated PyInstaller spec to build server-only executable (removed CLI build)
15. ✅ Achieved ~0.44s per file throughput (tested with 10 concurrent files)
16. ✅ **Added Open Key notation** support alongside Camelot (1m-12m, 1d-12d)
17. ✅ Implemented camelot_to_openkey() conversion function
18. ✅ Updated server response JSON to include both notations
19. ✅ **Fixed Open Key conversion bug** - corrected formula to use same offset (-8) for both minor and major keys
20. ✅ **Implemented per-thread model loading** - each worker gets own model instance for thread safety
21. ✅ **Changed default workers from 4 to 1** - reduces memory usage and prevents crashes
22. ✅ **Added memory tracking to test_server.py** - psutil integration with granular per-file tracking
23. ✅ **Reduced peak memory usage** - 1.1GB (1 worker) vs 1.3GB (4 workers) for 10 files

### Known Working Configuration
- Python: 3.13.2
- torch: 2.9.0
- torchaudio: 2.9.0
- librosa: 0.11.0
- numpy: 2.3.4
- PyInstaller: 6.16.0
- Platform: macOS 15.6.1 (Darwin 24.6.0), ARM64

---

*Last Updated: 2025-11-01*
