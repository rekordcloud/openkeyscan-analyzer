#!/usr/bin/env python3
"""
Optimized Key Detection Server with PyAV support
Based on openkeyscan_analyzer_server.py with performance improvements

Key optimizations:
1. Uses PyAV for compressed formats (MP3, MP4, M4A, AAC) on Windows
2. Adds warmup CQT to eliminate first-run overhead
3. Uses non-planar float format and contiguous arrays for better performance
"""

import sys
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# ============================================================================
# CRITICAL: UTF-8 Encoding Configuration for Windows/PyInstaller
# ============================================================================
# On Windows, Python defaults to cp1252 encoding for stdio, but Node.js
# child_process sends UTF-8. This mismatch causes UnicodeDecodeError when
# reading JSON with non-ASCII characters (e.g., file paths with accents).
#
# Solution: Reconfigure stdin/stdout to UTF-8 at runtime at MODULE LEVEL.
# Must be done BEFORE any sys.stdin.readline() calls and at module level
# (not inside a function) so it executes during import.
# ============================================================================
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

# Configure threading BEFORE importing numpy-dependent libraries
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()
os.environ.setdefault('OMP_NUM_THREADS', str(NUM_CORES))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(NUM_CORES))
os.environ.setdefault('MKL_NUM_THREADS', str(NUM_CORES))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(NUM_CORES))

# Import numpy-dependent libraries
import torch
import numpy as np

# Force scipy imports early to avoid lazy loading issues in bundled executables
# MUST be done before importing librosa which uses scipy
# Handle the known PyInstaller issue with scipy.stats
scipy_loaded = False
try:
    # Import base scipy first
    import scipy
    import scipy._lib

    # Import scipy.special (required by stats)
    import scipy.special
    try:
        import scipy.special._ufuncs
    except ImportError:
        pass

    # Import signal, fft, and linalg (used by librosa)
    import scipy.signal
    import scipy.fft
    import scipy.fftpack
    import scipy.linalg

    # Try to import stats - this may fail with 'obj' is not defined
    try:
        import scipy.stats
        import scipy.stats._distn_infrastructure
        import scipy.stats.distributions
        scipy_loaded = True
        print("Scipy modules pre-loaded successfully", file=sys.stderr)
    except NameError as e:
        if "'obj' is not defined" in str(e):
            print("Warning: Known scipy.stats PyInstaller issue detected, stats may not work", file=sys.stderr)
        else:
            raise
    except ImportError as e:
        print(f"Warning: Some scipy.stats modules could not be loaded: {e}", file=sys.stderr)

except Exception as e:
    print(f"Warning: Scipy initialization error: {e}", file=sys.stderr)

# Now import librosa after scipy is fully loaded
import librosa

# Import PyAV for fast audio loading on Windows
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

# Import from existing modules
from dataset import CAMELOT_MAPPING
from eval import load_model

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.mp4', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.au'}

# Formats that benefit from PyAV on Windows
PYAV_FORMATS = {'.mp3', '.mp4', '.m4a', '.aac'}


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        base_path = Path(__file__).parent
    return base_path / relative_path


def load_audio_pyav_optimized(audio_path, sample_rate=44100):
    """
    Optimized PyAV loading for fast audio decoding and CQT compatibility.

    Key optimizations:
    - Uses non-planar float format for better memory layout
    - Ensures contiguous array for efficient CQT computation
    - Applies consistent normalization
    - Includes warmup CQT to eliminate first-run overhead

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        np.ndarray: Audio waveform as contiguous float32 array
    """
    profile = os.environ.get('PROFILE_PERFORMANCE', '0') == '1'

    if profile:
        t0 = time.time()

    container = av.open(str(audio_path))
    audio_stream = container.streams.audio[0]

    # Use non-planar float format for better CQT performance
    resampler = av.AudioResampler(
        format='flt',  # Non-planar float32 (critical for fast CQT)
        layout='mono',
        rate=sample_rate
    )

    frames = []
    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            for resampled in resampled_frames:
                frame_array = resampled.to_ndarray()
                frames.append(frame_array)

    # Flush resampler to get remaining frames
    remaining = resampler.resample(None)
    if remaining:
        for resampled in remaining:
            frame_array = resampled.to_ndarray()
            frames.append(frame_array)

    container.close()

    # Concatenate all frames
    if not frames:
        raise ValueError(f"No audio frames found in {audio_path}")

    waveform = np.concatenate(frames, axis=1)
    if waveform.ndim > 1:
        waveform = waveform[0]

    # CRITICAL: Ensure contiguous memory layout for fast CQT
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    # Normalize to match librosa output characteristics
    max_val = np.abs(waveform).max()
    if max_val > 0:
        # Scale to 0.95 to match typical librosa amplitude
        waveform = waveform / max_val * 0.95

    if profile:
        print(f"    - load_audio_pyav_optimized: {time.time() - t0:.3f}s", file=sys.stderr)

    # NOTE: Warmup CQT moved to server initialization to avoid hanging in worker threads
    # librosa.cqt() has thread-safety issues when first called in worker threads

    return waveform


def preprocess_audio(audio_path, sample_rate=44100, n_bins=105, hop_length=8820):
    """
    Loads an audio file and extracts a log-magnitude CQT spectrogram.
    Uses optimized PyAV for compressed formats on Windows.

    Args:
        audio_path (Path): Path to audio file
        sample_rate (int): Target sampling rate
        n_bins (int): Number of CQT bins
        hop_length (int): Hop length for CQT

    Returns:
        torch.Tensor: Shape (1, freq_bins, time_frames), ready for model input
    """
    profile = os.environ.get('PROFILE_PERFORMANCE', '0') == '1'
    audio_path = Path(audio_path)
    suffix = audio_path.suffix.lower()

    # Use optimized PyAV for compressed formats on Windows
    use_pyav = (
        sys.platform == 'win32' and
        PYAV_AVAILABLE and
        suffix in PYAV_FORMATS
    )

    if use_pyav:
        if profile:
            print(f"  Using PyAV OPTIMIZED for {suffix}", file=sys.stderr)
        waveform = load_audio_pyav_optimized(audio_path, sample_rate)
    else:
        # Use librosa for native formats (WAV, FLAC, OGG) or non-Windows
        if profile:
            print(f"  Using librosa for {suffix}", file=sys.stderr)
            t0 = time.time()
        waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        waveform = waveform.astype(np.float32)
        if profile:
            print(f"    - librosa.load: {time.time() - t0:.3f}s", file=sys.stderr)

    # Compute CQT
    if profile:
        t0 = time.time()
    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length,
                      n_bins=n_bins, bins_per_octave=24, fmin=65)
    if profile:
        print(f"    - librosa.cqt: {time.time() - t0:.3f}s", file=sys.stderr)

    # Post-process
    if profile:
        t0 = time.time()
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # Remove last frequency bin
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)  # Shape: (1, freq, time)
    if profile:
        print(f"    - postprocess: {time.time() - t0:.3f}s", file=sys.stderr)

    return spec_tensor


def camelot_output(pred_camelot):
    """Formats the Camelot prediction."""
    idx = (pred_camelot % 12) + 1
    mode = "A" if pred_camelot < 12 else "B"
    camelot_str = f"{idx}{mode}"

    names = [k for k, v in CAMELOT_MAPPING.items() if v == pred_camelot]
    if names:
        key_text = "/".join(sorted(set(names)))
    else:
        key_text = "Unknown"
    return camelot_str, key_text


def camelot_to_openkey(camelot_str):
    """Converts Camelot notation to Open Key notation."""
    camelot_num = int(camelot_str[:-1])
    camelot_mode = camelot_str[-1]
    openkey_num = ((camelot_num - 8) % 12) + 1
    openkey_mode = "m" if camelot_mode == "A" else "d"
    return f"{openkey_num}{openkey_mode}"


class KeyDetectionServer:
    """Server that processes key detection requests via stdin/stdout."""

    def __init__(self, model_path=None, device=None, num_workers=1):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        if model_path is None:
            self.model_path = get_resource_path('checkpoints/keynet.pt')
        else:
            self.model_path = model_path

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.thread_local = threading.local()

        print(f"Optimized Server Configuration:", file=sys.stderr)
        print(f"  Workers: {self.num_workers}", file=sys.stderr)
        print(f"  Model path: {self.model_path}", file=sys.stderr)
        print(f"  Device: {self.device}", file=sys.stderr)
        print(f"  PyAV available: {PYAV_AVAILABLE}", file=sys.stderr)
        if PYAV_AVAILABLE and sys.platform == 'win32':
            print(f"  Using PyAV for: {', '.join(PYAV_FORMATS)}", file=sys.stderr)

        self.running = True
        self._preload_models()

    def _preload_models(self):
        """Pre-load models for all worker threads."""
        from concurrent.futures import as_completed

        print(f"Pre-loading models for {self.num_workers} worker(s)...", file=sys.stderr)

        def load_model_for_thread():
            thread_id = threading.current_thread().name
            print(f"[Thread {thread_id}] Loading model...", file=sys.stderr)
            self.thread_local.model = load_model(self.model_path, self.device)
            print(f"[Thread {thread_id}] Model loaded on {self.device}", file=sys.stderr)
            return thread_id

        futures = []
        for i in range(self.num_workers):
            future = self.executor.submit(load_model_for_thread)
            futures.append(future)

        for future in as_completed(futures):
            try:
                thread_id = future.result()
            except Exception as e:
                print(f"Error loading model in thread: {e}", file=sys.stderr)
                raise

        print(f"All {self.num_workers} model(s) loaded successfully", file=sys.stderr)

        # Warmup CQT in main thread to avoid hanging in worker threads
        self._warmup_cqt()

    def _warmup_cqt(self):
        """Warmup librosa.cqt() to initialize BLAS libraries in main thread."""

        # Check if we're running in a frozen executable
        if getattr(sys, 'frozen', False):
            print(f"Running in frozen executable - skipping CQT warmup", file=sys.stderr)
            print(f"  CQT will be initialized on first audio file", file=sys.stderr)
            # In PyInstaller bundles, scipy.stats issues can cause CQT to hang
            # Skip the warmup entirely and let it initialize on first use
            return

        # Normal (non-frozen) warmup
        print(f"Warming up CQT computation...", file=sys.stderr)

        try:
            # Simple warmup - just do a basic FFT operation first
            print(f"  Testing numpy FFT...", file=sys.stderr)
            test_signal = np.zeros(1024, dtype=np.float32)
            test_fft = np.fft.fft(test_signal)
            print(f"  NumPy FFT works", file=sys.stderr)

            # Now try the CQT warmup
            print(f"  Running CQT warmup...", file=sys.stderr)
            warmup_samples = np.zeros(44100, dtype=np.float32)

            # Do the full warmup
            result = librosa.cqt(warmup_samples, sr=44100, hop_length=8820,
                               n_bins=105, bins_per_octave=24, fmin=65)
            print(f"  CQT warmup complete (shape: {result.shape})", file=sys.stderr)

        except Exception as e:
            print(f"ERROR: CQT warmup failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"CQT warmup failed: {e}")

    def get_model(self):
        """Get the model instance for the current thread."""
        if not hasattr(self.thread_local, 'model'):
            thread_id = threading.current_thread().name
            print(f"[Thread {thread_id}] Loading model (fallback)...", file=sys.stderr)
            self.thread_local.model = load_model(self.model_path, self.device)
        return self.thread_local.model

    def send_message(self, message):
        """Send a JSON message to stdout."""
        try:
            json_str = json.dumps(message)
            print(json_str, flush=True)
        except Exception as e:
            print(f"Error sending message: {e}", file=sys.stderr)

    def process_request(self, request):
        """Process a single key detection request."""
        request_id = request.get('id', 'unknown')
        file_path = request.get('path', '')

        profile = os.environ.get('PROFILE_PERFORMANCE', '0') == '1'
        timings = {}
        request_start = time.time()

        try:
            audio_path = Path(file_path)

            if not audio_path.exists():
                return {
                    'id': request_id,
                    'status': 'error',
                    'error': 'File not found',
                    'filename': audio_path.name
                }

            if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
                formats_str = ', '.join(sorted(SUPPORTED_FORMATS))
                return {
                    'id': request_id,
                    'status': 'error',
                    'error': f'Unsupported format. Supported: {formats_str}',
                    'filename': audio_path.name
                }

            # Preprocess audio
            if profile:
                t0 = time.time()
            spec_tensor = preprocess_audio(audio_path)
            if profile:
                timings['preprocess_audio'] = time.time() - t0

            # Move to device and add batch dimension
            if profile:
                t0 = time.time()
            spec_tensor = spec_tensor.to(self.device)
            spec_tensor = spec_tensor.unsqueeze(0)  # Add batch dimension
            if profile:
                timings['to_device'] = time.time() - t0

            # Run inference
            if profile:
                t0 = time.time()
            model = self.get_model()
            with torch.no_grad():
                outputs = model(spec_tensor)
                pred_tensor = torch.argmax(outputs, dim=1)
                pred = int(pred_tensor.cpu().numpy()[0])
            if profile:
                timings['model_inference'] = time.time() - t0

            # Format output
            if profile:
                t0 = time.time()
            camelot_str, key_text = camelot_output(pred)
            openkey_str = camelot_to_openkey(camelot_str)
            if profile:
                timings['format_output'] = time.time() - t0

            if profile:
                total_time = time.time() - request_start
                timings['total'] = total_time
                print(f"[PROFILE] {audio_path.name}:", file=sys.stderr)
                print(f"  preprocess_audio: {timings['preprocess_audio']:.3f}s ({timings['preprocess_audio']/total_time*100:.1f}%)", file=sys.stderr)
                print(f"  to_device: {timings['to_device']:.3f}s ({timings['to_device']/total_time*100:.1f}%)", file=sys.stderr)
                print(f"  model_inference: {timings['model_inference']:.3f}s ({timings['model_inference']/total_time*100:.1f}%)", file=sys.stderr)
                print(f"  format_output: {timings['format_output']:.3f}s ({timings['format_output']/total_time*100:.1f}%)", file=sys.stderr)
                print(f"  TOTAL: {total_time:.3f}s", file=sys.stderr)

            return {
                'id': request_id,
                'status': 'success',
                'camelot': camelot_str,
                'openkey': openkey_str,
                'key': key_text,
                'class_id': pred,
                'filename': audio_path.name
            }

        except Exception as e:
            import traceback
            error_details = f"{str(e)} | {traceback.format_exc()}"
            print(f"[ERROR] Exception in process_request: {error_details}", file=sys.stderr)
            return {
                'id': request_id,
                'status': 'error',
                'error': str(e),
                'filename': Path(file_path).name if file_path else 'unknown'
            }

    def handle_request(self, line):
        """Parse and handle a request line."""
        try:
            request = json.loads(line)
            response = self.process_request(request)
            self.send_message(response)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error handling request: {e}", file=sys.stderr)

    def run(self):
        """Main server loop."""
        self.send_message({'type': 'ready'})
        print("Server ready, waiting for requests...", file=sys.stderr)

        # Start heartbeat thread
        def heartbeat():
            while self.running:
                time.sleep(30)
                if self.running:
                    self.send_message({'type': 'heartbeat'})

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        # Process requests
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                self.executor.submit(self.handle_request, line)

        except KeyboardInterrupt:
            print("Shutting down...", file=sys.stderr)
        finally:
            self.running = False
            self.executor.shutdown(wait=True)
            print("Server stopped", file=sys.stderr)


def main():
    """Entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Key Detection Server")
    parser.add_argument('-m', '--model_path', type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help="Number of worker threads (default: 1)")

    args = parser.parse_args()

    server = KeyDetectionServer(
        model_path=args.model_path,
        device=args.device,
        num_workers=args.workers
    )

    server.run()


if __name__ == '__main__':
    main()
