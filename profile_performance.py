#!/usr/bin/env python3
"""
Performance profiling script for key detection.

Measures timing for each step:
1. Audio loading (librosa.load)
2. CQT computation (librosa.cqt)
3. Preprocessing (log scaling, slicing)
4. Model inference
5. Post-processing (argmax, formatting)

Usage:
    python profile_performance.py path/to/audio.mp3
"""

import sys
import time
import torch
import librosa
import numpy as np
from pathlib import Path

# Import from existing modules
from dataset import CAMELOT_MAPPING
from eval import load_model

# Import PyAV for fast audio loading on Windows
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

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

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 44100)

    Returns:
        np.ndarray: Audio waveform as contiguous float32 array
    """
    container = av.open(str(audio_path))
    audio_stream = container.streams.audio[0]

    # Use non-planar float format for better performance
    resampler = av.AudioResampler(
        format='flt',  # Non-planar float32 (better for CQT)
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

    return waveform


def profile_audio_loading(audio_path, sample_rate=44100):
    """Profile audio loading step."""
    print(f"\n{'='*70}")
    print("STEP 1: Audio Loading")
    print(f"{'='*70}")

    audio_path_obj = Path(audio_path)
    suffix = audio_path_obj.suffix.lower()

    # Use optimized PyAV for compressed formats on Windows
    # Use librosa for native formats (WAV, FLAC, OGG, AIFF) or non-Windows
    use_pyav = (
        sys.platform == 'win32' and
        PYAV_AVAILABLE and
        suffix in {'.mp3', '.mp4', '.m4a', '.aac'}
    )

    start = time.time()

    if use_pyav:
        print(f"Using PyAV OPTIMIZED backend for {suffix}")
        waveform = load_audio_pyav_optimized(audio_path, sample_rate)
        sr = sample_rate  # PyAV returns audio at exact sample rate requested
    else:
        print(f"Using librosa backend for {suffix}")
        waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        waveform = waveform.astype(np.float32)

    elapsed = time.time() - start

    print(f"Duration: {elapsed:.3f}s")
    print(f"Samples: {len(waveform):,}")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio duration: {len(waveform)/sr:.2f}s")

    # Print memory layout info for debugging
    print(f"Memory layout: {'Contiguous' if waveform.flags['C_CONTIGUOUS'] else 'Non-contiguous'}")

    return waveform, sr, elapsed


def profile_cqt_computation(waveform, sample_rate=44100, n_bins=105, hop_length=8820):
    """Profile CQT computation step."""
    print(f"\n{'='*70}")
    print("STEP 2: CQT Computation")
    print(f"{'='*70}")

    start = time.time()
    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length,
                      n_bins=n_bins, bins_per_octave=24, fmin=65)
    elapsed = time.time() - start

    print(f"Duration: {elapsed:.3f}s")
    print(f"CQT shape: {cqt.shape}")
    print(f"Frequency bins: {cqt.shape[0]}")
    print(f"Time frames: {cqt.shape[1]}")

    return cqt, elapsed


def profile_preprocessing(cqt):
    """Profile preprocessing step."""
    print(f"\n{'='*70}")
    print("STEP 3: Preprocessing (Log Scaling + Slicing)")
    print(f"{'='*70}")

    start = time.time()
    spec = np.abs(cqt)
    spec = np.log1p(spec)
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)
    elapsed = time.time() - start

    print(f"Duration: {elapsed:.3f}s")
    print(f"Output shape: {spec_tensor.shape}")

    return spec_tensor, elapsed


def profile_model_loading(model_path, device):
    """Profile model loading step."""
    print(f"\n{'='*70}")
    print("STEP 4: Model Loading")
    print(f"{'='*70}")

    start = time.time()
    model = load_model(model_path, device)
    elapsed = time.time() - start

    print(f"Duration: {elapsed:.3f}s")
    print(f"Device: {device}")

    return model, elapsed


def profile_model_inference(model, spec_tensor, device):
    """Profile model inference step."""
    print(f"\n{'='*70}")
    print("STEP 5: Model Inference")
    print(f"{'='*70}")

    spec_tensor = spec_tensor.to(device)

    # Warmup run (first inference can be slower)
    with torch.no_grad():
        _ = model(spec_tensor.unsqueeze(0))

    # Actual timed run
    start = time.time()
    with torch.no_grad():
        logits = model(spec_tensor.unsqueeze(0))
        pred_class = torch.argmax(logits, dim=1).item()
    elapsed = time.time() - start

    print(f"Duration: {elapsed:.3f}s (after warmup)")
    print(f"Predicted class: {pred_class}")

    return pred_class, elapsed


def profile_post_processing(pred_class):
    """Profile post-processing step."""
    print(f"\n{'='*70}")
    print("STEP 6: Post-Processing (Format Output)")
    print(f"{'='*70}")

    start = time.time()
    idx = (pred_class % 12) + 1
    mode = "A" if pred_class < 12 else "B"
    camelot_str = f"{idx}{mode}"

    names = [k for k, v in CAMELOT_MAPPING.items() if v == pred_class]
    if names:
        key_text = "/".join(sorted(set(names)))
    else:
        key_text = "Unknown"

    elapsed = time.time() - start

    print(f"Duration: {elapsed:.3f}s")
    print(f"Camelot: {camelot_str}")
    print(f"Key: {key_text}")

    return camelot_str, key_text, elapsed


def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_performance.py <audio_file>")
        print("\nExample:")
        print('  python profile_performance.py "C:/Users/Chris/Music/song.mp3"')
        sys.exit(1)

    audio_path = sys.argv[1]
    model_path = get_resource_path('checkpoints/keynet.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'#'*70}")
    print("PERFORMANCE PROFILING - Key Detection Pipeline")
    print(f"{'#'*70}")
    print(f"\nAudio file: {audio_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")

    total_start = time.time()
    timings = {}

    # Step 1: Audio Loading
    waveform, sr, t1 = profile_audio_loading(audio_path)
    timings['audio_loading'] = t1

    # Step 2: CQT Computation
    cqt, t2 = profile_cqt_computation(waveform, sample_rate=sr)
    timings['cqt_computation'] = t2

    # Step 3: Preprocessing
    spec_tensor, t3 = profile_preprocessing(cqt)
    timings['preprocessing'] = t3

    # Step 4: Model Loading
    model, t4 = profile_model_loading(model_path, device)
    timings['model_loading'] = t4

    # Step 5: Model Inference
    pred_class, t5 = profile_model_inference(model, spec_tensor, device)
    timings['model_inference'] = t5

    # Step 6: Post-processing
    camelot_str, key_text, t6 = profile_post_processing(pred_class)
    timings['post_processing'] = t6

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'#'*70}")
    print("TIMING SUMMARY")
    print(f"{'#'*70}")
    print(f"\n{'Step':<30} {'Time (s)':<12} {'Percentage':<12}")
    print(f"{'-'*70}")

    # Exclude model loading from pipeline time (it's one-time overhead)
    pipeline_time = total_elapsed - timings['model_loading']

    for step, duration in timings.items():
        if step == 'model_loading':
            pct = (duration / total_elapsed) * 100
            print(f"{step:<30} {duration:>10.3f}s  {pct:>10.1f}%  (one-time)")
        else:
            pct = (duration / pipeline_time) * 100
            print(f"{step:<30} {duration:>10.3f}s  {pct:>10.1f}%")

    print(f"{'-'*70}")
    print(f"{'Pipeline total (excl. model load)':<30} {pipeline_time:>10.3f}s  {100.0:>10.1f}%")
    print(f"{'Total (incl. model load)':<30} {total_elapsed:>10.3f}s")

    print(f"\n{'#'*70}")
    print(f"RESULT: {camelot_str} ({key_text})")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
