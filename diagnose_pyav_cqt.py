#!/usr/bin/env python3
"""
Diagnose why CQT is slower with PyAV-loaded audio.
Compare PyAV vs librosa audio loading and test different optimizations.
"""

import sys
import time
import numpy as np
import librosa
from pathlib import Path

# Import PyAV
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    print("PyAV not available, install with: pip install av")
    sys.exit(1)


def load_audio_pyav_original(audio_path, sample_rate=44100):
    """Original PyAV loading implementation."""
    container = av.open(str(audio_path))
    audio_stream = container.streams.audio[0]

    resampler = av.AudioResampler(
        format='fltp',  # float32 planar
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

    remaining = resampler.resample(None)
    if remaining:
        for resampled in remaining:
            frame_array = resampled.to_ndarray()
            frames.append(frame_array)

    container.close()

    waveform = np.concatenate(frames, axis=1)
    if waveform.ndim > 1:
        waveform = waveform[0]

    waveform = waveform.astype(np.float32)

    # Original normalization
    max_val = np.abs(waveform).max()
    if max_val > 1.0:
        waveform = waveform / max_val

    return waveform


def load_audio_pyav_optimized(audio_path, sample_rate=44100):
    """Optimized PyAV loading with better normalization."""
    container = av.open(str(audio_path))
    audio_stream = container.streams.audio[0]

    # Try different resampler settings
    resampler = av.AudioResampler(
        format='flt',  # Try non-planar float32
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

    remaining = resampler.resample(None)
    if remaining:
        for resampled in remaining:
            frame_array = resampled.to_ndarray()
            frames.append(frame_array)

    container.close()

    waveform = np.concatenate(frames, axis=1)
    if waveform.ndim > 1:
        waveform = waveform[0]

    # Convert to float32 and ensure contiguous memory layout
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    # Match librosa's normalization more closely
    # librosa normalizes to [-1, 1] range during loading
    # Don't renormalize if already in range
    max_val = np.abs(waveform).max()
    if max_val > 0:
        # Scale to match typical librosa output amplitude
        waveform = waveform / max_val * 0.95  # Slight headroom like librosa

    return waveform


def load_audio_pyav_fast_decode(audio_path, sample_rate=44100):
    """PyAV with faster decoding strategy - decode all at once."""
    container = av.open(str(audio_path))
    audio_stream = container.streams.audio[0]

    # Decode entire audio at once
    packets = container.demux(audio_stream)
    frames_raw = []

    for packet in packets:
        for frame in packet.decode():
            frames_raw.append(frame)

    container.close()

    # Now resample all frames
    resampler = av.AudioResampler(
        format='flt',
        layout='mono',
        rate=sample_rate
    )

    frames = []
    for frame in frames_raw:
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            for resampled in resampled_frames:
                frame_array = resampled.to_ndarray()
                frames.append(frame_array)

    # Flush resampler
    remaining = resampler.resample(None)
    if remaining:
        for resampled in remaining:
            frame_array = resampled.to_ndarray()
            frames.append(frame_array)

    waveform = np.concatenate(frames, axis=1)
    if waveform.ndim > 1:
        waveform = waveform[0]

    # Ensure contiguous float32
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    # Normalize
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val * 0.95

    return waveform


def compute_cqt_with_timing(waveform, sample_rate=44100):
    """Compute CQT and return timing."""
    start = time.time()
    cqt = librosa.cqt(
        waveform,
        sr=sample_rate,
        hop_length=8820,
        n_bins=105,
        bins_per_octave=24,
        fmin=65
    )
    elapsed = time.time() - start
    return cqt, elapsed


def compare_waveforms(wf1, wf2, label1="Waveform 1", label2="Waveform 2"):
    """Compare two waveforms and print statistics."""
    print(f"\n{'-'*60}")
    print(f"Comparing {label1} vs {label2}")
    print(f"{'-'*60}")

    # Basic stats
    print(f"{label1}:")
    print(f"  Shape: {wf1.shape}")
    print(f"  Dtype: {wf1.dtype}")
    print(f"  Min: {wf1.min():.6f}, Max: {wf1.max():.6f}")
    print(f"  Mean: {wf1.mean():.6f}, Std: {wf1.std():.6f}")
    print(f"  Contiguous: {wf1.flags['C_CONTIGUOUS']}")

    print(f"\n{label2}:")
    print(f"  Shape: {wf2.shape}")
    print(f"  Dtype: {wf2.dtype}")
    print(f"  Min: {wf2.min():.6f}, Max: {wf2.max():.6f}")
    print(f"  Mean: {wf2.mean():.6f}, Std: {wf2.std():.6f}")
    print(f"  Contiguous: {wf2.flags['C_CONTIGUOUS']}")

    # Compare if same length
    if wf1.shape[0] == wf2.shape[0]:
        diff = np.abs(wf1 - wf2)
        print(f"\nDifferences:")
        print(f"  Max absolute diff: {diff.max():.6f}")
        print(f"  Mean absolute diff: {diff.mean():.6f}")
        print(f"  RMS diff: {np.sqrt(np.mean(diff**2)):.6f}")
    else:
        print(f"\nDifferent lengths: {wf1.shape[0]} vs {wf2.shape[0]}")


def test_optimized_cqt(waveform, sample_rate=44100):
    """Test CQT with different optimization strategies."""
    print(f"\n{'='*60}")
    print("Testing CQT optimization strategies")
    print(f"{'='*60}")

    # Strategy 1: Ensure contiguous array
    if not waveform.flags['C_CONTIGUOUS']:
        print("Making waveform contiguous...")
        waveform_contig = np.ascontiguousarray(waveform, dtype=np.float32)
    else:
        waveform_contig = waveform

    # Strategy 2: Pre-pad to power of 2 length for FFT efficiency
    n_samples = len(waveform_contig)
    next_pow2 = 2**int(np.ceil(np.log2(n_samples)))
    if next_pow2 > n_samples:
        print(f"Padding from {n_samples} to {next_pow2} samples...")
        waveform_padded = np.pad(waveform_contig, (0, next_pow2 - n_samples), mode='constant')
    else:
        waveform_padded = waveform_contig

    # Test different strategies
    strategies = [
        ("Original", waveform),
        ("Contiguous", waveform_contig),
        ("Power-of-2 padded", waveform_padded),
    ]

    for name, wf in strategies:
        print(f"\n{name}:")
        cqt, elapsed = compute_cqt_with_timing(wf, sample_rate)
        print(f"  CQT time: {elapsed:.3f}s")
        print(f"  CQT shape: {cqt.shape}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_pyav_cqt.py <audio_file>")
        print('Example: python diagnose_pyav_cqt.py "C:/Users/Chris/Music/audio formats/Burning Chrome - Christian Smith.mp3"')
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Testing file: {audio_path}")

    # Load with librosa (baseline)
    print("\n" + "="*60)
    print("LIBROSA LOADING (Baseline)")
    print("="*60)
    start = time.time()
    wf_librosa, sr = librosa.load(audio_path, sr=44100, mono=True)
    wf_librosa = wf_librosa.astype(np.float32)
    librosa_load_time = time.time() - start
    print(f"Load time: {librosa_load_time:.3f}s")

    # CQT with librosa audio
    cqt_librosa, cqt_librosa_time = compute_cqt_with_timing(wf_librosa)
    print(f"CQT time: {cqt_librosa_time:.3f}s")

    # Load with PyAV original
    print("\n" + "="*60)
    print("PYAV LOADING (Original)")
    print("="*60)
    start = time.time()
    wf_pyav_orig = load_audio_pyav_original(audio_path)
    pyav_orig_load_time = time.time() - start
    print(f"Load time: {pyav_orig_load_time:.3f}s")

    # CQT with original PyAV audio
    cqt_pyav_orig, cqt_pyav_orig_time = compute_cqt_with_timing(wf_pyav_orig)
    print(f"CQT time: {cqt_pyav_orig_time:.3f}s")

    # Compare waveforms
    compare_waveforms(wf_librosa, wf_pyav_orig, "Librosa", "PyAV Original")

    # Load with PyAV optimized
    print("\n" + "="*60)
    print("PYAV LOADING (Optimized)")
    print("="*60)
    start = time.time()
    wf_pyav_opt = load_audio_pyav_optimized(audio_path)
    pyav_opt_load_time = time.time() - start
    print(f"Load time: {pyav_opt_load_time:.3f}s")

    # CQT with optimized PyAV audio
    cqt_pyav_opt, cqt_pyav_opt_time = compute_cqt_with_timing(wf_pyav_opt)
    print(f"CQT time: {cqt_pyav_opt_time:.3f}s")

    # Compare waveforms
    compare_waveforms(wf_librosa, wf_pyav_opt, "Librosa", "PyAV Optimized")

    # Load with PyAV fast decode
    print("\n" + "="*60)
    print("PYAV LOADING (Fast Decode)")
    print("="*60)
    start = time.time()
    wf_pyav_fast = load_audio_pyav_fast_decode(audio_path)
    pyav_fast_load_time = time.time() - start
    print(f"Load time: {pyav_fast_load_time:.3f}s")

    # CQT with fast decode PyAV audio
    cqt_pyav_fast, cqt_pyav_fast_time = compute_cqt_with_timing(wf_pyav_fast)
    print(f"CQT time: {cqt_pyav_fast_time:.3f}s")

    # Test CQT optimizations on PyAV audio
    test_optimized_cqt(wf_pyav_opt)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Load Time':<12} {'CQT Time':<12} {'Total':<12}")
    print("-"*60)

    total_librosa = librosa_load_time + cqt_librosa_time
    print(f"{'Librosa':<20} {librosa_load_time:>10.3f}s {cqt_librosa_time:>10.3f}s {total_librosa:>10.3f}s")

    total_pyav_orig = pyav_orig_load_time + cqt_pyav_orig_time
    print(f"{'PyAV Original':<20} {pyav_orig_load_time:>10.3f}s {cqt_pyav_orig_time:>10.3f}s {total_pyav_orig:>10.3f}s")

    total_pyav_opt = pyav_opt_load_time + cqt_pyav_opt_time
    print(f"{'PyAV Optimized':<20} {pyav_opt_load_time:>10.3f}s {cqt_pyav_opt_time:>10.3f}s {total_pyav_opt:>10.3f}s")

    total_pyav_fast = pyav_fast_load_time + cqt_pyav_fast_time
    print(f"{'PyAV Fast Decode':<20} {pyav_fast_load_time:>10.3f}s {cqt_pyav_fast_time:>10.3f}s {total_pyav_fast:>10.3f}s")

    # Best improvement
    best_total = min(total_pyav_orig, total_pyav_opt, total_pyav_fast)
    improvement = (total_librosa - best_total) / total_librosa * 100
    print(f"\nBest PyAV total: {best_total:.3f}s")
    print(f"Improvement vs librosa: {improvement:.1f}%")


if __name__ == '__main__':
    main()