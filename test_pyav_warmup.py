#!/usr/bin/env python3
"""Test PyAV with warmup to eliminate first-run CQT overhead."""

import time
import numpy as np
import librosa
import av

def load_pyav_with_warmup(audio_path, sample_rate=44100):
    """PyAV loader with CQT warmup."""
    # Load with PyAV
    container = av.open(str(audio_path))
    audio_stream = container.streams.audio[0]

    resampler = av.AudioResampler(
        format='flt',  # Non-planar float32
        layout='mono',
        rate=sample_rate
    )

    frames = []
    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            for resampled in resampled_frames:
                frames.append(resampled.to_ndarray())

    remaining = resampler.resample(None)
    if remaining:
        for resampled in remaining:
            frames.append(resampled.to_ndarray())

    container.close()

    waveform = np.concatenate(frames, axis=1)
    if waveform.ndim > 1:
        waveform = waveform[0]

    # Ensure contiguous
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    # Normalize
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val * 0.95

    # WARMUP CQT - eliminates first-run overhead
    if len(waveform) > 44100:
        print("  Running CQT warmup...")
        warmup_start = time.time()
        warmup_samples = waveform[:44100]  # 1 second
        _ = librosa.cqt(warmup_samples, sr=sample_rate, hop_length=8820,
                        n_bins=24, bins_per_octave=12, fmin=65)
        print(f"  Warmup completed in {time.time() - warmup_start:.3f}s")

    return waveform


# Test file
audio_path = r"C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.mp3"

print("Testing PyAV with warmup CQT")
print("="*60)

# Load with warmup
print("\nLoading with PyAV + warmup...")
t0 = time.time()
waveform = load_pyav_with_warmup(audio_path)
load_time = time.time() - t0
print(f"Total load time (including warmup): {load_time:.3f}s")

# Test CQT performance (should be fast now)
print("\nTesting CQT performance after warmup:")
for i in range(3):
    t0 = time.time()
    cqt = librosa.cqt(waveform, sr=44100, hop_length=8820,
                      n_bins=105, bins_per_octave=24, fmin=65)
    cqt_time = time.time() - t0
    print(f"  Run {i+1}: {cqt_time:.3f}s")

print("\n" + "="*60)
print("✅ If all CQT runs are ~0.4s, warmup is working!")