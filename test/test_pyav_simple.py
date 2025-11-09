#!/usr/bin/env python3
"""Simple test comparing PyAV optimized vs librosa performance."""

import time
import numpy as np
import librosa
import av

def load_pyav_optimized(audio_path, sample_rate=44100):
    """Optimized PyAV loader."""
    container = av.open(str(audio_path))
    audio_stream = container.streams.audio[0]

    resampler = av.AudioResampler(
        format='flt',
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

    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val * 0.95

    return waveform

# Test file
audio_path = r"C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.mp3"

# Run 3 times to account for caching
print("Testing PyAV Optimized vs Librosa (3 runs each)")
print("="*60)

for run in range(3):
    print(f"\nRun {run+1}:")

    # Test PyAV
    t0 = time.time()
    wf_pyav = load_pyav_optimized(audio_path)
    t_load_pyav = time.time() - t0

    t0 = time.time()
    cqt_pyav = librosa.cqt(wf_pyav, sr=44100, hop_length=8820, n_bins=105, bins_per_octave=24, fmin=65)
    t_cqt_pyav = time.time() - t0

    # Test librosa
    t0 = time.time()
    wf_librosa, sr = librosa.load(audio_path, sr=44100, mono=True)
    wf_librosa = wf_librosa.astype(np.float32)
    t_load_librosa = time.time() - t0

    t0 = time.time()
    cqt_librosa = librosa.cqt(wf_librosa, sr=44100, hop_length=8820, n_bins=105, bins_per_octave=24, fmin=65)
    t_cqt_librosa = time.time() - t0

    print(f"  PyAV:    Load={t_load_pyav:.3f}s, CQT={t_cqt_pyav:.3f}s, Total={t_load_pyav+t_cqt_pyav:.3f}s")
    print(f"  Librosa: Load={t_load_librosa:.3f}s, CQT={t_cqt_librosa:.3f}s, Total={t_load_librosa+t_cqt_librosa:.3f}s")

    if run == 0:
        print(f"\n  Waveform comparison:")
        print(f"    PyAV:    shape={wf_pyav.shape}, range=[{wf_pyav.min():.3f}, {wf_pyav.max():.3f}]")
        print(f"    Librosa: shape={wf_librosa.shape}, range=[{wf_librosa.min():.3f}, {wf_librosa.max():.3f}]")

print("\n" + "="*60)
print("Average of last 2 runs (to exclude first-run overhead):")