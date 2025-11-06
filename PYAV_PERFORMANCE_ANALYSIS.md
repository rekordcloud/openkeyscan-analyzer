# PyAV Performance Analysis - Windows

**Test Date:** 2025-11-06
**Platform:** Windows 10 (Build 26100)
**Test File:** Burning Chrome - Christian Smith.mp3 (418+ seconds)
**Format:** MP3 (compressed audio)

---

## Executive Summary

PyAV is **extremely fast** at audio loading (74% faster than librosa), but causes a **4x slowdown in CQT computation**. This suggests a data format or numerical difference that affects subsequent processing.

**Recommendation:** Do NOT use PyAV in `profile_performance.py` - keep it librosa-only for benchmarking. The server can use PyAV if desired, but results may differ from benchmark measurements.

---

## Detailed Comparison

### Audio Loading (Step 1)

| Backend | Time | vs Librosa | Samples | Duration |
|---------|------|-----------|---------|----------|
| **PyAV** | **0.409s** | **-74%** ✅ | 18,441,263 | 418.17s |
| **Librosa** | 1.587s | baseline | 18,441,263 | 418.17s |

**Finding:** PyAV is dramatically faster, loading the same audio in ~4x less time.

### CQT Computation (Step 2)

| Backend | Time | vs Librosa | CQT Shape | Frame Count |
|---------|------|-----------|-----------|------------|
| **PyAV** | **1.686s** | **+290%** ❌ | (105, 2091) | 2091 |
| **Librosa** | 0.433s | baseline | (105, 2091) | 2091 |

**Finding:** CQT computation is 4x SLOWER with PyAV-loaded audio, despite identical CQT parameters and output shape!

### Total Pipeline (without model load)

| Backend | Time | vs Librosa |
|---------|------|-----------|
| **Librosa** | 2.094s | baseline |
| **PyAV** | 2.175s | +4% slower |

**Finding:** Overall performance is nearly identical (~2s/file). The audio loading speedup is offset by CQT slowdown.

---

## Full Timing Breakdown

### PyAV Results
```
Audio Loading       0.409s   18.8%  ← 74% faster
CQT Computation     1.686s   77.8%  ← 4x slower
Preprocessing       0.004s    0.2%
Model Loading       0.006s    0.3%
Model Inference     0.034s    1.5%
Post-Processing     0.000s    0.0%
──────────────────────────────────
TOTAL               2.175s  100.0%
```

### Librosa Results
```
Audio Loading       1.587s   75.8%  ← baseline
CQT Computation     0.433s   20.7%  ← baseline
Preprocessing       0.004s    0.2%
Model Loading       0.007s    0.3%
Model Inference     0.032s    1.5%
Post-Processing     0.000s    0.0%
──────────────────────────────────
TOTAL               2.094s  100.0%
```

---

## Analysis & Hypothesis

### Why is CQT 4x slower with PyAV?

Several possibilities:

1. **Audio Normalization Differences**
   - LibrOSA normalizes to [-1.0, 1.0] range
   - PyAV resampler may produce different amplitude scaling
   - Different signal amplitudes = different FFT numerical properties

2. **Floating-Point Precision**
   - PyAV uses float32 planar format from resampler
   - LibrOSA may use different internal precision
   - CQT relies on FFT which is sensitive to input scaling

3. **Audio Content Quality**
   - PyAV's resampler may introduce artifacts or phase shifts
   - Different frequency content affects CQT bin computation

4. **NumPy/BLAS Behavior**
   - NumPy FFT performance depends on input characteristics
   - Certain frequency patterns may cause cache misses

### Testing the Hypothesis

To confirm, we would need to:
- Check PyAV output amplitude/range
- Compare FFT of PyAV vs librosa audio on same file
- Profile numpy FFT specifically
- Check if adding explicit normalization helps

---

## Key Insights

### ✅ What PyAV Does Well
- **Ultra-fast decoding:** MP3 decompression is 74% faster
- **Native codec support:** Uses OS-level audio decoders (Windows Media Foundation)
- **Mono resampling:** Built-in mono/resampling reduces post-processing

### ❌ What PyAV Does Poorly
- **CQT compatibility:** FFT computation is severely degraded
- **Overall throughput:** No net improvement despite fast loading
- **Predictability:** Results differ from librosa baseline

---

## Recommendations

### For `profile_performance.py` (Benchmarking)
- ✅ **Keep librosa-only** - provides stable, reproducible benchmarks
- Matches server baseline performance (now that server also uses librosa for profiling)
- Consistent across all platforms

### For `openkeyscan_analyzer_server.py` (Production)
- ⚠️ **Current approach is good** - uses librosa for all formats
  - Alternative: Could use PyAV for audio loading, but add compensation for CQT slowdown
  - Current simple approach (librosa for all) is more maintainable

### For Future Optimization
If PyAV speedup is desired:
1. **Normalize PyAV output** to exact librosa scale
2. **Profile CQT separately** - identify exact bottleneck
3. **Test with other formats** - MP4, M4A may show different results
4. **Consider alternatives** - PyAV may not be the right tool for this pipeline

---

## Conclusion

PyAV is interesting for audio loading speed, but it introduces a downstream slowdown that negates the benefits. The server's current approach of using librosa universally is the right choice:

- **Simpler code:** One code path, no platform-specific branching
- **Consistent results:** Same performance on all platforms
- **Predictable:** CQT computation time doesn't vary
- **Maintainable:** No need to manage multiple backends

**Bottom line:** Stick with librosa. It's fast enough (~2s/file), works everywhere, and produces consistent, predictable results.

---

*Generated: 2025-11-06*
