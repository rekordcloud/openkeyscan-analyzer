#!/usr/bin/env python3
"""
Simple format test script - tests each format individually with profiling.
"""

import subprocess
import json
import os
import time
from pathlib import Path

# Test files
TEST_FILES = {
    "MP3": "/Users/chris/Music/cloud test/Mob Tactics - Now Is the Time.mp3",
    "WAV": "/Users/chris/Music/audio formats/Christian Smith - Burning Chrome.wav",
    "FLAC": "/Users/chris/Music/audio formats/Pleasurekraft - G.O.D. (Gospel of Doubt) ft Casey Gerald - Spektre Remix.flac",
    "M4A": "/Users/chris/Music/audio formats/Knife Party - Bonfire.m4a",
    "OGG": "/Users/chris/Music/audio formats/Burning Chrome - Christian Smith.ogg",
    "AAC": "/Users/chris/Music/audio formats/Burning Chrome - Christian Smith.aac",
    "MP4": "/Users/chris/Music/audio formats/serato.mp4",
}

EXE = "./dist/openkeyscan-analyzer/openkeyscan-analyzer"

def test_format(format_name, file_path):
    """Test a single format."""
    print(f"\n{'='*70}")
    print(f"Testing {format_name}: {Path(file_path).name}")
    print(f"{'='*70}")

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None

    # Set environment
    env = os.environ.copy()
    env['PROFILE_PERFORMANCE'] = '1'

    # Start server
    proc = subprocess.Popen(
        [EXE, '-w', '1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    # Wait for ready
    ready = False
    init_lines = []
    for line in proc.stdout:
        init_lines.append(line.strip())
        if '"type": "ready"' in line:
            ready = True
            break

    if not ready:
        print("❌ Server not ready")
        proc.kill()
        return None

    # Send request
    request = json.dumps({"id": "test", "path": file_path}) + '\n'
    proc.stdin.write(request)
    proc.stdin.flush()

    # Read response and profiling
    result = None
    profile_lines = []

    # Read stderr in background
    import threading
    import queue

    stderr_queue = queue.Queue()

    def read_stderr():
        for line in proc.stderr:
            stderr_queue.put(line.strip())

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    # Wait for response
    timeout = time.time() + 30
    while time.time() < timeout:
        # Check stdout
        if proc.stdout.readable():
            line = proc.stdout.readline()
            if line:
                line = line.strip()
                try:
                    msg = json.loads(line)
                    if msg.get('id') == 'test':
                        result = msg
                        break
                except:
                    pass

    # Get stderr
    time.sleep(0.5)
    while not stderr_queue.empty():
        profile_lines.append(stderr_queue.get())

    # Cleanup
    proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()

    # Print results
    if result and result.get('status') == 'success':
        print(f"✅ Key: {result['camelot']} - {result['key']}")
    elif result:
        print(f"❌ Error: {result.get('error', 'Unknown')}")
    else:
        print(f"❌ No result (timeout)")
        return None

    # Extract profiling
    loading_time = None
    cqt_time = None
    total_time = None
    backend = "Unknown"

    for line in profile_lines:
        if "PySoundFile failed" in line:
            backend = "audioread (fallback)"
        elif "load_audio_pyav:" in line:
            backend = "PyAV"
            try:
                loading_time = float(line.split(':')[1].strip().replace('s', ''))
            except:
                pass
        elif "librosa.load:" in line:
            if backend == "Unknown":
                backend = "librosa (soundfile or audioread)"
            try:
                loading_time = float(line.split(':')[1].strip().replace('s', ''))
            except:
                pass
        elif "librosa.cqt:" in line:
            try:
                cqt_time = float(line.split(':')[1].strip().replace('s', ''))
            except:
                pass
        elif "TOTAL:" in line:
            try:
                total_time = float(line.split(':')[1].strip().replace('s', ''))
            except:
                pass

    # Print performance
    if loading_time and cqt_time and total_time:
        loading_pct = (loading_time / total_time * 100) if total_time > 0 else 0
        cqt_pct = (cqt_time / total_time * 100) if total_time > 0 else 0
        print(f"\n   Backend: {backend}")
        print(f"   Loading: {loading_time:.3f}s ({loading_pct:.1f}%)")
        print(f"   CQT:     {cqt_time:.3f}s ({cqt_pct:.1f}%)")
        print(f"   Total:   {total_time:.3f}s")
    else:
        print(f"   ⚠️  No profiling data captured")

    return {
        'format': format_name,
        'loading_time': loading_time,
        'cqt_time': cqt_time,
        'total_time': total_time,
        'backend': backend
    }

def main():
    print("\n" + "="*70)
    print("AUDIO FORMAT PERFORMANCE TEST - Executable (macOS)")
    print("="*70)

    results = []
    for format_name, file_path in TEST_FILES.items():
        result = test_format(format_name, file_path)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results:
        print(f"\n{'Format':<10} {'Loading':<12} {'CQT':<12} {'Total':<12} {'Backend':<25}")
        print("-"*70)

        for r in results:
            loading_str = f"{r['loading_time']:.3f}s" if r['loading_time'] else "N/A"
            cqt_str = f"{r['cqt_time']:.3f}s" if r['cqt_time'] else "N/A"
            total_str = f"{r['total_time']:.3f}s" if r['total_time'] else "N/A"
            print(f"{r['format']:<10} {loading_str:<12} {cqt_str:<12} {total_str:<12} {r['backend']:<25}")

        # Average
        valid = [r for r in results if r['total_time']]
        if valid:
            avg_loading = sum(r['loading_time'] for r in valid if r['loading_time']) / len([r for r in valid if r['loading_time']])
            avg_total = sum(r['total_time'] for r in valid) / len(valid)
            print("-"*70)
            print(f"{'Average':<10} {avg_loading:.3f}s{' '*7} {avg_total:.3f}s")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
