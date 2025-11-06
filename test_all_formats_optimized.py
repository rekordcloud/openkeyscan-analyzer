#!/usr/bin/env python3
"""
Test all audio formats with the optimized PyAV server.
Compares performance between formats and shows which use PyAV vs librosa.
"""

import subprocess
import json
import sys
import os
import time
import threading
from pathlib import Path

# Test files for each format
TEST_FILES = {
    'MP3': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.mp3',
    'MP4': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.mp4',
    'WAV': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.wav',
    'FLAC': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.flac',
    'OGG': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.ogg',
    'M4A': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.m4a',
    'AAC': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.aac',
    'AIFF': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.aiff',
}

# Formats that use PyAV on Windows
PYAV_FORMATS = {'MP3', 'MP4', 'M4A', 'AAC'}


def test_optimized_server():
    """Test the optimized server with all formats."""
    print("="*80)
    print("TESTING OPTIMIZED PYAV SERVER - ALL FORMATS")
    print("="*80)

    # Set profiling environment
    env = os.environ.copy()
    env['PROFILE_PERFORMANCE'] = '1'

    # Start optimized server
    print("\nStarting optimized server with PyAV support...")
    process = subprocess.Popen(
        [sys.executable, "openkeyscan_analyzer_server_optimized.py", '-w', '1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    # Store stderr output for analysis
    stderr_lines = []
    stderr_lock = threading.Lock()

    def read_stderr():
        """Capture stderr for profiling info."""
        try:
            for line in process.stderr:
                with stderr_lock:
                    stderr_lines.append(line.rstrip())
                # Only print server startup messages
                if "Server" in line or "Model" in line or "PyAV" in line:
                    print(f"[SERVER] {line.rstrip()}")
        except:
            pass

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    # Wait for ready
    ready = False
    for line in process.stdout:
        try:
            msg = json.loads(line)
            if msg.get('type') == 'ready':
                ready = True
                print("[SERVER] Ready!")
                break
        except:
            pass

    if not ready:
        print("ERROR: Server failed to start")
        process.kill()
        return {}

    print("\nTesting each audio format...")
    print("-"*80)

    results = {}

    # Test each format
    for format_name, file_path in TEST_FILES.items():
        if not os.path.exists(file_path):
            print(f"\n[SKIP] {format_name}: File not found")
            continue

        # Determine backend
        backend = "PyAV (optimized)" if format_name in PYAV_FORMATS else "librosa"
        print(f"\n[{format_name}] Testing with {backend}...")

        # Send request
        request_id = f"test_{format_name.lower()}"
        request = {'id': request_id, 'path': file_path}

        start_time = time.time()
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()

        # Wait for response
        response = None
        timeout = time.time() + 30  # 30 second timeout

        for line in process.stdout:
            if time.time() > timeout:
                break
            try:
                msg = json.loads(line)
                if msg.get('id') == request_id:
                    response = msg
                    break
            except:
                pass

        elapsed = time.time() - start_time

        # Parse profiling info from stderr
        profile_info = {}
        with stderr_lock:
            # Look for profile info for this file
            for line in stderr_lines:
                if f"{format_name}" in line or "Christian Smith" in line:
                    if "load_audio_pyav_optimized:" in line:
                        parts = line.split(":")[-1].strip()
                        if "s" in parts:
                            profile_info['pyav_load'] = float(parts.replace('s', ''))
                    elif "warmup_cqt:" in line:
                        parts = line.split(":")[-1].strip()
                        if "s" in parts:
                            profile_info['warmup'] = float(parts.split('s')[0])
                    elif "librosa.load:" in line:
                        parts = line.split(":")[-1].strip()
                        if "s" in parts:
                            profile_info['librosa_load'] = float(parts.replace('s', ''))
                    elif "librosa.cqt:" in line:
                        parts = line.split(":")[-1].strip()
                        if "s" in parts:
                            profile_info['cqt'] = float(parts.replace('s', ''))
                    elif "preprocess_audio:" in line:
                        parts = line.split(":")[-1].strip()
                        if "%" in parts:
                            time_str = parts.split('s')[0]
                            profile_info['preprocess_total'] = float(time_str)
                    elif "TOTAL:" in line:
                        parts = line.split(":")[-1].strip()
                        if "s" in parts:
                            profile_info['total'] = float(parts.replace('s', ''))

        # Store results
        if response and response.get('status') == 'success':
            results[format_name] = {
                'backend': backend,
                'time': elapsed,
                'result': response.get('camelot', 'N/A'),
                'key': response.get('key', 'N/A'),
                'profile': profile_info
            }
            print(f"  [SUCCESS] {response['camelot']} ({response['key']})")
            print(f"  Total time: {elapsed:.3f}s")

            # Print detailed timing if available
            if profile_info:
                if 'pyav_load' in profile_info:
                    print(f"  PyAV load: {profile_info.get('pyav_load', 'N/A'):.3f}s")
                    if 'warmup' in profile_info:
                        print(f"  Warmup CQT: {profile_info.get('warmup', 'N/A'):.3f}s (one-time)")
                elif 'librosa_load' in profile_info:
                    print(f"  Librosa load: {profile_info.get('librosa_load', 'N/A'):.3f}s")
                if 'cqt' in profile_info:
                    print(f"  CQT computation: {profile_info.get('cqt', 'N/A'):.3f}s")
        else:
            error = response.get('error', 'No response') if response else 'Timeout'
            results[format_name] = {'error': error}
            print(f"  [ERROR] {error}")

    # Cleanup
    process.stdin.close()
    process.terminate()
    try:
        process.wait(timeout=5)
    except:
        process.kill()

    return results


def print_summary(results):
    """Print performance summary."""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY - OPTIMIZED SERVER")
    print("="*80)

    print("\n" + "-"*80)
    print(f"{'Format':<10} {'Backend':<20} {'Total Time':<12} {'Result':<15} {'Key':<20}")
    print("-"*80)

    pyav_times = []
    librosa_times = []

    for format_name in TEST_FILES.keys():
        if format_name in results:
            r = results[format_name]
            if 'error' not in r:
                backend = r['backend']
                time_str = f"{r['time']:.3f}s"
                result = r.get('result', 'N/A')
                key = r.get('key', 'N/A')[:20]

                print(f"{format_name:<10} {backend:<20} {time_str:<12} {result:<15} {key:<20}")

                # Track times by backend
                if 'PyAV' in backend:
                    pyav_times.append(r['time'])
                else:
                    librosa_times.append(r['time'])
            else:
                print(f"{format_name:<10} {'ERROR':<20} {'N/A':<12} {r['error'][:40]}")
        else:
            print(f"{format_name:<10} {'NOT TESTED':<20}")

    # Calculate averages
    print("\n" + "-"*80)
    print("AVERAGES BY BACKEND:")
    print("-"*80)

    if pyav_times:
        pyav_avg = sum(pyav_times) / len(pyav_times)
        print(f"  PyAV formats (MP3, MP4, M4A, AAC): {pyav_avg:.3f}s average")
        print(f"    - Tested: {len(pyav_times)} formats")

    if librosa_times:
        librosa_avg = sum(librosa_times) / len(librosa_times)
        print(f"  Librosa formats (WAV, FLAC, OGG, AIFF): {librosa_avg:.3f}s average")
        print(f"    - Tested: {len(librosa_times)} formats")

    if pyav_times and librosa_times:
        improvement = (librosa_avg - pyav_avg) / librosa_avg * 100
        print(f"\n  PyAV vs Librosa: {improvement:+.1f}% {'faster' if improvement > 0 else 'slower'}")

    # Note about warmup
    print("\n" + "-"*80)
    print("NOTES:")
    print("-"*80)
    print("- PyAV formats include a one-time warmup CQT (~1.7s) for the first file")
    print("- Subsequent files of the same format skip the warmup")
    print("- All files tested are the same 418-second audio in different formats")
    print("- Server uses 1 worker thread, model loaded once at startup")


def main():
    print("OPTIMIZED PYAV SERVER TEST - ALL AUDIO FORMATS")
    print("Testing with optimized PyAV implementation including warmup")
    print("")

    results = test_optimized_server()
    print_summary(results)


if __name__ == '__main__':
    main()