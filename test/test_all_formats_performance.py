#!/usr/bin/env python3
"""
Test performance of audio loading and processing for all supported formats.
Tests both individual file processing and server mode with profiling enabled.
"""

import os
import sys
import time
import json
import subprocess
import uuid
import threading
from pathlib import Path

# Map of format to example files
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

# Note: .au format not commonly available, excluding from test

def test_individual_files():
    """Test each format individually using profile_performance.py"""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL FILES WITH DETAILED PROFILING")
    print("="*80)

    results = {}

    for format_name, file_path in TEST_FILES.items():
        if not os.path.exists(file_path):
            print(f"\n[SKIP] {format_name}: File not found - {file_path}")
            continue

        print(f"\n[TESTING {format_name}]")
        print("-" * 40)

        # Run the profiling script
        try:
            cmd = [sys.executable, "../profile_performance.py", file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parse timing from output
            output = result.stdout

            # Extract key timings
            timings = {}
            for line in output.split('\n'):
                if 'audio_loading' in line and 's' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith('s'):
                            try:
                                timings['audio_loading'] = float(part[:-1])
                            except:
                                pass
                elif 'cqt_computation' in line and 's' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith('s'):
                            try:
                                timings['cqt_computation'] = float(part[:-1])
                            except:
                                pass
                elif 'Pipeline total' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith('s'):
                            try:
                                timings['pipeline_total'] = float(part[:-1])
                            except:
                                pass

            # Find result
            result_key = None
            for line in output.split('\n'):
                if 'RESULT:' in line:
                    result_key = line.split('RESULT:')[1].strip()
                    break

            results[format_name] = {
                'timings': timings,
                'result': result_key,
                'success': result.returncode == 0
            }

            # Print summary for this format
            if timings:
                print(f"  Audio Loading: {timings.get('audio_loading', 'N/A'):.3f}s")
                print(f"  CQT Computation: {timings.get('cqt_computation', 'N/A'):.3f}s")
                print(f"  Pipeline Total: {timings.get('pipeline_total', 'N/A'):.3f}s")
                if result_key:
                    print(f"  Result: {result_key}")
            else:
                print("  [ERROR] Could not parse timings from output")
                print("  Output snippet:", output[:500])

        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Timeout after 120 seconds")
            results[format_name] = {'error': 'timeout'}
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[format_name] = {'error': str(e)}

    return results


def test_server_mode():
    """Test server mode with all formats and profiling enabled"""
    print("\n" + "="*80)
    print("TESTING SERVER MODE WITH PROFILING")
    print("="*80)

    # Start server with profiling enabled
    env = os.environ.copy()
    env['PROFILE_PERFORMANCE'] = '1'

    print("\nStarting server with PROFILE_PERFORMANCE=1...")
    process = subprocess.Popen(
        [sys.executable, "../openkeyscan_analyzer_server.py", '-w', '1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    results = {}
    ready = False

    def read_stdout():
        nonlocal ready
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get('type') == 'ready':
                        ready = True
                    elif 'id' in msg:
                        results[msg['id']] = msg
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error reading stdout: {e}")

    def read_stderr():
        """Capture profiling information from stderr"""
        try:
            for line in process.stderr:
                # Print profiling information
                if 'PROFILE' in line or 'Loading:' in line or 'CQT:' in line or 'Total:' in line:
                    print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"Error reading stderr: {e}")

    # Start reader threads
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    # Wait for ready
    timeout = time.time() + 15
    while not ready and time.time() < timeout:
        time.sleep(0.1)

    if not ready:
        print("ERROR: Server did not become ready")
        process.kill()
        return {}

    print("Server ready! Testing each format...")

    # Test each format
    requests = {}
    server_results = {}

    for format_name, file_path in TEST_FILES.items():
        if not os.path.exists(file_path):
            print(f"\n[SKIP] {format_name}: File not found")
            continue

        print(f"\n[Testing {format_name}]")

        req_id = str(uuid.uuid4())
        requests[req_id] = format_name

        request = {'id': req_id, 'path': file_path}
        start_time = time.time()

        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()

        # Wait for result
        timeout = time.time() + 60
        while req_id not in results and time.time() < timeout:
            time.sleep(0.1)

        elapsed = time.time() - start_time

        if req_id in results:
            result = results[req_id]
            server_results[format_name] = {
                'time': elapsed,
                'status': result.get('status'),
                'result': result.get('camelot', 'N/A'),
                'error': result.get('error')
            }

            if result.get('status') == 'success':
                print(f"  Result: {result.get('camelot')} ({result.get('key')})")
                print(f"  Server processing time: {elapsed:.3f}s")
            else:
                print(f"  Error: {result.get('error')}")
        else:
            print(f"  [TIMEOUT] No response after 60s")
            server_results[format_name] = {'error': 'timeout'}

    # Cleanup
    process.stdin.close()
    process.terminate()
    process.wait(timeout=5)

    return server_results


def print_summary(individual_results, server_results):
    """Print summary of all tests"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY - ALL FORMATS")
    print("="*80)

    print("\n" + "-"*80)
    print("Individual File Processing (with detailed profiling):")
    print("-"*80)
    print(f"{'Format':<10} {'Audio Load':<12} {'CQT Comp':<12} {'Pipeline':<12} {'Result':<20}")
    print("-"*80)

    for format_name in TEST_FILES.keys():
        if format_name in individual_results:
            r = individual_results[format_name]
            if 'timings' in r:
                t = r['timings']
                audio = f"{t.get('audio_loading', 0):.3f}s" if 'audio_loading' in t else "N/A"
                cqt = f"{t.get('cqt_computation', 0):.3f}s" if 'cqt_computation' in t else "N/A"
                pipeline = f"{t.get('pipeline_total', 0):.3f}s" if 'pipeline_total' in t else "N/A"
                result = r.get('result', 'N/A')[:20]
                print(f"{format_name:<10} {audio:<12} {cqt:<12} {pipeline:<12} {result:<20}")
            else:
                print(f"{format_name:<10} ERROR: {r.get('error', 'Unknown')}")
        else:
            print(f"{format_name:<10} NOT TESTED")

    print("\n" + "-"*80)
    print("Server Mode Processing:")
    print("-"*80)
    print(f"{'Format':<10} {'Time':<12} {'Result':<20} {'Status':<10}")
    print("-"*80)

    for format_name in TEST_FILES.keys():
        if format_name in server_results:
            r = server_results[format_name]
            time_str = f"{r.get('time', 0):.3f}s" if 'time' in r else "N/A"
            result = r.get('result', 'N/A')
            status = r.get('status', r.get('error', 'error'))
            print(f"{format_name:<10} {time_str:<12} {result:<20} {status:<10}")
        else:
            print(f"{format_name:<10} NOT TESTED")

    # Calculate averages
    print("\n" + "-"*80)
    print("Averages:")
    print("-"*80)

    # Individual averages
    audio_times = []
    cqt_times = []
    pipeline_times = []

    for r in individual_results.values():
        if 'timings' in r:
            t = r['timings']
            if 'audio_loading' in t:
                audio_times.append(t['audio_loading'])
            if 'cqt_computation' in t:
                cqt_times.append(t['cqt_computation'])
            if 'pipeline_total' in t:
                pipeline_times.append(t['pipeline_total'])

    if audio_times:
        print(f"  Avg Audio Loading: {sum(audio_times)/len(audio_times):.3f}s")
    if cqt_times:
        print(f"  Avg CQT Computation: {sum(cqt_times)/len(cqt_times):.3f}s")
    if pipeline_times:
        print(f"  Avg Pipeline Total: {sum(pipeline_times)/len(pipeline_times):.3f}s")

    # Server mode average
    server_times = [r['time'] for r in server_results.values() if 'time' in r]
    if server_times:
        print(f"  Avg Server Processing: {sum(server_times)/len(server_times):.3f}s")


def main():
    print("="*80)
    print("AUDIO FORMAT PERFORMANCE TEST")
    print("Testing all supported formats on Windows")
    print("="*80)

    # Test individual files with detailed profiling
    individual_results = test_individual_files()

    # Test server mode
    server_results = test_server_mode()

    # Print summary
    print_summary(individual_results, server_results)


if __name__ == '__main__':
    main()