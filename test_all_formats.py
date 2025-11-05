#!/usr/bin/env python3
"""
Test all 9 supported audio formats with profiling.

Tests formats: MP3, MP4, WAV, FLAC, OGG, M4A, AAC, AIFF, AU
Measures performance and identifies which backend is used (soundfile vs audioread vs PyAV)
Works on both macOS and Windows.
"""

import subprocess
import json
import os
import sys
import threading
import queue
import time
from pathlib import Path


def find_test_files():
    """Find test files in Music directory for each format."""
    music_dir = Path.home() / 'Music'
    test_files = {}
    
    # Search patterns for each format
    patterns = {
        'MP3': '**/*.mp3',
        'MP4': '**/*.mp4',
        'WAV': '**/*.wav',
        'FLAC': '**/*.flac',
        'OGG': '**/*.ogg',
        'M4A': '**/*.m4a',
        'AAC': '**/*.aac',
        'AIFF': '**/*.aiff',
        'AU': '**/*.au',
    }
    
    print(f"Searching for test files in {music_dir}...")
    found_count = 0
    
    for format_name, pattern in patterns.items():
        matches = list(music_dir.glob(pattern))
        if matches:
            # Use the first match found
            test_files[format_name] = str(matches[0])
            found_count += 1
            print(f"  ✓ Found {format_name}: {Path(matches[0]).name}")
        else:
            print(f"  ✗ No {format_name} files found")
    
    print(f"\nFound {found_count}/{len(patterns)} formats\n")
    return test_files


def test_format(format_name, file_path, use_exe=False):
    """Test a single audio format."""
    
    if not Path(file_path).exists():
        print(f"⚠️  File not found: {file_path}")
        return None

    print(f"\n{'='*70}")
    print(f"Testing {format_name}: {Path(file_path).name}")
    print(f"{'='*70}")

    # Set environment for profiling
    env = os.environ.copy()
    env['PROFILE_PERFORMANCE'] = '1'

    # Choose command (Python or executable)
    if use_exe:
        if sys.platform == 'win32':
            exe_name = 'openkeyscan-analyzer.exe'
        else:
            exe_name = 'openkeyscan-analyzer'
        cmd = [
            os.path.abspath(f'dist/openkeyscan-analyzer/{exe_name}'),
            '-w', '1'
        ]
    else:
        # Use pipenv run python on both platforms
        if sys.platform == 'win32':
            cmd = ['pipenv', 'run', 'python', 'openkeyscan_analyzer_server.py', '-w', '1']
        else:
            cmd = ['pipenv', 'run', 'python', 'openkeyscan_analyzer_server.py', '-w', '1']

    # Start server
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
    except FileNotFoundError as e:
        print(f"❌ Error: Could not start process: {e}")
        print(f"   Command: {' '.join(cmd)}")
        return None

    # Wait for ready message (read from both stdout and stderr)
    init_messages = []
    msg_queue = queue.Queue()
    ready_event = threading.Event()
    
    def read_stderr():
        try:
            for line in process.stderr:
                line = line.rstrip()
                init_messages.append(line)
                msg_queue.put(('stderr', line))
                if "Server ready" in line or '"type": "ready"' in line:
                    ready_event.set()
                    break
        except Exception as e:
            print(f"Error reading stderr: {e}")
    
    def read_stdout():
        try:
            for line in process.stdout:
                line = line.rstrip()
                msg_queue.put(('stdout', line))
                if '"type": "ready"' in line:
                    ready_event.set()
                    break
        except Exception as e:
            print(f"Error reading stdout: {e}")
    
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread.start()
    stdout_thread.start()
    
    # Wait for ready with timeout
    if not ready_event.wait(timeout=15):
        print("⚠️  Server did not become ready in 15 seconds")
        process.kill()
        return None

    # Send request
    request = {'id': 'test', 'path': file_path}
    try:
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
    except Exception as e:
        print(f"❌ Error sending request: {e}")
        process.kill()
        return None

    # Read response and profiling output
    profile_lines = []
    result = None
    
    # Continue reading from queue
    timeout = time.time() + 120  # 2 minute timeout (Windows can be slow)
    while time.time() < timeout:
        try:
            source, line = msg_queue.get(timeout=2)
            if source == 'stderr':
                # Capture all profiling lines
                if any(keyword in line for keyword in [
                    'librosa.load:', 'load_audio_pyav:', 'librosa.cqt:', 
                    'TOTAL:', '[PROFILE]', '[FFMPEG]', '[THREADING]'
                ]):
                    profile_lines.append(line)
                if 'TOTAL:' in line:
                    break
            elif source == 'stdout':
                try:
                    msg = json.loads(line.strip())
                    if msg.get('id') == 'test':
                        result = msg
                        # Continue reading to get profiling output
                        if 'TOTAL:' in '\n'.join(profile_lines):
                            break
                except json.JSONDecodeError:
                    pass
        except queue.Empty:
            # Check if we have result and profiling
            if result and 'TOTAL:' in '\n'.join(profile_lines):
                break
            # Give it more time if we don't have both yet
            continue
    
    # Check for backend detection in init messages
    backend_info = {
        'pyav_detected': any('load_audio_pyav' in msg for msg in init_messages + profile_lines),
        'librosa_detected': any('librosa.load' in msg for msg in profile_lines),
        'ffmpeg_detected': any('[FFMPEG]' in msg for msg in init_messages),
        'platform': sys.platform
    }

    # Print results
    if result and result.get('status') == 'success':
        print(f"✅ Status: {result['status']}")
        print(f"   Key: {result['camelot']} - {result['key']}")
        if 'openkey' in result:
            print(f"   Open Key: {result['openkey']}")
    elif result:
        print(f"❌ Status: {result['status']} - {result.get('error', 'Unknown error')}")
    else:
        print("❌ No result received (timeout or error)")

    # Print backend detection
    backend_str = []
    if backend_info['pyav_detected']:
        backend_str.append("PyAV")
    if backend_info['librosa_detected']:
        backend_str.append("librosa")
    if backend_info['ffmpeg_detected']:
        backend_str.append("FFmpeg")
    
    if backend_str:
        print(f"   Backend: {' + '.join(backend_str)}")
    else:
        print(f"   Backend: Unknown")

    # Print profiling
    if profile_lines:
        print("\n   Performance:")
        loading_time = None
        cqt_time = None
        total_time = None
        
        for line in profile_lines:
            if 'load_audio_pyav:' in line:
                # Extract timing: "    - load_audio_pyav: 1.234s"
                try:
                    parts = line.split('load_audio_pyav:')
                    if len(parts) > 1:
                        loading_time = float(parts[1].strip().replace('s', ''))
                        print(f"     - Audio loading (PyAV): {parts[1].strip()}")
                except:
                    pass
            elif 'librosa.load:' in line:
                try:
                    parts = line.split('librosa.load:')
                    if len(parts) > 1:
                        loading_time = float(parts[1].strip().replace('s', ''))
                        print(f"     - Audio loading (librosa): {parts[1].strip()}")
                except:
                    pass
            elif 'librosa.cqt:' in line:
                try:
                    parts = line.split('librosa.cqt:')
                    if len(parts) > 1:
                        cqt_time = float(parts[1].strip().replace('s', ''))
                        print(f"     - CQT computation: {parts[1].strip()}")
                except:
                    pass
            elif 'TOTAL:' in line:
                try:
                    parts = line.split('TOTAL:')
                    if len(parts) > 1:
                        total_time = float(parts[1].strip().replace('s', ''))
                        print(f"     - TOTAL: {parts[1].strip()}")
                except:
                    pass
        
        # Performance summary
        if loading_time and total_time:
            loading_pct = (loading_time / total_time * 100) if total_time > 0 else 0
            print(f"\n   Summary: Loading {loading_pct:.1f}% | CQT {((total_time - loading_time) / total_time * 100):.1f}% | Total {total_time:.2f}s")

    # Cleanup
    try:
        process.stdin.close()
        process.terminate()
        process.wait(timeout=3)
    except:
        try:
            process.kill()
        except:
            pass

    return {
        'format': format_name,
        'status': result.get('status') if result else 'timeout',
        'loading_time': loading_time,
        'total_time': total_time,
        'backend_info': backend_info,
        'profile_lines': profile_lines
    }


def main():
    """Run tests for all formats."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Test all audio formats")
    parser.add_argument('--exe', action='store_true', help='Test executable instead of Python')
    parser.add_argument('--formats', nargs='+', help='Specific formats to test (e.g., MP3 WAV)')
    args = parser.parse_args()

    use_exe = args.exe
    env_type = "Executable" if use_exe else "Python (pipenv)"
    platform = "Windows" if sys.platform == 'win32' else "macOS"

    print("\n" + "="*70)
    print(f"TESTING ALL AUDIO FORMATS - {env_type} ({platform})")
    print("="*70)
    print(f"\nSupported formats: MP3, MP4, WAV, FLAC, OGG, M4A, AAC, AIFF, AU")
    
    # Find test files
    test_files = find_test_files()
    
    if not test_files:
        print("❌ No test files found! Please ensure you have audio files in ~/Music/")
        return
    
    # Filter formats if specified
    if args.formats:
        test_files = {k: v for k, v in test_files.items() if k.upper() in [f.upper() for f in args.formats]}
        if not test_files:
            print(f"❌ No matching formats found for: {args.formats}")
            return
    
    print(f"Testing {len(test_files)} format(s) with profiling enabled\n")

    results = []
    for format_name, file_path in test_files.items():
        result = test_format(format_name, file_path, use_exe)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nFormats tested: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")

    # Performance summary
    if results:
        print(f"\nPerformance by Format:")
        print(f"{'-'*70}")
        print(f"{'Format':<10} {'Status':<10} {'Loading':<12} {'Total':<12} {'Backend':<15}")
        print(f"{'-'*70}")
        
        for r in results:
            loading_str = f"{r['loading_time']:.3f}s" if r['loading_time'] else "N/A"
            total_str = f"{r['total_time']:.3f}s" if r['total_time'] else "N/A"
            
            backend_parts = []
            if r['backend_info']['pyav_detected']:
                backend_parts.append("PyAV")
            if r['backend_info']['librosa_detected']:
                backend_parts.append("librosa")
            backend_str = "+".join(backend_parts) if backend_parts else "Unknown"
            
            status_icon = "✅" if r['status'] == 'success' else "❌"
            print(f"{r['format']:<10} {status_icon} {r['status']:<8} {loading_str:<12} {total_str:<12} {backend_str:<15}")
        
        print(f"{'-'*70}")
        
        # Average performance
        successful_results = [r for r in results if r['status'] == 'success' and r['total_time']]
        if successful_results:
            avg_loading = sum(r['loading_time'] for r in successful_results if r['loading_time']) / len([r for r in successful_results if r['loading_time']])
            avg_total = sum(r['total_time'] for r in successful_results) / len(successful_results)
            print(f"\nAverage (successful): Loading {avg_loading:.3f}s | Total {avg_total:.3f}s")
            
            # Platform-specific expectations
            if sys.platform == 'win32':
                print(f"\nExpected (Windows): ~1.5-2.0s per file with PyAV, ~20s with audioread")
            else:
                print(f"\nExpected (macOS): ~0.4-0.5s per file with Core Audio")

    # Backend detection summary
    pyav_used = any(r['backend_info']['pyav_detected'] for r in results)
    librosa_used = any(r['backend_info']['librosa_detected'] for r in results)
    ffmpeg_used = any(r['backend_info']['ffmpeg_detected'] for r in results)
    
    print(f"\nBackend Detection:")
    if pyav_used:
        print(f"  ✅ PyAV was used (Windows MP3/MP4/M4A/AAC)")
    if librosa_used:
        print(f"  ✅ librosa was used (all formats on macOS, WAV/FLAC/OGG on Windows)")
    if ffmpeg_used:
        print(f"  ✅ FFmpeg was detected in PATH")
    if not (pyav_used or librosa_used):
        print(f"  ⚠️  No backend detected (may indicate issues)")

    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
