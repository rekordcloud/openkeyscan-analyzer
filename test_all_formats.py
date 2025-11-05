#!/usr/bin/env python3
"""
Test all 9 supported audio formats with profiling.

Tests formats: MP3, MP4, WAV, FLAC, OGG, M4A, AAC, AIFF, AU
Measures performance and identifies which backend is used (soundfile vs audioread+ffmpeg)
"""

import subprocess
import json
import os
import sys
from pathlib import Path

# Test files for each format (using audio formats directory)
TEST_FILES = {
    'MP3': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.mp3',
    'MP4': r'C:\Users\Chris\Music\audio formats\mine.mp4',
    'WAV': r'C:\Users\Chris\Music\audio formats\Andy C - Workout.wav',
    'FLAC': r'C:\Users\Chris\Music\audio formats\Christian Smith - Burning Chrome.flac',
    'OGG': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.ogg',
    'M4A': r'C:\Users\Chris\Music\audio formats\Knife Party - Bonfire.m4a',
    'AAC': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.aac',
    'AIFF': r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.aiff',
    # AU not available, will skip
}

def test_format(format_name, file_path, use_exe=False):
    """Test a single audio format."""

    if not Path(file_path).exists():
        return None

    print(f"\n{'='*70}")
    print(f"Testing {format_name}: {Path(file_path).name}")
    print(f"{'='*70}")

    # Set environment for profiling
    env = os.environ.copy()
    env['PROFILE_PERFORMANCE'] = '1'

    # Choose command (Python or executable)
    if use_exe:
        cmd = [
            os.path.abspath('dist/openkeyscan-analyzer/openkeyscan-analyzer.exe'),
            '-w', '1'
        ]
    else:
        cmd = ['pipenv', 'run', 'python', 'openkeyscan_analyzer_server.py', '-w', '1']

    # Start server
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    # Wait for ready message
    init_messages = []
    for line in process.stderr:
        init_messages.append(line.rstrip())
        if "Server ready" in line:
            break

    # Send request
    request = {'id': 'test', 'path': file_path}
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()

    # Read response and profiling output
    profile_lines = []
    result = None

    # Read stderr for profiling
    for line in process.stderr:
        if any(keyword in line for keyword in ['librosa.load:', 'librosa.cqt:', 'TOTAL:', 'load_audio_pyav:']):
            profile_lines.append(line.rstrip())
        if 'TOTAL:' in line:
            break

    # Read stdout for result
    for line in process.stdout:
        try:
            msg = json.loads(line.strip())
            if msg.get('id') == 'test':
                result = msg
                break
        except:
            pass

    # Check for FFMPEG messages in init
    ffmpeg_detected = any('[FFMPEG]' in msg for msg in init_messages)

    # Print results
    if result and result['status'] == 'success':
        print(f"✅ Status: {result['status']}")
        print(f"   Key: {result['camelot']} - {result['key']}")
    elif result:
        print(f"❌ Status: {result['status']} - {result.get('error', 'Unknown error')}")
    else:
        print("❌ No result received (timeout)")

    # Print FFmpeg detection
    if ffmpeg_detected:
        print(f"   FFmpeg: Detected in PATH")
        for msg in init_messages:
            if '[FFMPEG]' in msg:
                print(f"   {msg}")

    # Print profiling
    if profile_lines:
        print("\n   Profiling:")
        for line in profile_lines:
            # Extract timing info
            if 'librosa.load:' in line:
                print(f"     - Audio loading: {line.split('librosa.load:')[1].strip()}")
            elif 'load_audio_pyav:' in line:
                print(f"     - Audio loading (PyAV): {line.split('load_audio_pyav:')[1].strip()}")
            elif 'librosa.cqt:' in line:
                print(f"     - CQT computation: {line.split('librosa.cqt:')[1].strip()}")
            elif 'TOTAL:' in line:
                print(f"     - TOTAL: {line.split('TOTAL:')[1].strip()}")

    # Cleanup
    process.stdin.close()
    try:
        process.terminate()
        process.wait(timeout=2)
    except:
        process.kill()

    return {
        'format': format_name,
        'status': result['status'] if result else 'timeout',
        'profile_lines': profile_lines,
        'ffmpeg_detected': ffmpeg_detected
    }


def main():
    """Run tests for all formats."""

    use_exe = '--exe' in sys.argv
    env_type = "Executable" if use_exe else "Python (pipenv)"

    print("\n" + "="*70)
    print(f"TESTING ALL AUDIO FORMATS - {env_type}")
    print("="*70)
    print(f"\nSupported formats: MP3, MP4, WAV, FLAC, OGG, M4A, AAC, AIFF, AU")
    print(f"Testing {len(TEST_FILES)} formats with profiling enabled")

    results = []
    for format_name, file_path in TEST_FILES.items():
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

    if any(r['ffmpeg_detected'] for r in results):
        print(f"\n✅ FFmpeg was detected and added to PATH")
    else:
        print(f"\n⚠️ FFmpeg was NOT detected (may use system ffmpeg or fall back to other backends)")

    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
