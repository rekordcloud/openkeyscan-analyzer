#!/usr/bin/env python3
"""Test script to compare MP3 vs WAV loading performance."""

import subprocess
import os
import json
import time

exe_path = os.path.abspath("dist/openkeyscan-analyzer/openkeyscan-analyzer.exe")

# Test files
test_files = [
    ("MP3", r"C:\Users\Chris\Music\Athys & Duster - Barfight.mp3"),
    ("WAV", r"C:\Users\Chris\Music\audio formats\Andy C - Workout.wav"),
]

# Set environment variable for profiling
env = os.environ.copy()
env['PROFILE_PERFORMANCE'] = '1'

for format_name, test_file in test_files:
    print(f"\n{'='*70}")
    print(f"Testing {format_name}: {os.path.basename(test_file)}")
    print(f"{'='*70}\n")

    # Start the executable
    process = subprocess.Popen(
        [exe_path, '-w', '1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    # Wait for ready message
    for line in process.stderr:
        if "Server ready" in line:
            break

    # Send request
    request = {'id': 'test', 'path': test_file}
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()

    # Read response and profile output
    start_time = time.time()
    profile_output = []

    # Read stderr for profiling info
    for line in process.stderr:
        if "TOTAL:" in line or "librosa.load:" in line or "librosa.cqt:" in line:
            profile_output.append(line.strip())
        if "TOTAL:" in line:
            break

    # Read stdout for result
    for line in process.stdout:
        msg = json.loads(line.strip())
        if msg.get('id') == 'test':
            elapsed = time.time() - start_time
            print(f"Status: {msg['status']}")
            if msg['status'] == 'success':
                print(f"Key: {msg['camelot']} ({msg['key']})")
            break

    # Print profiling output
    print("\nProfiling:")
    for line in profile_output:
        print(f"  {line}")

    # Cleanup
    process.stdin.close()
    process.terminate()
    process.wait(timeout=2)

print(f"\n{'='*70}")
print("Done!")
