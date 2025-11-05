#!/usr/bin/env python3
"""Test MP3 vs WAV loading in Python (not executable)."""

import subprocess
import os
import json
import time

# Test files
test_files = [
    ("MP3", r"C:\Users\Chris\Music\Athys & Duster - Barfight.mp3"),
    ("WAV", r"C:\Users\Chris\Music\audio formats\Andy C - Workout.wav"),
]

for format_name, test_file in test_files:
    print(f"\n{'='*70}")
    print(f"Testing {format_name}: {os.path.basename(test_file)}")
    print(f"{'='*70}\n")

    # Start the server
    env = os.environ.copy()
    env['PROFILE_PERFORMANCE'] = '1'

    process = subprocess.Popen(
        ['pipenv', 'run', 'python', 'openkeyscan_analyzer_server.py', '-w', '1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    # Wait for ready message
    for line in process.stderr:
        print(f"[init] {line.rstrip()}")
        if "Server ready" in line:
            break

    # Send request
    request = {'id': 'test', 'path': test_file}
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()

    # Read response and profile output
    profile_lines = []
    results = []

    import threading
    def read_stderr():
        for line in process.stderr:
            if "librosa.load:" in line or "librosa.cqt:" in line or "TOTAL:" in line:
                profile_lines.append(line.rstrip())

    def read_stdout():
        for line in process.stdout:
            try:
                msg = json.loads(line.strip())
                if msg.get('id') == 'test':
                    results.append(msg)
                    break
            except:
                pass

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread.start()
    stdout_thread.start()

    # Wait for result
    stdout_thread.join(timeout=45)

    if results:
        result = results[0]
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Key: {result['camelot']} ({result['key']})")
    else:
        print("ERROR: No result received")

    # Print profiling output
    print("\nProfiling:")
    for line in profile_lines:
        print(f"  {line}")

    # Cleanup
    process.stdin.close()
    process.terminate()
    process.wait(timeout=2)

print(f"\n{'='*70}")
print("Done!")
