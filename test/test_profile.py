#!/usr/bin/env python3
"""Test script that enables profiling for the executable."""

import subprocess
import os
import sys

exe_path = os.path.abspath("../dist/openkeyscan-analyzer/openkeyscan-analyzer.exe")
test_file = r"C:\Users\Chris\Music\Athys & Duster - Barfight.mp3"

print(f"Testing: {test_file}")
print(f"Executable: {exe_path}")
print(f"\nEnabling PROFILE_PERFORMANCE...\n")

# Set environment variable
env = os.environ.copy()
env['PROFILE_PERFORMANCE'] = '1'

# Start the executable
process = subprocess.Popen(
    [exe_path, '-w', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    env=env  # Pass environment with profiling enabled
)

# Wait for ready message
for line in process.stderr:
    print(f"[stderr] {line.rstrip()}")
    if "Server ready" in line:
        break

# Send request
import json
request = {
    'id': 'profile-test',
    'path': test_file
}

print(f"\nSending request...\n")
process.stdin.write(json.dumps(request) + '\n')
process.stdin.flush()

# Read response and profile output
import threading

def read_stdout():
    for line in process.stdout:
        print(f"[stdout] {line.rstrip()}")

def read_stderr():
    for line in process.stderr:
        print(f"[stderr] {line.rstrip()}")

stdout_thread = threading.Thread(target=read_stdout, daemon=True)
stderr_thread = threading.Thread(target=read_stderr, daemon=True)

stdout_thread.start()
stderr_thread.start()

# Wait a bit for processing
import time
time.sleep(30)

# Close
process.stdin.close()
process.wait(timeout=5)

print("\nDone!")
