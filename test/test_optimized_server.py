#!/usr/bin/env python3
"""Test the optimized server with PyAV."""

import subprocess
import json
import sys
import os
import time
import threading

# Set profiling environment variable
env = os.environ.copy()
env['PROFILE_PERFORMANCE'] = '1'

# Start the optimized server
print("Starting optimized server with PROFILE_PERFORMANCE=1...")
process = subprocess.Popen(
    [sys.executable, "../openkeyscan_analyzer_server_optimized.py", '-w', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    env=env
)

# Read stderr for profiling info
def read_stderr():
    for line in process.stderr:
        print(f"[SERVER] {line.rstrip()}")

stderr_thread = threading.Thread(target=read_stderr, daemon=True)
stderr_thread.start()

# Wait for ready
ready = False
for line in process.stdout:
    print(f"[STDOUT] {line.rstrip()}")
    try:
        msg = json.loads(line)
        if msg.get('type') == 'ready':
            ready = True
            break
    except:
        pass

if not ready:
    print("Server failed to start")
    sys.exit(1)

print("\nServer ready! Testing MP3 file...")

# Test MP3 file
test_file = r"C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.mp3"
request = {'id': 'test1', 'path': test_file}

print(f"\nSending request for: {test_file}")
start_time = time.time()

process.stdin.write(json.dumps(request) + '\n')
process.stdin.flush()

# Read response
response = None
for line in process.stdout:
    print(f"[STDOUT] {line.rstrip()}")
    try:
        msg = json.loads(line)
        if msg.get('id') == 'test1':
            response = msg
            break
    except:
        pass

elapsed = time.time() - start_time

if response:
    if response['status'] == 'success':
        print(f"\n✅ SUCCESS in {elapsed:.3f}s")
        print(f"  Result: {response['camelot']} ({response['key']})")
    else:
        print(f"\n❌ ERROR: {response.get('error')}")
else:
    print(f"\n❌ No response received")

# Cleanup
process.stdin.close()
process.terminate()
process.wait(timeout=5)

print("\nDone!")