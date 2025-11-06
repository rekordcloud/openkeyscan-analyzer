#!/usr/bin/env python3
"""Simple test to check if CQT works in the bundled executable."""

import subprocess
import json
import sys
import os
import time

# Path to the executable
EXE_PATH = r'C:\Users\Chris\workspace\openkeyscan\MusicalKeyCNN\dist\openkeyscan-analyzer\openkeyscan-analyzer.exe'

if not os.path.exists(EXE_PATH):
    print(f"ERROR: Executable not found: {EXE_PATH}")
    sys.exit(1)

print(f"Starting executable: {EXE_PATH}")
print("Monitoring CQT warmup...")

# Start the server
process = subprocess.Popen(
    [EXE_PATH],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Read stderr in real-time
print("\nStderr output:")
print("-" * 50)
start_time = time.time()
timeout = 30  # 30 second timeout

try:
    while time.time() - start_time < timeout:
        # Check if process has ended
        if process.poll() is not None:
            print(f"\nProcess exited with code: {process.returncode}")
            break

        # Read stderr (non-blocking would be better but this is simpler)
        line = process.stderr.readline()
        if line:
            print(f"[{time.time() - start_time:.1f}s] {line.rstrip()}")

            # Check for key messages
            if "CQT warmup complete" in line:
                print("\n✓ CQT warmup completed successfully!")
            elif "ERROR: CQT warmup failed" in line:
                print("\n✗ CQT warmup failed!")
            elif "type\": \"ready\"" in line:
                print("\n✓ Server is ready!")

        # Also check stdout
        try:
            line = process.stdout.readline()
            if line:
                print(f"[STDOUT] {line.rstrip()}")
                if "ready" in line:
                    print("\n✓ Server sent ready message!")
        except:
            pass

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

# Kill the process
process.terminate()
try:
    process.wait(timeout=2)
except:
    process.kill()

print(f"\nTotal time: {time.time() - start_time:.1f}s")
print("Test completed.")