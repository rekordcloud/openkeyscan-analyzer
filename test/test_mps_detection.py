#!/usr/bin/env python3
"""
Quick test to verify MPS device detection.
"""
import subprocess
import sys
import json
import time
from pathlib import Path

# Find project root
project_root = Path(__file__).parent.parent

# Start the server
print("Starting server to test MPS detection...")
process = subprocess.Popen(
    ['pipenv', 'run', 'python', 'openkeyscan_analyzer_server.py'],
    cwd=project_root,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Read stderr to see device configuration
print("\n" + "="*70)
print("SERVER INITIALIZATION OUTPUT:")
print("="*70)

stderr_lines = []
while True:
    line = process.stderr.readline()
    if not line:
        break
    stderr_lines.append(line.rstrip())
    print(line.rstrip())

    # Look for device configuration
    if 'Device:' in line:
        device_line = line.strip()
        print("\n" + "="*70)
        if 'mps' in line.lower():
            print("✅ SUCCESS! MPS device detected and configured!")
        elif 'cpu' in line.lower():
            print("⚠️  Using CPU (MPS not available on this system)")
        elif 'cuda' in line.lower():
            print("✅ Using CUDA GPU")
        print("="*70)

    # Stop reading after we see "ready"
    if 'Server ready' in line or 'ready' in line.lower():
        break

# Clean shutdown
process.stdin.write(json.dumps({'type': 'shutdown'}) + '\n')
process.stdin.flush()
time.sleep(0.5)
process.terminate()
process.wait(timeout=2)

print("\nTest complete!")
