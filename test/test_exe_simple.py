#!/usr/bin/env python3
"""Simple test to check if the executable's stdout is readable."""

import subprocess
import sys
import time
import threading

import os
exe_path = os.path.abspath("../dist/openkeyscan-analyzer/openkeyscan-analyzer.exe")

print("Starting executable...")
process = subprocess.Popen(
    [exe_path, '-w', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,  # Line buffered
    universal_newlines=True
)

def read_stdout():
    print("\n=== STDOUT ===")
    try:
        for line in process.stdout:
            print(f"STDOUT: {line.rstrip()}")
    except Exception as e:
        print(f"Error reading stdout: {e}")

def read_stderr():
    print("\n=== STDERR ===")
    try:
        for line in process.stderr:
            print(f"STDERR: {line.rstrip()}")
    except Exception as e:
        print(f"Error reading stderr: {e}")

# Start reader threads
stdout_thread = threading.Thread(target=read_stdout, daemon=True)
stderr_thread = threading.Thread(target=read_stderr, daemon=True)

stdout_thread.start()
stderr_thread.start()

# Wait a bit to see if we get the ready message
print("\nWaiting 5 seconds for output...")
time.sleep(5)

print("\nClosing stdin...")
process.stdin.close()

# Wait for process to finish
process.wait(timeout=5)
print(f"\nProcess exited with code {process.returncode}")
