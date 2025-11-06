#!/usr/bin/env python3
"""Debug test for server hanging issue."""

import subprocess
import json
import sys
import os
import time
import threading

# Test file
TEST_FILE = r'C:\Users\Chris\Music\audio formats\Burning Chrome - Christian Smith.mp3'

def test_server():
    """Test the server with one file."""
    if not os.path.exists(TEST_FILE):
        print(f"ERROR: Test file not found: {TEST_FILE}")
        return

    print("Starting server...")
    env = os.environ.copy()
    env['PROFILE_PERFORMANCE'] = '1'

    process = subprocess.Popen(
        [sys.executable, "openkeyscan_analyzer_server_optimized.py", '-w', '1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    # Stderr reading thread
    stderr_lines = []
    stderr_lock = threading.Lock()

    def read_stderr():
        """Capture stderr for profiling info."""
        try:
            for line in process.stderr:
                with stderr_lock:
                    stderr_lines.append(line.rstrip())
                print(f"[STDERR] {line.rstrip()}")
        except:
            pass

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    print("\n1. Waiting for 'ready' message...")
    ready = False
    timeout_ready = time.time() + 5

    for line in process.stdout:
        print(f"   Got line from stdout: {repr(line[:100])}")
        if time.time() > timeout_ready:
            print("   TIMEOUT waiting for ready message!")
            break
        try:
            msg = json.loads(line)
            if msg.get('type') == 'ready':
                ready = True
                print("   ✓ Server ready!")
                break
        except:
            pass

    if not ready:
        print("ERROR: Server did not send ready message")
        process.kill()
        return

    print("\n2. Sending test request for MP3...")
    request = {'id': 'test_mp3', 'path': TEST_FILE}
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()
    print(f"   Request sent: {json.dumps(request)}")

    print("\n3. Waiting for response...")
    response = None
    timeout = time.time() + 30
    line_count = 0

    for line in process.stdout:
        line_count += 1
        print(f"   Line {line_count}: {repr(line[:100])}")

        if time.time() > timeout:
            print(f"   TIMEOUT after 30s (read {line_count} lines)")
            break

        try:
            msg = json.loads(line)
            print(f"   Parsed message: type={msg.get('type')}, id={msg.get('id')}")
            if msg.get('id') == 'test_mp3':
                response = msg
                print("   ✓ Got matching response!")
                break
        except Exception as e:
            print(f"   Parse error: {e}")

    print(f"\n4. Response received: {response is not None}")
    if response:
        print(f"   Status: {response.get('status')}")
        print(f"   Result: {response.get('camelot')} ({response.get('key')})")

    print("\n5. Cleanup...")
    process.terminate()
    try:
        process.wait(timeout=2)
    except:
        process.kill()
    print("Done!")


if __name__ == '__main__':
    test_server()
