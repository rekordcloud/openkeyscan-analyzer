#!/usr/bin/env python3
"""
Simple reset test for quick verification during development.
Tests the server script directly (not the built executable).
"""

import subprocess
import json
import sys
import time
from pathlib import Path

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def test_reset():
    """Quick test of reset functionality."""
    print("Starting server...")

    # Start server directly from Python script
    server_path = Path(__file__).parent.parent / 'openkeyscan_analyzer_server.py'
    process = subprocess.Popen(
        [sys.executable, str(server_path), '--workers', '1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Wait for ready
    print("Waiting for ready signal...")
    while True:
        line = process.stdout.readline()
        if not line:
            print("ERROR: Server terminated")
            return False

        try:
            msg = json.loads(line.strip())
            if msg.get('type') == 'ready':
                print("✓ Server ready\n")
                break
        except json.JSONDecodeError:
            pass

    # Find a test file
    test_file = None
    music_dirs = [
        Path.home() / 'Music' / 'spotify',
        Path.home() / 'Music',
        Path.home() / 'Downloads'
    ]

    for music_dir in music_dirs:
        if music_dir.exists():
            for ext in ['*.mp3', '*.wav', '*.flac']:
                files = list(music_dir.rglob(ext))
                if files:
                    test_file = files[0]
                    break
        if test_file:
            break

    if not test_file:
        print("ERROR: No audio files found for testing")
        process.terminate()
        return False

    print(f"Test file: {test_file.name}\n")

    # Send 3 requests
    print("Sending 3 analysis requests...")
    for i in range(3):
        request = {'id': f'req-{i}', 'path': str(test_file)}
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        print(f"  Sent: req-{i}")

    # Wait a bit for processing to start
    print("\nWaiting 0.5s for processing to start...")
    time.sleep(0.5)

    # Send reset
    print("\nSending reset command...")
    process.stdin.write(json.dumps({'type': 'reset'}) + '\n')
    process.stdin.flush()

    # Check for reset_complete and count any results
    print("Waiting for reset_complete...")
    results_received = []
    reset_ok = False

    for _ in range(50):  # Check for 5 seconds
        try:
            line = process.stdout.readline()
            if not line:
                break

            msg = json.loads(line.strip())

            if msg.get('type') == 'reset_complete':
                print(f"✓ Reset complete (generation: {msg.get('generation')})\n")
                reset_ok = True

            if 'id' in msg and msg['id'].startswith('req-'):
                results_received.append(msg['id'])
                print(f"✗ UNEXPECTED: Received result for {msg['id']}")

        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        time.sleep(0.1)

    # Send new request after reset
    print("\nSending new request after reset...")
    process.stdin.write(json.dumps({'id': 'after-reset', 'path': str(test_file)}) + '\n')
    process.stdin.flush()

    # Wait for the new result
    print("Waiting for new result...")
    new_result_ok = False
    for _ in range(100):  # Wait up to 10 seconds
        try:
            line = process.stdout.readline()
            if not line:
                break

            msg = json.loads(line.strip())

            if msg.get('id') == 'after-reset':
                if msg.get('status') == 'success':
                    print(f"✓ New request completed: {msg.get('camelot')}\n")
                    new_result_ok = True
                else:
                    print(f"✗ New request failed: {msg.get('error')}\n")
                break

        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        time.sleep(0.1)

    # Cleanup
    process.terminate()
    process.wait()

    # Results
    print("=" * 60)
    if reset_ok and len(results_received) == 0 and new_result_ok:
        print("✓ TEST PASSED!")
        print(f"  - Reset acknowledged: YES")
        print(f"  - Stale results discarded: YES (0 received)")
        print(f"  - New request works: YES")
        print("=" * 60)
        return True
    else:
        print("✗ TEST FAILED!")
        print(f"  - Reset acknowledged: {'YES' if reset_ok else 'NO'}")
        print(f"  - Stale results discarded: {'YES' if len(results_received) == 0 else f'NO ({len(results_received)} received)'}")
        print(f"  - New request works: {'YES' if new_result_ok else 'NO'}")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = test_reset()
    sys.exit(0 if success else 1)
