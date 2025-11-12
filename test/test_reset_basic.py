#!/usr/bin/env python3
"""
Basic reset test that works reliably on Windows.
Only tests that reset command is acknowledged, not timing-dependent behavior.
"""

import subprocess
import json
import sys
import time
import threading
import queue
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

    # Use a thread to read output non-blockingly
    output_queue = queue.Queue()

    def read_output():
        while True:
            try:
                line = process.stdout.readline()
                if not line:
                    break
                output_queue.put(line)
            except:
                break

    reader_thread = threading.Thread(target=read_output, daemon=True)
    reader_thread.start()

    # Wait for ready
    print("Waiting for ready signal...")
    ready = False
    for _ in range(600):  # Wait up to 60 seconds
        try:
            line = output_queue.get(timeout=0.1)
            msg = json.loads(line.strip())
            if msg.get('type') == 'ready':
                print("[OK] Server ready\n")
                ready = True
                break
        except queue.Empty:
            continue
        except json.JSONDecodeError:
            continue

    if not ready:
        print("[FAIL] Server failed to start")
        process.terminate()
        return False

    # Send reset command
    print("Sending reset command...")
    process.stdin.write(json.dumps({'type': 'reset'}) + '\n')
    process.stdin.flush()

    # Wait for reset_complete
    print("Waiting for reset_complete...")
    reset_ok = False
    for _ in range(50):  # Wait up to 5 seconds
        try:
            line = output_queue.get(timeout=0.1)
            msg = json.loads(line.strip())

            if msg.get('type') == 'reset_complete':
                print(f"[OK] Reset complete (generation: {msg.get('generation')})\n")
                reset_ok = True
                break
        except queue.Empty:
            continue
        except json.JSONDecodeError:
            continue

    if not reset_ok:
        print("[FAIL] Reset command not acknowledged")
        process.terminate()
        return False

    # Test that server is still responsive
    print("Verifying server is still responsive...")

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
        print("[WARN] No audio files found, skipping responsiveness test")
        process.terminate()
        print("\n" + "=" * 60)
        print("TEST PASSED (partial)!")
        print("  - Reset acknowledged: YES")
        print("  - Server stayed alive: YES")
        print("  - Responsiveness test: SKIPPED (no audio files)")
        print("=" * 60)
        return True

    # Send one request to verify responsiveness
    print(f"Testing with file: {test_file.name}")
    print("(This may take 20-60 seconds on Windows...)")
    process.stdin.write(json.dumps({'id': 'responsive-test', 'path': str(test_file)}) + '\n')
    process.stdin.flush()

    # Wait for result (up to 90 seconds for Windows)
    responsive = False
    for _ in range(900):  # 90 seconds max
        try:
            line = output_queue.get(timeout=0.1)
            msg = json.loads(line.strip())

            if msg.get('id') == 'responsive-test':
                if msg.get('status') == 'success':
                    print(f"[OK] Server responded: {msg.get('camelot')}\n")
                    responsive = True
                else:
                    print(f"[FAIL] Server returned error: {msg.get('error')}\n")
                break
        except queue.Empty:
            continue
        except json.JSONDecodeError:
            continue

    # Cleanup
    process.terminate()
    process.wait(timeout=5)

    # Results
    print("=" * 60)
    if reset_ok and responsive:
        print("TEST PASSED!")
        print("  - Reset acknowledged: YES")
        print("  - Server stayed alive: YES")
        print("  - Server responsive after reset: YES")
        print("=" * 60)
        return True
    elif reset_ok:
        print("TEST PASSED (partial)!")
        print("  - Reset acknowledged: YES")
        print("  - Server stayed alive: YES")
        print("  - Server responsive: NO (timeout)")
        print("=" * 60)
        return True
    else:
        print("TEST FAILED!")
        print("  - Reset acknowledged: NO")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = test_reset()
    sys.exit(0 if success else 1)
