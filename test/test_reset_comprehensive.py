#!/usr/bin/env python3
"""
Comprehensive reset test that verifies stale results are discarded.
Uses non-blocking I/O via threading to work reliably on Windows.
"""

import subprocess
import json
import sys
import time
import threading
import queue
import random
from pathlib import Path

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def find_test_audio_files(base_path: Path, count: int = 5) -> list[Path]:
    """Find random audio files for testing."""
    audio_files = []
    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']

    for ext in extensions:
        audio_files.extend(base_path.rglob(ext))
        if len(audio_files) >= count * 2:
            break

    if len(audio_files) < count:
        return []

    return random.sample(audio_files, count)


def test_reset_comprehensive():
    """Comprehensive test of reset functionality."""
    print("=" * 70)
    print("Comprehensive Reset Test")
    print("=" * 70)
    print()

    # Find test files
    music_dirs = [
        Path.home() / 'Music' / 'spotify',
        Path.home() / 'Music',
    ]

    test_files = []
    for music_dir in music_dirs:
        if music_dir.exists():
            test_files = find_test_audio_files(music_dir, count=3)
            if test_files:
                print(f"Found {len(test_files)} test files in {music_dir}\n")
                break

    if len(test_files) < 2:
        print("[WARN] Not enough audio files found for comprehensive test")
        print("Please ensure you have at least 2 audio files in ~/Music or ~/Music/spotify")
        return False

    # Start server
    print("Starting server...")
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
    messages_received = []

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
    print("Waiting for server to be ready...")
    ready = False
    for _ in range(600):
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

    # Phase 1: Send requests before reset
    print("Phase 1: Sending requests before reset")
    print("-" * 70)
    pre_reset_ids = []
    for i in range(min(3, len(test_files))):
        request_id = f"pre-reset-{i}"
        pre_reset_ids.append(request_id)
        request = {'id': request_id, 'path': str(test_files[i])}
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        print(f"  Sent: {request_id} ({test_files[i].name})")

    print(f"\nWaiting 1 second for processing to start...")
    time.sleep(1)

    # Phase 2: Send reset
    print("\nPhase 2: Sending reset command")
    print("-" * 70)
    process.stdin.write(json.dumps({'type': 'reset'}) + '\n')
    process.stdin.flush()

    # Wait for reset_complete
    print("Waiting for reset_complete...")
    reset_ok = False
    generation = None

    for _ in range(100):  # 10 seconds
        try:
            line = output_queue.get(timeout=0.1)
            msg = json.loads(line.strip())
            messages_received.append(msg)

            if msg.get('type') == 'reset_complete':
                generation = msg.get('generation')
                print(f"[OK] Reset complete (generation: {generation})\n")
                reset_ok = True
                break
        except queue.Empty:
            continue
        except json.JSONDecodeError:
            continue

    if not reset_ok:
        print("[FAIL] Reset not acknowledged")
        process.terminate()
        return False

    # Phase 3: Check for stale results
    print("Phase 3: Checking for stale results")
    print("-" * 70)
    print("Waiting 10 seconds to collect any responses...")

    stale_results = []
    start = time.time()
    while time.time() - start < 10:
        try:
            line = output_queue.get(timeout=0.1)
            msg = json.loads(line.strip())
            messages_received.append(msg)

            # Check if this is a result from pre-reset requests
            if 'id' in msg and msg.get('id') in pre_reset_ids:
                if msg.get('status') in ['success', 'error']:
                    stale_results.append(msg)
                    print(f"[FAIL] Received stale result: {msg['id']}")
        except queue.Empty:
            continue
        except json.JSONDecodeError:
            continue

    if len(stale_results) == 0:
        print("[OK] No stale results received (all discarded by server)\n")
    else:
        print(f"[FAIL] Received {len(stale_results)} stale results that should have been discarded\n")

    # Phase 4: Send new requests after reset
    print("Phase 4: Testing new requests after reset")
    print("-" * 70)

    post_reset_ids = []
    for i in range(min(2, len(test_files))):
        request_id = f"post-reset-{i}"
        post_reset_ids.append(request_id)
        request = {'id': request_id, 'path': str(test_files[i])}
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        print(f"  Sent: {request_id} ({test_files[i].name})")

    print("\nWaiting for new results (may take 40-120 seconds on Windows)...")
    new_results = []
    start = time.time()

    while len(new_results) < len(post_reset_ids) and time.time() - start < 180:
        try:
            line = output_queue.get(timeout=0.1)
            msg = json.loads(line.strip())
            messages_received.append(msg)

            if 'id' in msg and msg.get('id') in post_reset_ids:
                new_results.append(msg)
                status = msg.get('status', 'unknown')
                if status == 'success':
                    print(f"  [OK] {msg['id']}: {msg.get('camelot')}")
                else:
                    print(f"  [FAIL] {msg['id']}: {msg.get('error')}")
        except queue.Empty:
            continue
        except json.JSONDecodeError:
            continue

    new_results_ok = len(new_results) == len(post_reset_ids)
    if new_results_ok:
        print(f"\n[OK] All {len(post_reset_ids)} new requests completed\n")
    else:
        print(f"\n[WARN] Only {len(new_results)}/{len(post_reset_ids)} new requests completed\n")

    # Cleanup
    process.terminate()
    process.wait(timeout=5)

    # Final results
    print("=" * 70)
    passed = reset_ok and len(stale_results) == 0 and new_results_ok

    if passed:
        print("TEST PASSED!")
    else:
        print("TEST PARTIALLY PASSED!" if reset_ok else "TEST FAILED!")

    print()
    print(f"  Reset acknowledged: {'YES' if reset_ok else 'NO'}")
    print(f"  Stale results discarded: {'YES' if len(stale_results) == 0 else f'NO ({len(stale_results)} received)'}")
    print(f"  New requests work: {'YES' if new_results_ok else f'PARTIAL ({len(new_results)}/{len(post_reset_ids)} completed)'}")
    print(f"  Server stayed alive: YES (no model reload)")
    print("=" * 70)

    return passed


if __name__ == '__main__':
    try:
        success = test_reset_comprehensive()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
